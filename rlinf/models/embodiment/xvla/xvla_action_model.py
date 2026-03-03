# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""XVLA (Flow-Matching Vision-Language-Action) model for embodied RL.

Uses HuggingFace Transformers Florence2 as the backbone with a custom
SoftPromptedTransformer policy head for flow-matching action generation.
"""

from typing import Any, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartTokenizerFast

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.xvla.configuration_xvla import XVLAConfig
from rlinf.models.embodiment.xvla.soft_transformer import (
    SoftPromptedTransformer,
    ValueHead,
)
from rlinf.models.embodiment.xvla.action_space import build_action_space
from rlinf.utils.logging import get_logger

# Import custom Florence2 implementation (not from transformers to avoid dependency issues)
from rlinf.models.embodiment.xvla.modeling_florence2 import (
    Florence2ForConditionalGeneration,
    Florence2Config,
)


class XVLAForRLActionPrediction(nn.Module, BasePolicy):
    """XVLA model for reinforcement learning action prediction using flow-matching.
    
    Architecture:
    1. Florence2 VLM (frozen): DaViT vision + BART language encoder
    2. SoftPromptedTransformer (trainable): Policy head with domain soft prompts
    3. Flow-matching sampler: Generate actions via iterative denoising
    """
    
    def __init__(self, config: XVLAConfig, proprio_dim: int = 0):
        """Initialize XVLA model.
        
        Args:
            config: XVLA configuration
            proprio_dim: Proprioception dimension (0 if not used)
        """
        super().__init__()
        
        self.config = config
        self.logger = get_logger()
        self.proprio_dim = proprio_dim
        self.chunk_size = config.chunk_size
        self.use_proprio = config.use_proprio
        
        # Build Florence2 config from nested structure
        florence_config = self._build_florence_config()
        
        # 1. Initialize Florence2 VLM
        self.vlm = Florence2ForConditionalGeneration(florence_config)
        
        # 2. Remove unused BART decoder to save memory
        self._remove_unused_decoder()
        
        projection_dim = florence_config.projection_dim

        # 3. Action space for preprocessing/postprocessing
        if config.action_mode.lower() == "auto":
            action_feature = getattr(config, "action_feature", None)
            real_dim = action_feature.shape[-1] if action_feature is not None else config.max_action_dim
            self.action_space = build_action_space(
                config.action_mode.lower(),
                real_dim=real_dim,
                max_dim=config.max_action_dim,
            )
        else:
            self.action_space = build_action_space(config.action_mode.lower())
        self.dim_action = self.action_space.dim_action

        # 4. Initialize SoftPromptedTransformer policy head
        self.policy_head = SoftPromptedTransformer(
            hidden_size=config.hidden_size,
            multi_modal_input_size=projection_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            num_domains=config.num_domains,
            len_soft_prompts=config.len_soft_prompts,
            dim_time=config.dim_time,
            dim_action=self.dim_action,
            dim_propio=proprio_dim,
            max_len_seq=config.max_len_seq,
            use_hetero_proj=config.use_hetero_proj,
        )
        
        # 5. Optional value head for PPO
        if config.add_value_head:
            self.value_head = ValueHead(
                input_dim=projection_dim,
                hidden_dim=config.hidden_size,
            )
        else:
            self.value_head = None
        
        # 6. Apply freezing
        self._apply_freezing()

        # 7. Apply dtype
        self._apply_dtype()

        # 8. Initialize BART tokenizer for language instructions
        self.tokenizer = BartTokenizerFast.from_pretrained(
            config.tokenizer_name,
            padding_side=config.tokenizer_padding_side,
        )
        self.tokenizer_max_length = config.tokenizer_max_length
        
        self.logger.info(f"Initialized XVLA model with config: {config.config_name}")
        self.logger.info(f"  Florence2 projection dim: {projection_dim}")
        self.logger.info(f"  Policy head hidden size: {config.hidden_size}")
        self.logger.info(f"  Action dimension: {self.dim_action}")
        self.logger.info(f"  Proprio dimension: {proprio_dim}")
        self.logger.info(f"  Domain ID: {config.domain_id}")
        self.logger.info(f"  Tokenizer: {config.tokenizer_name} (max_length={config.tokenizer_max_length})")
    
    def _build_florence_config(self) -> Florence2Config:
        """Build Florence2Config from nested config dict."""
        from omegaconf import DictConfig, OmegaConf

        florence_cfg = self.config.florence_config
        if isinstance(florence_cfg, DictConfig):
            config_dict = OmegaConf.to_container(florence_cfg, resolve=True)
        else:
            config_dict = dict(florence_cfg)

        return Florence2Config(**config_dict)
    
    def _remove_unused_decoder(self):
        """Remove BART decoder from Florence2 to save memory.
        
        XVLA only uses the encoder for visual-language features.
        """
        if hasattr(self.vlm, "language_model"):
            lm = self.vlm.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "decoder"):
                del lm.model.decoder
                self.logger.debug("Removed Florence2 decoder")
            if hasattr(lm, "lm_head"):
                del lm.lm_head
                self.logger.debug("Removed Florence2 lm_head")
    
    def _apply_freezing(self):
        """Apply freezing based on config."""
        if self.config.freeze_vision_encoder and hasattr(self.vlm, "vision_tower"):
            for param in self.vlm.vision_tower.parameters():
                param.requires_grad = False

        if self.config.freeze_language_encoder and hasattr(self.vlm, "language_model"):
            lm = self.vlm.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "encoder"):
                for param in lm.model.encoder.parameters():
                    param.requires_grad = False
            if hasattr(lm, "model") and hasattr(lm.model, "shared"):
                for param in lm.model.shared.parameters():
                    param.requires_grad = False

        if not self.config.train_policy_transformer:
            for name, param in self.policy_head.named_parameters():
                if "soft_prompt" not in name:
                    param.requires_grad = False

        if not self.config.train_soft_prompts and hasattr(self.policy_head, "soft_prompt_hub"):
            for param in self.policy_head.soft_prompt_hub.parameters():
                param.requires_grad = False

    def _get_target_dtype(self) -> torch.dtype:
        """Get target dtype from config."""
        if self.config.dtype == "bfloat16":
            return torch.bfloat16
        return torch.float32

    def _apply_dtype(self) -> None:
        """Apply dtype casting to model components."""
        self.to(dtype=self._get_target_dtype())

    def _get_default_image_mask(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Create default all-valid image mask with shape [B, V]."""
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(1)
        return torch.ones(
            pixel_values.shape[0],
            pixel_values.shape[1],
            dtype=torch.bool,
            device=pixel_values.device,
        )

    def _prepare_image_mask(self, image_mask: Optional[torch.Tensor], pixel_values: torch.Tensor) -> torch.Tensor:
        """Normalize image mask to bool [B, V]."""
        if image_mask is None:
            return self._get_default_image_mask(pixel_values)

        if image_mask.dim() > 2:
            reduce_dims = tuple(range(2, image_mask.dim()))
            image_mask = image_mask.any(dim=reduce_dims)

        if image_mask.dim() == 1:
            image_mask = image_mask.unsqueeze(1)

        if image_mask.shape[0] != pixel_values.shape[0]:
            image_mask = image_mask.expand(pixel_values.shape[0], -1)

        if image_mask.shape[1] < pixel_values.shape[1]:
            pad = torch.zeros(
                pixel_values.shape[0],
                pixel_values.shape[1] - image_mask.shape[1],
                dtype=image_mask.dtype,
                device=image_mask.device,
            )
            image_mask = torch.cat([image_mask, pad], dim=1)
        elif image_mask.shape[1] > pixel_values.shape[1]:
            image_mask = image_mask[:, : pixel_values.shape[1]]

        return image_mask.to(device=pixel_values.device, dtype=torch.bool)

    def _prepare_proprio(
        self,
        proprio: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Normalize proprio to [B, proprio_dim] tensor."""
        if proprio is None:
            return torch.zeros(batch_size, self.proprio_dim, device=device, dtype=dtype)

        if isinstance(proprio, np.ndarray):
            proprio = torch.from_numpy(proprio)

        proprio = proprio.to(device=device, dtype=dtype)
        if proprio.dim() == 1:
            proprio = proprio.unsqueeze(0)

        if proprio.shape[0] != batch_size:
            proprio = proprio.expand(batch_size, -1)

        if proprio.shape[-1] < self.proprio_dim:
            pad = torch.zeros(batch_size, self.proprio_dim - proprio.shape[-1], device=device, dtype=dtype)
            proprio = torch.cat([proprio, pad], dim=-1)
        elif proprio.shape[-1] > self.proprio_dim:
            proprio = proprio[..., : self.proprio_dim]

        return proprio

    def _prepare_domain_id(self, domain_id: Optional[torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
        """Normalize domain ids to [B] long tensor."""
        if domain_id is None:
            return torch.full((batch_size,), self.config.domain_id, dtype=torch.long, device=device)

        if not isinstance(domain_id, torch.Tensor):
            domain_id = torch.as_tensor(domain_id, device=device)
        else:
            domain_id = domain_id.to(device=device)

        if domain_id.ndim == 0:
            domain_id = domain_id.expand(batch_size)
        if domain_id.ndim > 1:
            domain_id = domain_id.reshape(domain_id.shape[0], -1)[:, 0]
        if domain_id.shape[0] != batch_size:
            domain_id = domain_id.expand(batch_size)

        return domain_id.to(dtype=torch.long)

    def _extract_env_state_dim(self, env_obs: dict[str, Any]) -> Optional[int]:
        """Best-effort extraction of state dimension from raw env observations."""
        state = env_obs.get("states")
        if state is None:
            return None

        if isinstance(state, np.ndarray):
            if state.ndim == 0:
                return None
            return int(state.shape[-1])
        if isinstance(state, torch.Tensor):
            if state.ndim == 0:
                return None
            return int(state.shape[-1])

        try:
            state_tensor = torch.as_tensor(state)
            if state_tensor.ndim == 0:
                return None
            return int(state_tensor.shape[-1])
        except Exception:
            return None

    def _infer_target_action_dim(self, env_obs: dict[str, Any], action: torch.Tensor) -> int:
        """Infer output action dimension from generic env metadata/state shape."""
        for key in ("action_dim", "expected_action_dim", "control_dim"):
            if key in env_obs and env_obs[key] is not None:
                value = env_obs[key]
                if isinstance(value, torch.Tensor):
                    if value.numel() > 0:
                        return int(value.reshape(-1)[0].item())
                elif isinstance(value, np.ndarray):
                    if value.size > 0:
                        return int(value.reshape(-1)[0])
                else:
                    return int(value)

        state_dim = self._extract_env_state_dim(env_obs)
        if state_dim is None:
            return int(action.shape[-1])

        if state_dim == 8 and action.shape[-1] >= 10:
            return 7
        if 0 < state_dim <= action.shape[-1]:
            return state_dim
        return int(action.shape[-1])

    def _convert_action_to_target_dim(self, action: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Convert/pad/trim action to target dimension without env-specific branching."""
        if target_dim <= 0:
            return action
        if action.shape[-1] == target_dim:
            return action

        if target_dim == 7 and action.shape[-1] >= 10:
            from rlinf.models.embodiment.xvla.rotation_utils import rotation_6d_to_axis_angle

            pos = action[..., :3]
            rot6d = action[..., 3:9]
            axis_angle = rotation_6d_to_axis_angle(rot6d)
            gripper = action[..., 9:10]
            return torch.cat([pos, axis_angle, gripper], dim=-1)

        if target_dim < action.shape[-1]:
            return action[..., :target_dim]

        pad_shape = (*action.shape[:-1], target_dim - action.shape[-1])
        return torch.cat([action, action.new_zeros(pad_shape)], dim=-1)
    
    @property
    def _no_split_modules(self) -> list[str]:
        """Modules that should not be split during FSDP wrapping."""
        return [
            # Florence2 components
            "Florence2VisionModel",
            "Florence2Encoder",
            "Florence2VisionEncoder",
            "Florence2EncoderLayer",
            # Policy head
            "SoftPromptedTransformer",
            "SoftPromptHub",
            "TransformerBlock",
        ]
    
    @property
    def _no_split_names(self) -> list[str]:
        """Parameter names that should not be split."""
        return [
            # Action projections
            "action_encoder",
            "action_decoder",
            "vlm_proj",
            "aux_visual_proj",
            # Soft prompts
            "soft_prompts",
            "soft_prompt_hub",
            # Value head
            "value_head",
        ]
    
    def _get_vlm_features(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        image_mask: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Get visual-language features from Florence2."""
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(1)
        image_mask = self._prepare_image_mask(image_mask, pixel_values)
        return self.forward_vlm(input_ids=input_ids, pixel_values=pixel_values, image_mask=image_mask)
    
    def forward(
        self,
        forward_type: ForwardType = ForwardType.DEFAULT,
        **kwargs
    ) -> dict[str, Any]:
        """Main forward method dispatching to specific forward implementations."""
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        elif forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        elif forward_type == ForwardType.SAC:
            return self.sac_forward(**kwargs)
        else:
            raise NotImplementedError(f"Forward type {forward_type} not supported")
    
    def sft_forward(
        self,
        data: dict[str, Any],
        **kwargs
    ) -> dict[str, Any]:
        """Supervised fine-tuning forward pass.
        
        Args:
            data: Dictionary containing:
                - observations: Processed observations
                - actions: Ground truth actions
                
        Returns:
            Dictionary with loss and metrics
        """
        observations = data["observations"]
        actions = data["actions"]

        target_dtype = self._get_target_dtype()
        pixel_values = observations["pixel_values"].to(dtype=target_dtype)
        input_ids = observations["input_ids"]
        image_mask = observations.get("image_mask")

        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(1)

        batch_size = actions.shape[0]
        image_mask = self._prepare_image_mask(image_mask, pixel_values)
        proprio = self._prepare_proprio(
            observations.get("proprio"),
            batch_size=batch_size,
            device=actions.device,
            dtype=target_dtype,
        )
        domain_id = self._prepare_domain_id(observations.get("domain_id"), batch_size, actions.device)

        action_target = actions.to(dtype=target_dtype)
        enc = self._get_vlm_features(pixel_values, input_ids, image_mask)

        t = (
            torch.rand(1, device=actions.device, dtype=target_dtype)
            + torch.arange(batch_size, device=actions.device, dtype=target_dtype) / batch_size
        ) % (1 - 1e-5)

        action_noisy = torch.randn_like(action_target) * t.view(-1, 1, 1) + action_target * (1 - t).view(-1, 1, 1)
        proprio_m, action_noisy_m = self.action_space.preprocess(proprio, action_noisy)

        pred_action = self.policy_head(
            domain_id=domain_id,
            action_with_noise=action_noisy_m,
            proprio=proprio_m,
            t=t,
            **enc,
        )

        loss_dict = self.action_space.compute_loss(pred_action, action_target)
        total_loss = sum(loss_dict.values())

        metrics = {name: value.detach().item() for name, value in loss_dict.items()}
        metrics["sft_loss"] = total_loss.detach().item()
        return {"loss": total_loss, "metrics": metrics}
    
    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        **kwargs
    ) -> dict[str, Any]:
        """Default RL forward pass for computing log-probs and values.
        
        Args:
            forward_inputs: Dictionary containing:
                - chains: Generated action sequences [batch, num_steps, action_chunk, action_dim]
                - timesteps: Denoising timesteps [batch, num_steps]
                - observations: Processed observations
                
        Returns:
            Dictionary with logprobs, values, entropy
        """
        chains = forward_inputs["chains"]
        timesteps = forward_inputs["timesteps"]
        observations = forward_inputs["observations"]

        target_dtype = self._get_target_dtype()
        batch_size = chains.shape[0]
        num_steps = chains.shape[1]

        pixel_values = observations["pixel_values"].to(dtype=target_dtype)
        input_ids = observations["input_ids"]
        image_mask = self._prepare_image_mask(observations.get("image_mask"), pixel_values)
        proprio = self._prepare_proprio(
            observations.get("proprio"),
            batch_size=batch_size,
            device=chains.device,
            dtype=target_dtype,
        )
        domain_id = self._prepare_domain_id(observations.get("domain_id"), batch_size, chains.device)

        feature_dict = self._get_vlm_features(pixel_values, input_ids, image_mask)
        vlm_features = feature_dict["vlm_features"]
        aux_visual_inputs = feature_dict["aux_visual_inputs"]

        logprobs_list = []
        for i in range(num_steps):
            action_noisy = chains[:, i].to(dtype=target_dtype)
            t = timesteps[:, i].to(dtype=target_dtype)

            proprio_m, action_noisy_m = self.action_space.preprocess(proprio, action_noisy)
            pred_action = self.policy_head(
                domain_id=domain_id,
                action_with_noise=action_noisy_m,
                proprio=proprio_m,
                t=t,
                vlm_features=vlm_features,
                aux_visual_inputs=aux_visual_inputs,
            )

            residual = pred_action - action_noisy_m
            logprob = -0.5 * (residual**2).sum(dim=-1)
            logprobs_list.append(logprob)

        logprobs = torch.stack(logprobs_list, dim=1)

        values = None
        if self.value_head is not None:
            pooled_features = vlm_features.mean(dim=1)
            values = self.value_head(pooled_features).squeeze(-1)

        entropy = torch.zeros(batch_size, device=chains.device)

        return {
            "logprobs": logprobs,
            "values": values,
            "entropy": entropy,
        }
    
    def sac_forward(self, **kwargs) -> dict[str, Any]:
        """Soft Actor-Critic forward pass (placeholder)."""
        raise NotImplementedError("SAC forward not implemented for XVLA")
    
    def forward_vlm(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Encode text and multi-view images via Florence2 encoder.
        """
        batch_size, num_views = pixel_values.shape[:2]
        flat_mask = image_mask.view(-1).to(dtype=torch.bool)
        flat_images = pixel_values.flatten(0, 1)
        num_valid = int(flat_mask.sum().item())
        if num_valid == 0:
            raise ValueError("At least one image view must be valid per batch.")

        valid_images = flat_images[flat_mask]
        valid_feats = self.vlm._encode_image(valid_images)
        tokens_per_view, hidden_dim = valid_feats.shape[1:]

        image_features = valid_feats.new_zeros((batch_size * num_views, tokens_per_view, hidden_dim))
        image_features[flat_mask] = valid_feats
        image_features = image_features.view(batch_size, num_views, tokens_per_view, hidden_dim)
        inputs_embeds = self.vlm.get_input_embeddings()(input_ids)
        merged_embeds, attention_mask = self.vlm._merge_input_ids_with_image_features(
            image_features[:, 0],
            inputs_embeds,
        )

        enc_out = self.vlm.language_model.model.encoder(
            attention_mask=attention_mask,
            inputs_embeds=merged_embeds,
        )[0]

        aux_visual_inputs = image_features[:, 1:].reshape(batch_size, -1, hidden_dim)
        return {"vlm_features": enc_out, "aux_visual_inputs": aux_visual_inputs}

    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "eval",
        sampling_params: Optional[dict] = None,
        **kwargs
    ) -> tuple[torch.Tensor, None]:
        """Generate actions for rollout (inference).
        
        Args:
            env_obs: Observation dictionary with images, states, prompts
            mode: "train" or "eval" mode
            sampling_params: Unused for XVLA (kept for API compatibility)
            
        Returns:
            Tuple of (actions, None)
        """
        transformed_obs = self.input_transform(env_obs)

        pixel_values = transformed_obs["pixel_values"]
        input_ids = transformed_obs["input_ids"]
        image_mask = self._prepare_image_mask(transformed_obs.get("image_mask"), pixel_values)

        batch_size = pixel_values.shape[0]
        target_dtype = self._get_target_dtype()
        device = pixel_values.device

        proprio = self._prepare_proprio(
            transformed_obs.get("proprio"),
            batch_size=batch_size,
            device=device,
            dtype=target_dtype,
        )
        domain_id = self._prepare_domain_id(transformed_obs.get("domain_id"), batch_size, device)
        pixel_values = pixel_values.to(dtype=target_dtype)

        with torch.no_grad():
            enc = self.forward_vlm(input_ids, pixel_values, image_mask)

        actions = torch.zeros(
            batch_size,
            self.chunk_size,
            self.dim_action,
            device=device,
            dtype=target_dtype,
        )
        x1 = torch.randn_like(actions)

        steps = self.config.num_steps
        if sampling_params is not None and "num_steps" in sampling_params:
            steps = int(sampling_params["num_steps"])
        steps = max(1, steps)

        for i in range(steps, 0, -1):
            t = torch.full(
                (batch_size,),
                float(i) / float(steps),
                device=device,
                dtype=actions.dtype,
            )
            t_expanded = t.view(-1, 1, 1)
            x_t = x1 * t_expanded + actions * (1.0 - t_expanded)
            proprio_m, x_t_m = self.action_space.preprocess(proprio, x_t)
            actions = self.policy_head(
                domain_id=domain_id,
                action_with_noise=x_t_m,
                proprio=proprio_m,
                t=t,
                **enc
            )

        actions = self.action_space.postprocess(actions)
        target_dim = self._infer_target_action_dim(env_obs=env_obs, action=actions)
        actions = self._convert_action_to_target_dim(actions, target_dim)
        return actions.to(dtype=torch.float32), None
    
    def sample_actions(
        self,
        observation: Any,
        num_steps: Optional[int] = None,
        noise_level: Optional[float] = None,
        **kwargs
    ) -> torch.Tensor:
        """Sample actions using flow-matching.
        
        This is a simplified interface for direct action sampling.
        """
        actions, _ = self.predict_action_batch(env_obs=observation)
        return actions
    
    def get_log_prob_value(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        chains: torch.Tensor,
        timesteps: torch.Tensor,
        compute_values: bool = False,
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute log-probabilities of actions under current policy.
        
        Args:
            images: List of image tensors [batch, C, H, W]
            img_masks: List of image masks [batch, H, W]
            lang_tokens: Language token IDs [batch, seq_len]
            lang_masks: Language attention masks [batch, seq_len]
            state: Proprioceptive state [batch, state_dim]
            chains: Action sequences [batch, num_steps, action_chunk, action_dim]
            timesteps: Denoising timesteps [batch, num_steps]
            compute_values: Whether to compute value estimates
            
        Returns:
            Tuple of (log_probs, values, entropy)
        """
        stacked_images = torch.stack(images, dim=1)
        if len(img_masks) > 0:
            stacked_mask = torch.stack(img_masks, dim=1)
        else:
            stacked_mask = self._get_default_image_mask(stacked_images)

        observations = {
            "pixel_values": stacked_images,
            "image_mask": stacked_mask,
            "input_ids": lang_tokens,
            "attention_mask": lang_masks,
            "proprio": state,
            "domain_id": torch.full(
                (lang_tokens.shape[0],),
                self.config.domain_id,
                dtype=torch.long,
                device=lang_tokens.device,
            ),
        }
        
        forward_inputs = {
            "chains": chains,
            "timesteps": timesteps,
            "observations": observations,
        }
        
        outputs = self.default_forward(forward_inputs)
        
        return (
            outputs["logprobs"],
            outputs["values"] if outputs["values"] is not None else torch.zeros(chains.shape[0], device=chains.device),
            outputs["entropy"],
        )
    
    def compute_flow_matching_loss(
        self,
        actions: torch.Tensor,
        observations: Any,
        **kwargs
    ) -> torch.Tensor:
        """Compute flow-matching loss for training."""
        data = {
            "observations": observations,
            "actions": actions,
        }
        outputs = self.sft_forward(data)
        return outputs["loss"]
    
    def input_transform(
        self,
        env_obs: dict[str, Any],
        transpose: bool = False
    ) -> dict[str, Any]:
        """Transform environment observations to model format.
        
        Combines obs_processor and input_transform into a single method.
        Handles generic conversion to model-ready tensors.
        
        Args:
            env_obs: Environment observation dictionary with images, states, task_descriptions
            transpose: Whether to transpose dimensions (not used)
            
        Returns:
            Dictionary with pixel_values, input_ids, attention_mask, proprio, domain_id
        """
        del transpose

        device = next(self.parameters()).device
        target_dtype = self._get_target_dtype()

        image_keys = [key for key in env_obs.keys() if "image" in key and env_obs[key] is not None]
        if len(image_keys) == 0:
            raise ValueError("No image observations found in env_obs.")

        images: list[torch.Tensor] = []
        for key in image_keys:
            image = env_obs[key]
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image)
            elif not isinstance(image, torch.Tensor):
                image = torch.as_tensor(image)

            if image.dim() == 3:
                image = image.unsqueeze(0)

            if image.dim() != 4:
                raise ValueError(f"Expected image tensor with rank 4, got shape {tuple(image.shape)} for key '{key}'.")

            if image.shape[-1] == 3:
                image = image.permute(0, 3, 1, 2)

            if image.shape[1] != 3:
                raise ValueError(f"Expected 3 channels, got shape {tuple(image.shape)} for key '{key}'.")

            images.append(image.float())

        batch_size = images[0].shape[0]
        pixel_values = torch.stack(images, dim=1)
        image_mask = torch.ones(batch_size, pixel_values.shape[1], dtype=torch.bool, device=pixel_values.device)

        total_views = self.config.num_images_in_input or pixel_values.shape[1]
        total_views = max(total_views, pixel_values.shape[1])
        if total_views > pixel_values.shape[1]:
            pad_views = total_views - pixel_values.shape[1]
            pad_images = pixel_values.new_zeros((batch_size, pad_views, *pixel_values.shape[2:]))
            pad_mask = image_mask.new_zeros((batch_size, pad_views))
            pixel_values = torch.cat([pixel_values, pad_images], dim=1)
            image_mask = torch.cat([image_mask, pad_mask], dim=1)

        if tuple(pixel_values.shape[-2:]) != (224, 224):
            pixel_values = F.interpolate(
                pixel_values.flatten(0, 1),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            ).unflatten(0, (batch_size, pixel_values.shape[1]))

        pixel_values = pixel_values.to(device=device, dtype=target_dtype)
        image_mask = image_mask.to(device=device, dtype=torch.bool)

        task_desc = env_obs.get("task_descriptions", "")
        if isinstance(task_desc, str):
            task_desc = [task_desc] * batch_size
        elif isinstance(task_desc, list):
            if len(task_desc) == 1 and batch_size > 1:
                task_desc = task_desc * batch_size
        else:
            task_desc = [str(task_desc)] * batch_size

        tokenized = self.tokenizer(
            task_desc,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_max_length,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids.to(device)
        attention_mask = tokenized.attention_mask.to(device)

        state = env_obs.get("states")
        proprio = self._prepare_proprio(
            state,
            batch_size=batch_size,
            device=device,
            dtype=target_dtype,
        )
        domain_id = self._prepare_domain_id(env_obs.get("domain_id"), batch_size, device)

        return {
            "pixel_values": pixel_values,
            "image_mask": image_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "proprio": proprio,
            "domain_id": domain_id,
        }
    
    def output_transform(self, outputs: dict[str, Any]) -> dict[str, Any]:
        """Transform model outputs to environment format."""
        return outputs
    
    def precision_processor(self, processed_obs: dict[str, Any]) -> dict[str, Any]:
        """Process precision/dtype of observations."""
        dtype = torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float32
        device = next(self.parameters()).device
        
        for key, value in processed_obs.items():
            if isinstance(value, torch.Tensor):
                if key in {"input_ids", "attention_mask", "domain_id"}:
                    processed_obs[key] = value.to(device=device)
                else:
                    processed_obs[key] = value.to(dtype=dtype, device=device)
        
        return processed_obs
    
    def load_checkpoint(self, checkpoint_path: str, strict: bool = True):
        """Load checkpoint weights.
        
        Args:
            checkpoint_path: Path to checkpoint file (.safetensors or .bin)
            strict: Whether to strictly enforce matching keys
        """
        if checkpoint_path.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
                state_dict = load_file(checkpoint_path)
            except ImportError:
                raise ImportError("safetensors is required to load .safetensors files")
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")

        remapped_state_dict: dict[str, torch.Tensor] = {}
        model_state_dict = self.state_dict()

        alias_map = {
            # Legacy/LeRobot naming → RLinf DomainAwareLinear naming
            "policy_head.action_in_proj.weight": "policy_head.action_encoder.fc.weight",
            "policy_head.action_in_proj.bias": "policy_head.action_encoder.bias.weight",
            "policy_head.action_out_proj.weight": "policy_head.action_decoder.fc.weight",
            "policy_head.action_out_proj.bias": "policy_head.action_decoder.bias.weight",
            # For non-hetero (nn.Linear) case
            "policy_head.input_proj.weight": "policy_head.vlm_proj.weight",
            "policy_head.input_proj.bias": "policy_head.vlm_proj.bias",
            # For hetero (DomainAwareLinear) case
            "policy_head.input_proj.fc.weight": "policy_head.vlm_proj.fc.weight",
            "policy_head.input_proj.bias.weight": "policy_head.vlm_proj.bias.weight",
            "policy_head.soft_prompt_hub.soft_prompts": "policy_head.soft_prompt_hub.weight",
        }

        def _map_key(key: str) -> str | None:
            mapped = key
            if mapped.startswith("model."):
                mapped = mapped[6:]
                if mapped.startswith("transformer."):
                    mapped = "policy_head." + mapped[len("transformer.") :]

            mapped = alias_map.get(mapped, mapped)

            if ".mlp.0." in mapped:
                mapped = mapped.replace(".mlp.0.", ".mlp.fc1.")
            if ".mlp.2." in mapped:
                mapped = mapped.replace(".mlp.2.", ".mlp.fc2.")
            if ".mlp.3." in mapped:
                mapped = mapped.replace(".mlp.3.", ".mlp.fc2.")

            # Handle DomainAwareLinear bias (Embedding layer stores as .bias.weight)
            if mapped.startswith("policy_head.action_encoder.bias") and not mapped.endswith(".weight"):
                mapped = mapped.replace("policy_head.action_encoder.bias", "policy_head.action_encoder.bias.weight")
            if mapped.startswith("policy_head.action_decoder.bias") and not mapped.endswith(".weight"):
                mapped = mapped.replace("policy_head.action_decoder.bias", "policy_head.action_decoder.bias.weight")
            # Handle nn.Linear bias for non-hetero vlm_proj
            if mapped.startswith("policy_head.vlm_proj.bias") and not mapped.endswith(".weight") and "fc" not in mapped:
                mapped = mapped.replace("policy_head.vlm_proj.bias", "policy_head.vlm_proj.bias")
            # Handle DomainAwareLinear bias for hetero vlm_proj
            if mapped.startswith("policy_head.vlm_proj.bias") and not mapped.endswith(".weight") and "fc" in mapped:
                mapped = mapped.replace("policy_head.vlm_proj.bias.weight", "policy_head.vlm_proj.bias.weight")

            if mapped == "policy_head.soft_prompt_hub.soft_prompts":
                mapped = "policy_head.soft_prompt_hub.weight"

            return mapped

        shape_mismatch: list[tuple[str, torch.Size, torch.Size]] = []
        unexpected_source: list[str] = []

        for key, value in state_dict.items():
            new_key = _map_key(key)
            if new_key is None:
                continue

            if new_key == "policy_head.soft_prompt_hub.weight" and value.dim() == 3:
                value = value.reshape(value.shape[0], -1)

            if new_key in model_state_dict:
                expected_shape = model_state_dict[new_key].shape
                if value.shape != expected_shape:
                    shape_mismatch.append((new_key, value.shape, expected_shape))
                    continue

                remapped_state_dict[new_key] = value
            else:
                unexpected_source.append(key)

        # Fill shared embedding when only encoder embedding exists in checkpoint.
        shared_key = "vlm.language_model.model.shared.weight"
        encoder_embed_key = "vlm.language_model.model.encoder.embed_tokens.weight"
        if shared_key in model_state_dict and shared_key not in remapped_state_dict:
            if encoder_embed_key in remapped_state_dict:
                remapped_state_dict[shared_key] = remapped_state_dict[encoder_embed_key]

        if shape_mismatch:
            preview = ", ".join(
                [f"{name}: ckpt={tuple(src)} model={tuple(dst)}" for name, src, dst in shape_mismatch[:5]]
            )
            raise RuntimeError(
                f"Checkpoint has shape mismatches for {len(shape_mismatch)} parameters. {preview}"
            )

        missing_keys, unexpected_keys = self.load_state_dict(remapped_state_dict, strict=strict)

        if strict and (missing_keys or unexpected_keys):
            raise RuntimeError(
                f"Strict load failed. Missing={len(missing_keys)} Unexpected={len(unexpected_keys)}"
            )

        if unexpected_source:
            self.logger.warning(
                "Ignored %d source keys that do not map to model parameters",
                len(unexpected_source),
            )

        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")

        loaded_count = len(remapped_state_dict)
        total_count = len(model_state_dict)
        self.logger.info(f"Loaded {loaded_count}/{total_count} parameters ({100*loaded_count//total_count}%)")
