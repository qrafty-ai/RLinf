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

Builds on top of `lerobot.policies.xvla.modeling_xvla.XVLAModel` for core
XVLA model construction and flow-matching generation, while adding RLinf
interfaces (input/output transform, RL log-prob/value computation, and
checkpoint compatibility helpers).
"""

from typing import Any, Literal, Optional, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartTokenizerFast

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.utils.logging import get_logger

# Import from lerobot directly
from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.policies.xvla.modeling_xvla import XVLAModel
from lerobot.policies.xvla.utils import rotate6d_to_axis_angle

from .adapter import XVLAAdapter


class ValueHead(nn.Module):
    """Value head for PPO (predicts scalar value from features).

    Simple MLP that takes pooled visual features and outputs a value estimate.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            [batch_size, input_dim] pooled features

        Returns
        -------
        Tensor
            [batch_size, 1] value estimates
        """
        return self.network(x)


class XVLAForRLActionPrediction(nn.Module, BasePolicy):
    """XVLA model for reinforcement learning action prediction using flow-matching.
    
    Architecture:
    1. Florence2 VLM (frozen): DaViT vision + BART language encoder
    2. SoftPromptedTransformer (trainable): Policy head with domain soft prompts
    3. Flow-matching sampler: Generate actions via iterative denoising
    
    Uses lerobot's XVLA implementation directly for checkpoint compatibility.
    """
    
    def __init__(
        self, 
        config: XVLAConfig, 
        proprio_dim: int = 0,
        config_name: Optional[str] = None,
        domain_id: int = 0,
        add_value_head: bool = False,
        adapter: Optional[XVLAAdapter] = None,
    ):
        """Initialize XVLA model.
        
        Args:
            config: XVLA configuration (from lerobot)
            proprio_dim: Proprioception dimension (0 if not used)
            config_name: Config name for logging
            domain_id: Domain ID for multi-domain training
            add_value_head: Whether to add value head for PPO
            adapter: Optional adapter for input/output transformation
        """
        super().__init__()
        
        self.config = config
        self.logger = get_logger()
        self.proprio_dim = proprio_dim
        self.chunk_size = config.chunk_size
        self.use_proprio = config.use_proprio
        self.config_name = config_name or "xvla"
        self.domain_id = domain_id
        self.num_denoising_steps = config.num_denoising_steps
        self.add_value_head = add_value_head
        self.adapter = adapter

        florence_config = config.get_florence_config()
        xvla_model = XVLAModel(
            config=config,
            florence_config=florence_config,
            proprio_dim=proprio_dim,
        )
        self._xvla_model: XVLAModel
        object.__setattr__(self, "_xvla_model", xvla_model)

        # Keep historical attribute names for checkpoint compatibility.
        self.vlm = xvla_model.vlm
        self.policy_head = xvla_model.transformer
        self.action_space = xvla_model.action_space
        self.dim_action = xvla_model.dim_action
        projection_dim = xvla_model.vlm.config.projection_dim
        
        # 5. Optional value head for PPO
        if add_value_head:
            self.value_head = ValueHead(
                input_dim=projection_dim,
                hidden_dim=config.hidden_size,
            )
        else:
            self.value_head = None
        
        # Initialize BART tokenizer for language instructions.
        self.tokenizer = BartTokenizerFast.from_pretrained(
            config.tokenizer_name,
            padding_side=config.tokenizer_padding_side,
        )
        self.tokenizer_max_length = config.tokenizer_max_length
        self._apply_dtype()
        
        self.logger.info(f"Initialized XVLA model with config: {self.config_name}")
        self.logger.info(f"  Florence2 projection dim: {projection_dim}")
        self.logger.info(f"  Policy head hidden size: {config.hidden_size}")
        self.logger.info(f"  Action dimension: {self.dim_action}")
        self.logger.info(f"  Proprio dimension: {proprio_dim}")
        self.logger.info(f"  Domain ID: {domain_id}")
        self.logger.info(f"  Tokenizer: {config.tokenizer_name} (max_length={config.tokenizer_max_length})")
    
    @classmethod
    def from_lerobot_policy(cls, lerobot_policy, config_name: str = "xvla", add_value_head: bool = False, adapter: Optional[XVLAAdapter] = None):
        """Create XVLAForRLActionPrediction from a LeRobot XVLAPolicy.
        
        This allows loading LeRobot checkpoints directly without conversion.
        
        Args:
            lerobot_policy: LeRobot XVLAPolicy instance (loaded via from_pretrained)
            config_name: Config name for logging
            add_value_head: Whether to add value head for PPO
            adapter: Optional adapter for input/output transformation
            
        Returns:
            XVLAForRLActionPrediction instance with loaded weights from LeRobot policy
        """
        # Create instance without calling __init__ to avoid re-initializing components
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        
        # Copy config and core components from LeRobot policy
        if not hasattr(lerobot_policy, "model") or not isinstance(lerobot_policy.model, XVLAModel):
            raise ValueError("Expected a LeRobot XVLAPolicy with a valid XVLAModel in `model`.")

        xvla_model = lerobot_policy.model
        instance.config = lerobot_policy.config
        instance.logger = get_logger()
        instance.proprio_dim = lerobot_policy.config.max_state_dim if lerobot_policy.config.use_proprio else 0
        instance.chunk_size = lerobot_policy.config.chunk_size
        instance.use_proprio = lerobot_policy.config.use_proprio
        instance.config_name = config_name
        instance.num_denoising_steps = lerobot_policy.config.num_denoising_steps
        instance.add_value_head = add_value_head
        instance.adapter = adapter

        object.__setattr__(instance, "_xvla_model", xvla_model)
        instance.vlm = xvla_model.vlm
        instance.policy_head = xvla_model.transformer
        instance.action_space = xvla_model.action_space
        instance.dim_action = xvla_model.dim_action

        if hasattr(lerobot_policy, "tokenizer"):
            instance.tokenizer = lerobot_policy.tokenizer
        else:
            instance.tokenizer = BartTokenizerFast.from_pretrained(
                lerobot_policy.config.tokenizer_name,
                padding_side=lerobot_policy.config.tokenizer_padding_side,
            )
        instance.tokenizer_max_length = lerobot_policy.config.tokenizer_max_length

        # Set domain_id from config or default
        instance.domain_id = getattr(lerobot_policy.config, "domain_id", 3)

        # Add value head if requested
        if add_value_head:
            projection_dim = xvla_model.vlm.config.projection_dim
            instance.value_head = ValueHead(
                input_dim=projection_dim,
                hidden_dim=lerobot_policy.config.hidden_size,
            )
            instance.value_head.to(dtype=xvla_model._get_target_dtype())
        else:
            instance.value_head = None
        
        instance.logger.info(f"Initialized XVLA model from LeRobot policy: {config_name}")
        instance.logger.info(f"  Florence2 projection dim: {lerobot_policy.config.get_florence_config().projection_dim}")
        instance.logger.info(f"  Policy head hidden size: {lerobot_policy.config.hidden_size}")
        instance.logger.info(f"  Action dimension: {instance.dim_action}")
        instance.logger.info(f"  Proprio dimension: {instance.proprio_dim}")
        instance.logger.info(f"  Domain ID: {instance.domain_id}")
        instance.logger.info(f"  Tokenizer: {lerobot_policy.config.tokenizer_name} (max_length={instance.tokenizer_max_length})")
        
        return instance
    
    def _get_target_dtype(self) -> torch.dtype:
        """Get target dtype from core XVLA model config."""
        return self._xvla_model._get_target_dtype()

    def _apply_dtype(self) -> None:
        """Apply dtype casting to XVLA model and value head."""
        self._xvla_model._apply_dtype()
        if self.value_head is not None:
            self.value_head.to(dtype=self._get_target_dtype())

    def _apply_freezing(self) -> None:
        """Apply freezing to core XVLA model."""
        self._xvla_model._apply_freezing()

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

    def _prepare_domain_id(self, domain_id: Optional[torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
        """Normalize domain ids to [B] long tensor."""
        if domain_id is None:
            return torch.full((batch_size,), self.domain_id, dtype=torch.long, device=device)

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
            pos = action[..., :3]
            rot6d = action[..., 3:9]
            # Convert to float32 before numpy (BFloat16 not supported by numpy)
            # Flatten to (N, 6) for rotation conversion
            original_shape = rot6d.shape
            flat_shape = (-1, 6)
            rot6d_flat = rot6d.reshape(flat_shape)
            rot6d_np = rot6d_flat.detach().cpu().to(torch.float32).numpy() if isinstance(rot6d_flat, torch.Tensor) else rot6d_flat
            axis_angle_np = rotate6d_to_axis_angle(rot6d_np)
            # Reshape back to original dimensions except last which is now 3
            axis_angle = torch.from_numpy(axis_angle_np).to(action.device, action.dtype)
            axis_angle = axis_angle.reshape(*original_shape[:-1], 3)
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
        input_ids_long = cast(torch.LongTensor, input_ids.to(dtype=torch.long))
        pixel_values_float = cast(torch.FloatTensor, pixel_values.to(dtype=self._get_target_dtype()))
        return self._xvla_model.forward_vlm(
            input_ids=input_ids_long,
            pixel_values=pixel_values_float,
            image_mask=image_mask,
        )
    
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
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
                - observations: Processed observations with either:
                    - input_ids: Pre-tokenized text [B, seq_len]
                    - task_descriptions: Raw text strings (list[str]) to be tokenized
                - actions: Ground truth actions

        Returns:
            Dictionary with loss and metrics
        """
        observations = data["observations"]
        actions = data["actions"]

        target_dtype = self._get_target_dtype()
        pixel_values = observations["pixel_values"].to(dtype=target_dtype)
        image_mask = observations.get("image_mask")

        # Handle both pre-tokenized input_ids and raw task_descriptions
        if "input_ids" in observations:
            input_ids = observations["input_ids"]
        elif "task_descriptions" in observations:
            # Tokenize task descriptions
            task_desc = observations["task_descriptions"]
            batch_size = actions.shape[0]

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
            input_ids = tokenized.input_ids.to(actions.device)
        else:
            raise ValueError("observations must contain either 'input_ids' or 'task_descriptions'")

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
        loss_dict = self._xvla_model(
            input_ids=cast(torch.LongTensor, input_ids.to(dtype=torch.long)),
            image_input=cast(torch.FloatTensor, pixel_values),
            image_mask=image_mask,
            domain_id=cast(torch.LongTensor, domain_id.to(dtype=torch.long)),
            proprio=proprio,
            action=action_target,
        )

        total_loss = torch.stack(list(loss_dict.values())).sum()

        metrics = {name: value.detach().item() for name, value in loss_dict.items()}
        metrics["sft_loss"] = total_loss.detach().item()
        return {"loss": total_loss, "metrics": metrics}
    
    def default_forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        forward_inputs: dict[str, Any],
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
        input_ids = observations["input_ids"].to(dtype=torch.long)
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
        return self._xvla_model.forward_vlm(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_mask=image_mask,
        )

    def predict_action_batch(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "eval",
        sampling_params: Optional[dict[str, Any]] = None,
        **kwargs
    ) -> tuple[torch.Tensor, None]:
        """Generate actions for rollout (inference).
        
        Args:
            env_obs: Observation dictionary with images, states, prompts
            mode: "train" or "eval" mode
            sampling_params: Optional sampling parameters (e.g., num_steps)
            
        Returns:
            Tuple of (actions, None)
        """
        transformed_obs = self.input_transform(env_obs)

        pixel_values = transformed_obs["pixel_values"]
        input_ids = transformed_obs["input_ids"]
        image_mask = transformed_obs["image_mask"]

        target_dtype = self._get_target_dtype()

        proprio = transformed_obs["proprio"]
        domain_id = transformed_obs["domain_id"]
        pixel_values = pixel_values.to(dtype=target_dtype)

        self.logger.info(f"proprio: {proprio}, domain_id: {domain_id}")
        self.logger.info(f"pixel shape: {pixel_values.shape}, dtype={pixel_values.dtype}")
        steps = self.num_denoising_steps
        if sampling_params is not None and "num_steps" in sampling_params:
            steps = int(sampling_params["num_steps"])

        with torch.no_grad():
            domain_id_long = cast(torch.LongTensor, domain_id.to(dtype=torch.long))
            actions = self._xvla_model.generate_actions(
                input_ids=input_ids.to(dtype=torch.long),
                image_input=pixel_values,
                image_mask=image_mask,
                domain_id=domain_id_long,
                proprio=proprio,
                steps=steps,
            )

        if self.adapter is None:
            raise RuntimeError(
                "XVLA adapter is not configured. "
                "Please enable the adapter in your config:\n"
                "  xvla:\n"
                "    adapter:\n"
                "      simulator: \"libero\"\n"
                "This is required for proper input/output transformation."
            )

        actions = self.adapter.transform_output(actions)

        self.logger.info(f"shape after processing: {actions.shape}, dtype={actions.dtype}")
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
                self.domain_id,
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
    ) -> dict[str, Any]:
        """Transform environment observations to model format.

        When adapter is used, it returns pre-processed tensors directly.
        Otherwise, processes raw environment observations.

        Args:
            env_obs: Environment observation dictionary with images, states, task_descriptions

        Returns:
            Dictionary with pixel_values, input_ids, attention_mask, proprio, domain_id
        """
        device = next(self.parameters()).device
        target_dtype = self._get_target_dtype()

        # Check if adapter has already processed the observations
        if self.adapter is  None:
            raise ValueError("Adapter not set")

        adapter_output = self.adapter.transform_input(env_obs)

        # Use pre-processed values from adapter
        pixel_values = adapter_output["pixel_values"]
        image_mask = adapter_output["image_mask"]
        proprio = adapter_output["proprio"]
        assert isinstance(pixel_values, torch.Tensor)
        assert isinstance(image_mask, torch.Tensor)
        assert isinstance(proprio, torch.Tensor)
        batch_size = pixel_values.shape[0]

        # Only tokenization and device transfer needed
        task_desc = adapter_output.get("task_descriptions", "")
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

        domain_id = self._prepare_domain_id(env_obs.get("domain_id"), batch_size, device)

        return {
            "pixel_values": pixel_values.to(device=device, dtype=target_dtype),
            "image_mask": image_mask.to(device=device),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "proprio": proprio.to(device=device, dtype=target_dtype),
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
