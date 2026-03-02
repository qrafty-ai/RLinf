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
from rlinf.models.embodiment.xvla.action_space import ActionHub
from rlinf.models.embodiment.xvla.flow_matching import FlowMatchingSampler
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
        
        # Build Florence2 config from nested structure
        florence_config = self._build_florence_config()
        
        # 1. Initialize Florence2 VLM
        self.vlm = Florence2ForConditionalGeneration(florence_config)
        
        # 2. Remove unused BART decoder to save memory
        self._remove_unused_decoder()
        
        # 3. Initialize SoftPromptedTransformer policy head
        projection_dim = florence_config.projection_dim
        # Policy head expects full action chunk, not single action
        total_action_dim = config.chunk_size * config.max_action_dim
        self.policy_head = SoftPromptedTransformer(
            hidden_size=config.hidden_size,
            multi_modal_input_size=projection_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            num_domains=config.num_domains,
            len_soft_prompts=config.len_soft_prompts,
            dim_time=config.dim_time,
            dim_action=total_action_dim,
            dim_proprio=proprio_dim,
        )
        
        # 4. Action space for preprocessing/postprocessing
        self.action_space = ActionHub.build(config.action_mode)
        
        # 5. Flow-matching sampler
        self.flow_sampler = FlowMatchingSampler(
            num_steps=config.num_steps,
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max,
            rho=config.rho,
            time_schedule=config.time_schedule,
        )
        
        # 6. Optional value head for PPO
        if config.add_value_head:
            self.value_head = ValueHead(
                input_dim=projection_dim,
                hidden_dim=config.hidden_size,
            )
        else:
            self.value_head = None
        
        # 7. Apply freezing
        self._apply_freezing()
        
        # 8. Initialize BART tokenizer for language instructions
        self.tokenizer = BartTokenizerFast.from_pretrained(
            config.tokenizer_name,
            padding_side=config.tokenizer_padding_side,
        )
        self.tokenizer_max_length = config.tokenizer_max_length
        
        self.logger.info(f"Initialized XVLA model with config: {config.config_name}")
        self.logger.info(f"  Florence2 projection dim: {projection_dim}")
        self.logger.info(f"  Policy head hidden size: {config.hidden_size}")
        self.logger.info(f"  Action dimension: {config.max_action_dim}")
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
        if hasattr(self.vlm, "model"):
            model = self.vlm.model
            if hasattr(model, "decoder"):
                del model.decoder
                self.logger.debug("Removed Florence2 decoder")
        
        # Remove lm_head if present
        if hasattr(self.vlm, "lm_head"):
            del self.vlm.lm_head
            self.logger.debug("Removed Florence2 lm_head")
    
    def _apply_freezing(self):
        """Apply freezing based on config."""
        # Freeze vision encoder
        if self.config.freeze_vision_encoder:
            if hasattr(self.vlm, "vision_tower"):
                for param in self.vlm.vision_tower.parameters():
                    param.requires_grad = False
                self.logger.debug("Frozen vision encoder")
        
        # Freeze language encoder
        if self.config.freeze_language_encoder:
            if hasattr(self.vlm, "model") and hasattr(self.vlm.model, "encoder"):
                for param in self.vlm.model.encoder.parameters():
                    param.requires_grad = False
                self.logger.debug("Frozen language encoder")
    
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
            "action_in_proj",
            "action_out_proj",
            # Time embeddings
            "time_mlp_in",
            "time_mlp_out",
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
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get visual-language features from Florence2.
        
        Args:
            pixel_values: Image pixels [batch, num_images, C, H, W]
            input_ids: Language token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            
        Returns:
            Visual-language features [batch, seq_len, projection_dim]
        """
        if not hasattr(self, "_debug_vlm_shapes_logged"):
            self._debug_vlm_shapes_logged = True
            self.logger.info(f"_get_vlm_features pixel_values shape: {tuple(pixel_values.shape)}")
            self.logger.info(f"_get_vlm_features input_ids shape: {tuple(input_ids.shape)}")

        # pixel_values shape: [batch, num_images, C, H, W]
        # Florence2 custom implementation expects 4D image tensor, so we use
        # the first view for now to keep compatibility and stability.
        if pixel_values.dim() == 5:
            pixel_values = pixel_values[:, 0]

        # Encode image tokens with Florence2 custom image encoder
        image_features = self.vlm._encode_image(pixel_values)

        # Get text token embeddings
        text_embeds = self.vlm.get_input_embeddings()(input_ids)

        # Merge image and text tokens and attention masks
        combined_embeds, combined_attention_mask = self.vlm._merge_input_ids_with_image_features(
            image_features,
            text_embeds,
        )
        if attention_mask is not None:
            combined_attention_mask[:, image_features.shape[1] :] = attention_mask

        # Run encoder-only path
        encoder_outputs = self.vlm.language_model.model.encoder(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
        )

        return encoder_outputs.last_hidden_state
    
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
        actions = data["actions"]  # [batch, action_chunk, action_dim]
        
        # Get VLM features
        pixel_values = observations["pixel_values"]
        input_ids = observations["input_ids"]
        attention_mask = observations["attention_mask"]
        
        vlm_features = self._get_vlm_features(pixel_values, input_ids, attention_mask)
        
        # Compute flow-matching loss
        # For SFT, we train the policy head to predict the flow
        batch_size = actions.shape[0]
        action_dim = actions.shape[-1]
        
        # Sample time
        t = torch.rand(batch_size, device=actions.device)
        
        # Sample noise
        z0 = torch.randn_like(actions)
        
        # Interpolate
        t_expanded = t.view(-1, 1, 1)
        z_t = (1 - t_expanded) * z0 + t_expanded * actions
        
        # Ground truth vector field
        u_t = actions - z0
        
        # Predict vector field
        proprio = observations.get("proprio", None)
        v_t = self.policy_head(
            z_t=z_t.reshape(batch_size, -1),  # Flatten action chunk
            t=t,
            multi_modal_features=vlm_features,
            proprio=proprio,
        )
        
        v_t = v_t.reshape_as(actions)
        
        # MSE loss
        loss = F.mse_loss(v_t, u_t)
        
        return {
            "loss": loss,
            "metrics": {"sft_loss": loss.item()},
        }
    
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
        chains = forward_inputs["chains"]  # [batch, num_steps, action_chunk, action_dim]
        timesteps = forward_inputs["timesteps"]  # [batch, num_steps]
        observations = forward_inputs["observations"]
        
        batch_size = chains.shape[0]
        num_steps = chains.shape[1]
        
        # Get VLM features (shared across all timesteps)
        pixel_values = observations["pixel_values"]
        input_ids = observations["input_ids"]
        attention_mask = observations["attention_mask"]
        proprio = observations.get("proprio", None)
        
        vlm_features = self._get_vlm_features(pixel_values, input_ids, attention_mask)
        
        # Compute log-probs by evaluating the flow at each timestep
        logprobs_list = []
        for i in range(num_steps):
            z_t = chains[:, i]  # [batch, action_chunk, action_dim]
            t = timesteps[:, i]  # [batch]
            
            # Flatten action for policy head
            z_t_flat = z_t.reshape(batch_size, -1)
            
            # Predict vector field
            v_t = self.policy_head(
                z_t=z_t_flat,
                t=t,
                multi_modal_features=vlm_features,
                proprio=proprio,
            )
            
            # Reshape back
            v_t = v_t.reshape_as(z_t)
            
            # Log-prob is related to the vector field magnitude
            # For simplicity, use negative squared error
            logprob = -0.5 * (v_t ** 2).sum(dim=-1)  # [batch, action_chunk]
            logprobs_list.append(logprob)
        
        logprobs = torch.stack(logprobs_list, dim=1)  # [batch, num_steps, action_chunk]
        
        # Compute values if value head exists
        values = None
        if self.value_head is not None:
            # Pool VLM features
            pooled_features = vlm_features.mean(dim=1)  # [batch, hidden_dim]
            values = self.value_head(pooled_features).squeeze(-1)  # [batch]
        
        # Compute entropy (simplified)
        entropy = torch.zeros(batch_size, device=chains.device)
        
        return {
            "logprobs": logprobs,
            "values": values,
            "entropy": entropy,
        }
    
    def sac_forward(self, **kwargs) -> dict[str, Any]:
        """Soft Actor-Critic forward pass (placeholder)."""
        raise NotImplementedError("SAC forward not implemented for XVLA")
    
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "eval",
        sampling_params: Optional[dict] = None,
        **kwargs
    ) -> dict[str, torch.Tensor]:
        """Generate actions for rollout (inference).
        
        Args:
            env_obs: Observation dictionary with images, states, prompts
            mode: "train" or "eval" mode
            sampling_params: Unused for XVLA (kept for API compatibility)
            
        Returns:
            Dictionary with actions
        """
        # Process observations through input_transform (merged obs_processor + transform)
        transformed_obs = self.input_transform(env_obs)
        
        # Extract transformed inputs
        pixel_values = transformed_obs["pixel_values"]
        input_ids = transformed_obs["input_ids"]
        attention_mask = transformed_obs["attention_mask"]
        proprio = transformed_obs.get("proprio", None)
        
        with torch.no_grad():
            vlm_features = self._get_vlm_features(pixel_values, input_ids, attention_mask)
        
        # Sample actions using flow-matching
        batch_size = pixel_values.shape[0]
        action_dim = self.config.max_action_dim * self.config.chunk_size
        device = pixel_values.device
        
        def vector_field_fn(z_t: torch.Tensor, t: torch.Tensor, cond=None) -> torch.Tensor:
            """Vector field function for flow-matching."""
            return self.policy_head(
                z_t=z_t,
                t=t,
                multi_modal_features=vlm_features,
                proprio=proprio,
            )
        
        # Sample using flow-matching
        actions_flat = self.flow_sampler.sample(
            vector_field_fn=vector_field_fn,
            batch_size=batch_size,
            action_dim=action_dim,
            device=device,
        )
        
        # Reshape to [batch, action_chunk, action_dim]
        actions = actions_flat.reshape(batch_size, self.config.chunk_size, -1)
        
        # Postprocess
        actions_np = self.action_space.postprocess(actions)
        
        actions_tensor = torch.from_numpy(actions_np).to(device)
        return actions_tensor, None
    
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
        # Build observations dict
        observations = {
            "pixel_values": torch.stack(images, dim=1),  # [batch, num_images, C, H, W]
            "input_ids": lang_tokens,
            "attention_mask": lang_masks,
            "proprio": state if self.proprio_dim > 0 else None,
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
        
        Combines obs_processor and input_transform into single method.
        Handles conversion from environment-specific state formats to LeRobot format.
        Currently supports:
          - LIBERO: 8D [pos(3), axis_angle(3), gripper(2)] → 20D LeRobot format
          - LeRobot: 20D [pos(3), rot6d(6), gripper(1), zeros(10)] (passthrough)
        
        Args:
            env_obs: Environment observation dictionary with images, states, task_descriptions
            transpose: Whether to transpose dimensions (not used)
            
        Returns:
            Dictionary with pixel_values, input_ids, attention_mask, proprio, domain_id
        """
        device = next(self.parameters()).device
        
        # === Step 1: Extract and process images ===
        if "images" in env_obs:
            images = env_obs["images"]
        elif "main_images" in env_obs:
            # LIBERO format: main_images and wrist_images
            images = [env_obs["main_images"]]
            if "wrist_images" in env_obs:
                images.append(env_obs["wrist_images"])
        elif "main_image" in env_obs:
            images = [env_obs["main_image"]]
            if "wrist_image" in env_obs:
                images.append(env_obs["wrist_image"])
        else:
            raise ValueError(f"No images found in observation. Keys: {list(env_obs.keys())}")
        
        # Infer batch size from images
        if isinstance(images[0], torch.Tensor):
            batch_size = images[0].shape[0]
        elif isinstance(images[0], np.ndarray):
            batch_size = images[0].shape[0]
        else:
            batch_size = 1

        if not hasattr(self, "_debug_input_transform_logged"):
            self._debug_input_transform_logged = True
            first_shape = tuple(images[0].shape) if hasattr(images[0], "shape") else type(images[0])
            self.logger.info(f"input_transform first image shape: {first_shape}")
        
        # Stack images and normalize tensor layout to [batch, num_views, 3, H, W]
        if isinstance(images[0], torch.Tensor):
            pixel_values = torch.stack(images, dim=1)
        else:
            pixel_values = torch.from_numpy(np.stack(images, axis=0)).float()
            pixel_values = pixel_values.unsqueeze(1)

        # Handle channel-last inputs, e.g. [B, V, H, W, 3] -> [B, V, 3, H, W]
        if pixel_values.dim() == 5 and pixel_values.shape[-1] == 3:
            pixel_values = pixel_values.permute(0, 1, 4, 2, 3)

        # Handle single-view channel-last [B, H, W, 3] -> [B, 1, 3, H, W]
        if pixel_values.dim() == 4 and pixel_values.shape[-1] == 3:
            pixel_values = pixel_values.permute(0, 3, 1, 2).unsqueeze(1)

        # Resize to 224x224 if needed
        if pixel_values.shape[-1] != 224:
            pixel_values = F.interpolate(
                pixel_values.flatten(0, 1),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            ).unflatten(0, (batch_size, -1))

        # Ensure image tensor is on the same device as model weights
        pixel_values = pixel_values.to(device)
        
        # === Step 2: Extract and tokenize language ===
        task_desc = env_obs.get("task_descriptions", env_obs.get("language_instruction", ""))
        if isinstance(task_desc, str):
            task_desc = [task_desc] * batch_size
        
        tokenized = self.tokenizer(
            task_desc,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_max_length,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids.to(device)
        attention_mask = tokenized.attention_mask.to(device)
        
        # === Step 3: Extract and process proprioception ===
        state = env_obs.get("states", env_obs.get("agent_pos", None))
        
        if state is not None and self.proprio_dim > 0:
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float().to(device)
            elif isinstance(state, torch.Tensor):
                state = state.to(device)
            
            # Detect state format and convert if needed
            state_dim = state.shape[-1]
            
            if state_dim == 8:
                # LIBERO format: [pos(3), axis_angle(3), gripper(2)]
                proprio = self._convert_libero_state_to_lerobot(state)
            elif state_dim == 20:
                # Already LeRobot format
                proprio = state
            else:
                # Unknown format - pad as needed
                self.logger.warning(f"Unknown state dimension {state_dim}, using as-is")
                if state_dim < self.proprio_dim:
                    padding = torch.zeros(batch_size, self.proprio_dim - state_dim, device=device)
                    proprio = torch.cat([state, padding], dim=-1)
                else:
                    proprio = state[:, :self.proprio_dim]
            
            # Ensure correct shape
            if proprio.dim() == 1:
                proprio = proprio.unsqueeze(0)
            if proprio.shape[-1] < self.proprio_dim:
                padding = torch.zeros(batch_size, self.proprio_dim - proprio.shape[-1], device=device)
                proprio = torch.cat([proprio, padding], dim=-1)
        else:
            proprio = None
        
        # === Step 4: Add domain ID ===
        domain_id = torch.full((batch_size,), self.config.domain_id, dtype=torch.long, device=device)
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "proprio": proprio,
            "domain_id": domain_id,
        }
    
    def _convert_libero_state_to_lerobot(self, libero_state: torch.Tensor) -> torch.Tensor:
        """Convert LIBERO 8D state to LeRobot 20D format.
        
        Args:
            libero_state: [batch, 8] = [pos(3), axis_angle(3), gripper(2)]
            
        Returns:
            lerobot_state: [batch, 20] = [pos(3), rot6d(6), gripper(1), zeros(10)]
        """
        from rlinf.models.embodiment.xvla.rotation_utils import axis_angle_to_rotation_6d
        
        # Extract components
        pos = libero_state[..., :3]  # [batch, 3]
        axis_angle = libero_state[..., 3:6]  # [batch, 3]
        gripper = libero_state[..., 6:8]  # [batch, 2]
        
        # Convert axis-angle to 6D rotation
        rot6d = axis_angle_to_rotation_6d(axis_angle)  # [batch, 6]
        
        # Use first gripper finger (or mean)
        gripper_1d = gripper[..., :1]  # [batch, 1]
        
        # Build 10D LeRobot state
        state_10d = torch.cat([pos, rot6d, gripper_1d], dim=-1)  # [batch, 10]
        
        # Zero-pad to 20D
        padding = torch.zeros(*state_10d.shape[:-1], 10, device=state_10d.device)
        state_20d = torch.cat([state_10d, padding], dim=-1)  # [batch, 20]
        
        return state_20d
    
    def output_transform(self, outputs: dict[str, Any]) -> dict[str, Any]:
        """Transform model outputs to environment format."""
        return outputs
    
    def precision_processor(self, processed_obs: dict[str, Any]) -> dict[str, Any]:
        """Process precision/dtype of observations."""
        dtype = torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float32
        device = next(self.parameters()).device
        
        for key, value in processed_obs.items():
            if isinstance(value, torch.Tensor):
                processed_obs[key] = value.to(dtype=dtype, device=device)
        
        return processed_obs
    
    def load_checkpoint(self, checkpoint_path: str, strict: bool = True):
        """Load checkpoint weights.
        
        Args:
            checkpoint_path: Path to checkpoint file (.safetensors or .bin)
            strict: Whether to strictly enforce matching keys
        """
        import os
        
        if checkpoint_path.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
                state_dict = load_file(checkpoint_path)
            except ImportError:
                raise ImportError("safetensors is required to load .safetensors files")
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
        
        # Load state dict
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=strict)
        
        if missing_keys:
            self.logger.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            self.logger.warning(f"Unexpected keys: {unexpected_keys}")
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
