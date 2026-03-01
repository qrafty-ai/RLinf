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
        
        self.logger.info(f"Initialized XVLA model with config: {config.config_name}")
        self.logger.info(f"  Florence2 projection dim: {projection_dim}")
        self.logger.info(f"  Policy head hidden size: {config.hidden_size}")
        self.logger.info(f"  Action dimension: {config.max_action_dim}")
        self.logger.info(f"  Proprio dimension: {proprio_dim}")
    
    def _build_florence_config(self) -> Florence2Config:
        """Build Florence2Config from nested config dict."""
        config_dict = dict(self.config.florence_config)
        
        # Ensure vision_config and text_config are properly nested
        if "vision_config" in config_dict and isinstance(config_dict["vision_config"], dict):
            pass  # Already nested
        
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
        # Note: pixel_values should be [batch, num_images, C, H, W]
        # The Florence2Model expects pixel_values with batch dimension matching input_ids
        # Each image in the batch is processed separately by the vision encoder
        batch_size = pixel_values.shape[0]
        
        # Forward through Florence2 model encoder only (skip decoder for XVLA)
        # First get image embeddings from vision model
        image_embeds = self.vlm.model.vision_model(pixel_values)
        if self.vlm.model.vision_projection is not None:
            image_embeds = self.vlm.model.vision_projection(image_embeds)
        
        # Expand to sequence length if needed (vision model returns [batch, dim])
        if len(image_embeds.shape) == 2:
            image_embeds = image_embeds.unsqueeze(1)  # [batch, 1, dim]
        
        # Get text embeddings
        text_embeds = self.vlm.model.language_model.encoder.embed_tokens(input_ids) * \
                      self.vlm.model.language_model.encoder.embed_scale
        
        # Concatenate image and text embeddings
        combined_embeds = torch.cat([image_embeds, text_embeds], dim=1)  # [batch, 1+seq_len, dim]
        
        # Create attention mask for combined sequence
        batch_size = pixel_values.shape[0]
        image_attention_mask = torch.ones(
            (batch_size, image_embeds.shape[1]), 
            dtype=attention_mask.dtype, 
            device=attention_mask.device
        )
        combined_attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)
        
        # Forward through encoder only
        encoder_outputs = self.vlm.model.language_model.encoder(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
        )
        
        features = encoder_outputs.last_hidden_state
        
        return features
    
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
        # Process observations
        processed_obs = self.obs_processor(env_obs)
        transformed_obs = self.input_transform(processed_obs)
        
        # Get VLM features
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
    
    def obs_processor(self, env_obs: dict[str, Any]) -> dict[str, Any]:
        """Process raw environment observations into model inputs.
        
        Args:
            env_obs: Raw observation containing:
                - images: List of images or dict of cameras
                - states: Robot state
                - task_descriptions: Language instructions
                
        Returns:
            Processed observation dictionary
        """
        # Extract images
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
        
        # Extract state
        state = env_obs.get("states", env_obs.get("agent_pos", None))
        
        # Extract task description
        task_desc = env_obs.get("task_descriptions", env_obs.get("language_instruction", ""))
        
        return {
            "images": images,
            "state": state,
            "task_description": task_desc,
        }
    
    def input_transform(
        self,
        inputs: dict[str, Any],
        transpose: bool = False
    ) -> dict[str, Any]:
        """Transform inputs to model format.
        
        This should integrate with the tokenizer/processor.
        """
        # TODO: Integrate with Florence2Processor for proper tokenization
        # For now, return as-is with dummy values
        device = next(self.parameters()).device
        
        # Infer batch size from state tensor
        state = inputs.get("state")
        if isinstance(state, torch.Tensor):
            batch_size = state.shape[0]
        elif isinstance(state, np.ndarray):
            batch_size = state.shape[0]
        else:
            # Fallback: try to get batch size from images
            images = inputs.get("images", [])
            if images and len(images) > 0:
                first_img = images[0]
                if isinstance(first_img, torch.Tensor):
                    batch_size = first_img.shape[0]
                elif isinstance(first_img, np.ndarray):
                    batch_size = first_img.shape[0]
                else:
                    batch_size = 1
            else:
                batch_size = 1
        
        # Create dummy tensors (should be replaced with actual processing)
        return {
            "pixel_values": torch.randn(batch_size, self.config.num_images_in_input, 3, 224, 224, device=device),
            "input_ids": torch.zeros(batch_size, self.config.tokenizer_max_length, dtype=torch.long, device=device),
            "attention_mask": torch.ones(batch_size, self.config.tokenizer_max_length, device=device),
            "proprio": torch.zeros(batch_size, self.proprio_dim, device=device) if self.proprio_dim > 0 else None,
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
