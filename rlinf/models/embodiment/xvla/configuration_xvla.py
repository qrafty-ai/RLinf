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

"""XVLA configuration for flow-matching VLA with Florence2 backbone.

Adapted from LeRobot's XVLA implementation:
https://github.com/huggingface/lerobot

Key components:
- Florence2 (DaViT + BART) as multimodal encoder
- SoftPromptedTransformer as policy head
- Flow-matching for action generation
- Multi-domain support via soft prompts
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from rlinf.models.embodiment.xvla.configuration_florence2 import (
    Florence2Config,
    Florence2VisionConfig,
    Florence2LanguageConfig,
)


@dataclass(frozen=True)
class XVLAConfig:
    """Configuration for XVLA (Flow-Matching Vision-Language-Action) model.
    
    XVLA uses Florence2 as the vision-language backbone and a SoftPromptedTransformer
    as the policy head for flow-matching based action generation.
    """
    
    # =========================================================================
    # Model Identification
    # =========================================================================
    config_name: str = "xvla_libero"  # xvla_libero, xvla_maniskill, etc.
    
    # =========================================================================
    # Florence2 Backbone Configuration
    # =========================================================================
    # Nested Florence2 config containing vision (DaViT) and text (BART) configs
    florence_config: dict = field(default_factory=lambda: {
        "vision_config": {
            "drop_path_rate": 0.1,
            "patch_size": [7, 3, 3, 3],
            "patch_stride": [4, 2, 2, 2],
            "patch_padding": [3, 1, 1, 1],
            "patch_prenorm": [False, True, True, True],
            "dim_embed": [256, 512, 1024, 2048],
            "num_heads": [8, 16, 32, 64],
            "num_groups": [8, 16, 32, 64],
            "depths": [1, 1, 9, 1],
            "window_size": 12,
            "projection_dim": 1024,
            "visual_temporal_embedding": {
                "type": "COSINE",
                "max_temporal_embeddings": 100,
            },
            "image_pos_embed": {
                "type": "learned_abs_2d",
                "max_pos_embeddings": 1000,
            },
            "image_feature_source": ["spatial_avg_pool", "temporal_avg_pool"],
            "enable_checkpoint": False,
        },
        "text_config": {
            "vocab_size": 51289,
            "d_model": 1024,
            "max_position_embeddings": 1024,
            "encoder_layers": 12,
            "encoder_ffn_dim": 4096,
            "encoder_attention_heads": 16,
            "decoder_layers": 12,
            "decoder_ffn_dim": 4096,
            "decoder_attention_heads": 16,
            "dropout": 0.1,
            "attention_dropout": 0.0,
            "activation_dropout": 0.0,
            "activation_function": "gelu",
            "init_std": 0.02,
            "scale_embedding": False,
            "use_cache": True,
            "pad_token_id": 1,
            "bos_token_id": 0,
            "eos_token_id": 2,
            "is_encoder_decoder": True,
        },
        "projection_dim": 1024,
        "ignore_index": -100,
        "vocab_size": 51289,
    })
    
    # Tokenizer configuration
    tokenizer_name: str = "facebook/bart-large"
    tokenizer_max_length: int = 64
    tokenizer_padding_side: str = "right"
    pad_language_to: str = "max_length"
    
    # =========================================================================
    # SoftPromptedTransformer Policy Head
    # =========================================================================
    hidden_size: int = 1024  # Transformer hidden dimension
    depth: int = 24  # Number of transformer layers
    num_heads: int = 16  # Number of attention heads
    mlp_ratio: float = 4.0  # MLP hidden dim ratio
    num_domains: int = 30  # Number of domains for soft prompts
    len_soft_prompts: int = 32  # Length of soft prompt sequences
    dim_time: int = 32  # Time embedding dimension
    max_len_seq: int = 512  # Maximum sequence length
    use_hetero_proj: bool = False  # Use heterogeneous projection
    
    # =========================================================================
    # Flow-Matching Configuration
    # =========================================================================
    noise_method: str = "flow_matching"  # flow_matching, flow_sde, consistency_model
    num_steps: int = 10  # Number of denoising steps for generation
    chunk_size: int = 32  # Action chunk size (number of future actions)
    n_action_steps: int = 32  # Number of action steps to execute
    n_obs_steps: int = 1  # Number of observation steps
    
    # Flow-matching noise schedule
    sigma_min: float = 0.001
    sigma_max: float = 1.0
    rho: float = 7.0  # Schedule parameter
    time_schedule: str = "lognorm"  # lognorm, uniform, cosine
    
    # =========================================================================
    # Action Space Configuration
    # =========================================================================
    action_mode: str = "ee6d"  # Action representation mode:
                               # - "ee6d": End-effector 6D pose
                               # - "joint": Joint positions
                               # - "auto": Auto-detect from data
                               # - "ee7d": End-effector 6D + gripper
                               # - "joint_gripper": Joint + gripper
    max_action_dim: int = 20  # Maximum action dimension (for padding)
    
    # =========================================================================
    # Observation Configuration
    # =========================================================================
    num_images_in_input: int = 2  # Number of camera views
    num_image_views: Optional[int] = None  # Total image views (including padding)
    empty_cameras: int = 0  # Number of empty camera slots for padding
    resize_imgs_with_padding: Optional[tuple] = None  # (height, width) for resizing
    
    # Proprioception
    use_proprio: bool = True  # Use proprioceptive state
    max_state_dim: int = 32  # Maximum state dimension (for padding)
    domain_feature_key: Optional[str] = None  # Key for domain ID in batch
    
    # =========================================================================
    # Training Configuration
    # =========================================================================
    dtype: str = "bfloat16"  # Model dtype: "bfloat16" or "float32"
    
    # Freezing options (VLM components are frozen by default)
    freeze_vision_encoder: bool = True
    freeze_language_encoder: bool = True
    train_policy_transformer: bool = True
    train_soft_prompts: bool = True
    
    # Training-specific flags
    train_expert_only: bool = False  # If True, only train action expert components
    
    # =========================================================================
    # RL-Specific Configuration
    # =========================================================================
    add_value_head: bool = False  # Add value head for PPO
    detach_critic_input: bool = False  # Detach critic gradients from action expert
    chunk_critic_input: bool = False  # Use only action chunk for value estimation
    
    # Value head configuration (when add_value_head=True)
    value_vlm_mode: str = "mean_token"  # Value aggregation: last_token, mean_token, first_token
    value_after_vlm: bool = False  # Place value head after VLM (vs. after transformer)
    
    # =========================================================================
    # Optional: Noise injection for exploration
    # =========================================================================
    noise_level: float = 0.0
    noise_anneal: bool = False
    
    # =========================================================================
    # Data and Preprocessing
    # =========================================================================
    # Normalization modes for different feature types
    normalization_mapping: dict = field(default_factory=lambda: {
        "VISUAL": "IDENTITY",
        "STATE": "IDENTITY",
        "ACTION": "IDENTITY",
    })
    
    # Observation/action temporal indices
    @property
    def observation_delta_indices(self) -> Optional[List[int]]:
        return None
    
    @property
    def action_delta_indices(self) -> List[int]:
        return list(range(self.chunk_size))
    
    @property
    def reward_delta_indices(self) -> Optional[List[int]]:
        return None
    
    def get_florence_config(self) -> Florence2Config:
        """Build and return Florence2Config from nested dict.
        
        Returns:
            Florence2Config instance
        """
        config_dict = dict(self.florence_config)
        return Florence2Config(**config_dict)
    
    def validate(self) -> None:
        """Validate configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be strictly positive")
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) must be <= "
                f"chunk_size ({self.chunk_size})"
            )
        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")
        if self.action_mode not in ["ee6d", "joint", "auto", "ee7d", "joint_gripper"]:
            raise ValueError(f"Invalid action_mode: {self.action_mode}")
