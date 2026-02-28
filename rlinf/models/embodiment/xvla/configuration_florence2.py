# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Florence-2 configuration for XVLA backbone.

Adapted from LeRobot's XVLA implementation:
https://github.com/huggingface/lerobot
"""

from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class Florence2VisionConfig:
    """Configuration for Florence2 vision encoder (DaViT architecture).
    
    DaViT: Dual Attention Vision Transformer with hierarchical feature extraction.
    """
    model_type: str = "davit"
    
    # Dropout and regularization
    drop_path_rate: float = 0.1
    
    # Patch embedding parameters (hierarchical stages)
    patch_size: List[int] = field(default_factory=lambda: [7, 3, 3, 3])
    patch_stride: List[int] = field(default_factory=lambda: [4, 2, 2, 2])
    patch_padding: List[int] = field(default_factory=lambda: [3, 1, 1, 1])
    patch_prenorm: List[bool] = field(default_factory=lambda: [False, True, True, True])
    
    # Architecture dimensions
    dim_embed: List[int] = field(default_factory=lambda: [256, 512, 1024, 2048])
    num_heads: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    num_groups: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    depths: List[int] = field(default_factory=lambda: [1, 1, 9, 1])
    window_size: int = 12
    projection_dim: int = 1024
    
    # Temporal and positional embeddings
    visual_temporal_embedding: dict = field(default_factory=lambda: {
        "type": "COSINE",
        "max_temporal_embeddings": 100,
    })
    image_pos_embed: dict = field(default_factory=lambda: {
        "type": "learned_abs_2d",
        "max_pos_embeddings": 1000,
    })
    
    # Feature extraction
    image_feature_source: List[str] = field(default_factory=lambda: [
        "spatial_avg_pool", "temporal_avg_pool"
    ])
    
    # Gradient checkpointing
    enable_checkpoint: bool = False


@dataclass
class Florence2LanguageConfig:
    """Configuration for Florence2 language model (BART architecture).
    
    BART: Bidirectional and Auto-Regressive Transformers.
    """
    model_type: str = "florence2_language"
    
    # Vocabulary and embeddings
    vocab_size: int = 51289
    d_model: int = 1024
    max_position_embeddings: int = 1024
    scale_embedding: bool = False
    
    # Encoder
    encoder_layers: int = 12
    encoder_ffn_dim: int = 4096
    encoder_attention_heads: int = 16
    encoder_layerdrop: float = 0.0
    
    # Decoder (not used in XVLA, kept for completeness)
    decoder_layers: int = 12
    decoder_ffn_dim: int = 4096
    decoder_attention_heads: int = 16
    decoder_layerdrop: float = 0.0
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    classifier_dropout: float = 0.0
    
    # Activation and initialization
    activation_function: str = "gelu"
    init_std: float = 0.02
    
    # Other
    use_cache: bool = True
    num_labels: int = 3
    
    # Special tokens
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2
    is_encoder_decoder: bool = True
    decoder_start_token_id: int = 2
    forced_eos_token_id: int = 2


@dataclass
class Florence2Config:
    """Configuration for Florence2 multimodal model.
    
    Combines DaViT vision encoder with BART language model for multimodal understanding.
    """
    model_type: str = "florence2"
    is_composition: bool = False
    
    # Sub-configs
    vision_config: Florence2VisionConfig = None
    text_config: Florence2LanguageConfig = None
    
    # Fusion and projection
    projection_dim: int = 1024
    
    # Loss
    ignore_index: int = -100
    vocab_size: int = 51289
    
    def __post_init__(self):
        """Initialize nested configs if provided as dicts."""
        if isinstance(self.vision_config, dict):
            self.vision_config = Florence2VisionConfig(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = Florence2VisionConfig()
            
        if isinstance(self.text_config, dict):
            self.text_config = Florence2LanguageConfig(**self.text_config)
        elif self.text_config is None:
            self.text_config = Florence2LanguageConfig()
