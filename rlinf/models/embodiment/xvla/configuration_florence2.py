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

from typing import Any, List, Optional

from transformers import PretrainedConfig


class Florence2VisionConfig(PretrainedConfig):
    """Configuration for Florence2 vision encoder (DaViT architecture).
    
    DaViT: Dual Attention Vision Transformer with hierarchical feature extraction.
    """
    model_type = "davit"
    
    def __init__(
        self,
        drop_path_rate: float = 0.1,
        patch_size: List[int] = None,
        patch_stride: List[int] = None,
        patch_padding: List[int] = None,
        patch_prenorm: List[bool] = None,
        dim_embed: List[int] = None,
        num_heads: List[int] = None,
        num_groups: List[int] = None,
        depths: List[int] = None,
        window_size: int = 12,
        projection_dim: int = 1024,
        visual_temporal_embedding: dict = None,
        image_pos_embed: dict = None,
        image_feature_source: List[str] = None,
        enable_checkpoint: bool = False,
        **kwargs: Any,
    ):
        self.drop_path_rate = drop_path_rate
        self.patch_size = patch_size if patch_size is not None else [7, 3, 3, 3]
        self.patch_stride = patch_stride if patch_stride is not None else [4, 2, 2, 2]
        self.patch_padding = patch_padding if patch_padding is not None else [3, 1, 1, 1]
        self.patch_prenorm = patch_prenorm if patch_prenorm is not None else [False, True, True, True]
        self.dim_embed = dim_embed if dim_embed is not None else [256, 512, 1024, 2048]
        self.num_heads = num_heads if num_heads is not None else [8, 16, 32, 64]
        self.num_groups = num_groups if num_groups is not None else [8, 16, 32, 64]
        self.depths = depths if depths is not None else [1, 1, 9, 1]
        self.window_size = window_size
        self.projection_dim = projection_dim
        
        self.visual_temporal_embedding = visual_temporal_embedding if visual_temporal_embedding is not None else {
            "type": "COSINE",
            "max_temporal_embeddings": 100,
        }
        self.image_pos_embed = image_pos_embed if image_pos_embed is not None else {
            "type": "learned_abs_2d",
            "max_pos_embeddings": 1000,
        }
        
        self.image_feature_source = image_feature_source if image_feature_source is not None else [
            "spatial_avg_pool", "temporal_avg_pool"
        ]
        self.enable_checkpoint = enable_checkpoint
        
        super().__init__(**kwargs)


class Florence2LanguageConfig(PretrainedConfig):
    """Configuration for Florence2 language model (BART architecture).
    
    BART: Bidirectional and Auto-Regressive Transformers.
    """
    model_type = "florence2_language"
    
    def __init__(
        self,
        vocab_size: int = 51289,
        d_model: int = 1024,
        max_position_embeddings: int = 1024,
        scale_embedding: bool = False,
        encoder_layers: int = 12,
        encoder_ffn_dim: int = 4096,
        encoder_attention_heads: int = 16,
        encoder_layerdrop: float = 0.0,
        decoder_layers: int = 12,
        decoder_ffn_dim: int = 4096,
        decoder_attention_heads: int = 16,
        decoder_layerdrop: float = 0.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        classifier_dropout: float = 0.0,
        activation_function: str = "gelu",
        init_std: float = 0.02,
        use_cache: bool = True,
        num_labels: int = 3,
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        is_encoder_decoder: bool = True,
        decoder_start_token_id: int = 2,
        forced_eos_token_id: int = 2,
        _attn_implementation: str = "sdpa",
        **kwargs: Any,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_position_embeddings = max_position_embeddings
        self.scale_embedding = scale_embedding
        
        self.encoder_layers = encoder_layers
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_layerdrop = encoder_layerdrop
        
        self.decoder_layers = decoder_layers
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_layerdrop = decoder_layerdrop
        
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.classifier_dropout = classifier_dropout
        
        self.activation_function = activation_function
        self.init_std = init_std
        
        self.use_cache = use_cache
        self.num_labels = num_labels
        
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.is_encoder_decoder = is_encoder_decoder
        self.decoder_start_token_id = decoder_start_token_id
        self.forced_eos_token_id = forced_eos_token_id
        
        self._attn_implementation = _attn_implementation
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )


class Florence2Config(PretrainedConfig):
    """Configuration for Florence2 multimodal model.
    
    Combines DaViT vision encoder with BART language model for multimodal understanding.
    """
    model_type = "florence2"
    is_composition = True
    
    def __init__(
        self,
        vision_config: Optional[Florence2VisionConfig] = None,
        text_config: Optional[Florence2LanguageConfig] = None,
        projection_dim: int = 1024,
        ignore_index: int = -100,
        vocab_size: int = 51289,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        
        if vision_config is None:
            self.vision_config = Florence2VisionConfig()
        elif isinstance(vision_config, dict):
            self.vision_config = Florence2VisionConfig(**vision_config)
        else:
            self.vision_config = vision_config
        
        if text_config is None:
            self.text_config = Florence2LanguageConfig()
        elif isinstance(text_config, dict):
            self.text_config = Florence2LanguageConfig(**text_config)
        else:
            self.text_config = text_config
        
        self.projection_dim = projection_dim
        self.ignore_index = ignore_index
        self.vocab_size = vocab_size
