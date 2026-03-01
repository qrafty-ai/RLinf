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

"""XVLA model factory and exports."""

import os
from typing import Any

try:
    import numpy as np
    import torch

    # Configuration classes
    from rlinf.models.embodiment.xvla.configuration_florence2 import (
        Florence2Config,
        Florence2VisionConfig,
        Florence2LanguageConfig,
    )
    from rlinf.models.embodiment.xvla.configuration_xvla import XVLAConfig
    
    # Model class (scaffold - implement actual model)
    from rlinf.models.embodiment.xvla.xvla_action_model import XVLAForRLActionPrediction
    
    # Data config
    from rlinf.models.embodiment.xvla.dataconfig import get_xvla_config

    def get_model(cfg, torch_dtype=None) -> XVLAForRLActionPrediction:
        """Factory function to instantiate XVLA model with Florence2 backbone.
        
        Args:
            cfg: Model configuration containing:
                - config_name: Environment-specific config name
                - model_path: Path to pretrained weights
                - xvla: Nested XVLA configuration dict
            torch_dtype: Optional torch dtype for model weights (ignored, uses config dtype)
                
        Returns:
            XVLAForRLActionPrediction instance with Florence2 backbone
        """
        # Get XVLA-specific config from nested structure
        xvla_cfg = getattr(cfg, "xvla", cfg)
        
        # Get config name
        config_name = getattr(xvla_cfg, "config_name", None)
        if config_name is None:
            raise ValueError("config_name is required for XVLA model (e.g., 'xvla_libero')")
        
        # Get environment-specific training config (for data transforms) - optional for eval
        try:
            train_config = get_xvla_config(config_name, model_path=cfg.model_path)
        except (ValueError, NotImplementedError):
            # Config not registered or not implemented - skip for evaluation
            train_config = None
        
        # Build Florence2 config from nested structure
        florence_config_dict = getattr(xvla_cfg, "florence_config", {
            "vision_config": {
                "drop_path_rate": 0.1,
                "patch_size": [7, 3, 3, 3],
                "dim_embed": [256, 512, 1024, 2048],
                "num_heads": [8, 16, 32, 64],
                "depths": [1, 1, 9, 1],
                "projection_dim": 1024,
            },
            "text_config": {
                "vocab_size": 51289,
                "d_model": 1024,
                "encoder_layers": 12,
                "projection_dim": 1024,
            },
            "projection_dim": 1024,
        })
        
        # Build XVLA config with nested Florence2 config
        xvla_config = XVLAConfig(
            config_name=config_name,
            # Florence2 backbone
            florence_config=florence_config_dict,
            tokenizer_name=getattr(xvla_cfg, "tokenizer_name", "facebook/bart-large"),
            tokenizer_max_length=getattr(xvla_cfg, "tokenizer_max_length", 64),
            # SoftPromptedTransformer policy head
            hidden_size=getattr(xvla_cfg, "hidden_size", 1024),
            depth=getattr(xvla_cfg, "depth", 24),
            num_heads=getattr(xvla_cfg, "num_heads", 16),
            mlp_ratio=getattr(xvla_cfg, "mlp_ratio", 4.0),
            num_domains=getattr(xvla_cfg, "num_domains", 30),
            len_soft_prompts=getattr(xvla_cfg, "len_soft_prompts", 32),
            dim_time=getattr(xvla_cfg, "dim_time", 32),
            use_hetero_proj=getattr(xvla_cfg, "use_hetero_proj", False),
            # Flow-matching
            noise_method=getattr(xvla_cfg, "noise_method", "flow_matching"),
            num_steps=getattr(xvla_cfg, "num_steps", 10),
            chunk_size=getattr(xvla_cfg, "num_action_chunks", getattr(cfg, "num_action_chunks", 32)),
            n_action_steps=getattr(xvla_cfg, "n_action_steps", 32),
            # Action space
            action_mode=getattr(xvla_cfg, "action_mode", "ee6d"),
            max_action_dim=getattr(xvla_cfg, "max_action_dim", 20),
            # Observation
            num_images_in_input=getattr(xvla_cfg, "num_images_in_input", 2),
            use_proprio=getattr(xvla_cfg, "use_proprio", True),
            max_state_dim=getattr(xvla_cfg, "max_state_dim", 32),
            # Training
            dtype=getattr(xvla_cfg, "dtype", "bfloat16"),
            freeze_vision_encoder=getattr(xvla_cfg, "freeze_vision_encoder", True),
            freeze_language_encoder=getattr(xvla_cfg, "freeze_language_encoder", True),
            train_policy_transformer=getattr(xvla_cfg, "train_policy_transformer", True),
            train_soft_prompts=getattr(xvla_cfg, "train_soft_prompts", True),
            # RL-specific
            add_value_head=getattr(xvla_cfg, "add_value_head", False),
        )
        
        # Get proprio dimension
        proprio_dim = xvla_config.max_state_dim if xvla_config.use_proprio else 0
        
        # Create model with Florence2 backbone
        model = XVLAForRLActionPrediction(
            config=xvla_config,
            proprio_dim=proprio_dim,
        )
        
        # Load checkpoint weights if path provided
        if hasattr(cfg, "model_path") and cfg.model_path:
            # TODO: Implement weight loading from checkpoint
            # - Load safetensors weights
            # - Handle 'model.' prefix for XVLA checkpoints
            # - Map encoder/decoder embeddings if needed
            pass
        
        return model

except ImportError:
    # XVLA dependencies not available
    def get_model(cfg) -> Any:
        raise ImportError(
            "XVLA model requires additional dependencies. "
            "Please install transformers and flow-matching libraries."
        )
