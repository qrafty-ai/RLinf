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

"""XVLA model factory and exports.

Uses lerobot's XVLA implementation directly for checkpoint compatibility.
"""

import torch
from typing import Optional

from omegaconf import DictConfig

from rlinf.models.embodiment.xvla.adapter import XVLAAdapter
from rlinf.models.embodiment.xvla.xvla_action_model import XVLAForRLActionPrediction

__all__ = ["XVLAForRLActionPrediction", "get_model", "create_xvla_adapter"]


def create_xvla_adapter(config: DictConfig) -> Optional[XVLAAdapter]:
    """Create XVLA adapter from Hydra config.
    
    Args:
        config: DictConfig with optional 'adapter' key containing:
            - simulator: str - Required simulator name (e.g., "libero")
            - Other optional overrides
            
    Returns:
        XVLAAdapter instance or None if adapter config is not provided
    """
    if not hasattr(config, 'adapter') or config.adapter is None:
        return None
    
    simulator = config.adapter.get('simulator')
    if simulator is None:
        return None
    
    overrides = {k: v for k, v in config.adapter.items() if k != 'simulator'}
    return XVLAAdapter(simulator, overrides if overrides else None)


def get_model(cfg, torch_dtype=None) -> XVLAForRLActionPrediction:
    del torch_dtype

    import os

    from lerobot.policies.xvla.configuration_xvla import XVLAConfig
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

    xvla_cfg = getattr(cfg, "xvla", cfg)
    config_name = getattr(xvla_cfg, "config_name", None)
    if config_name is None:
        raise ValueError("config_name is required for XVLA model (e.g., 'xvla_libero')")

    model_path = getattr(cfg, "model_path", None)

    if model_path:
        if os.path.isabs(model_path) and not os.path.exists(model_path):
            if config_name == "xvla_libero":
                model_path = "lerobot/xvla-libero"
            else:
                raise FileNotFoundError(f"XVLA model_path does not exist: {model_path}")
        lerobot_policy = XVLAPolicy.from_pretrained(model_path)
    else:
        florence_config_dict = getattr(
            xvla_cfg,
            "florence_config",
            {
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
            },
        )
        xvla_config = XVLAConfig(
            florence_config=florence_config_dict,
            tokenizer_name=getattr(xvla_cfg, "tokenizer_name", "facebook/bart-base"),
            tokenizer_max_length=getattr(xvla_cfg, "tokenizer_max_length", 96),
            tokenizer_padding_side=getattr(xvla_cfg, "tokenizer_padding_side", "right"),
            hidden_size=getattr(xvla_cfg, "hidden_size", 1024),
            depth=getattr(xvla_cfg, "depth", 24),
            num_heads=getattr(xvla_cfg, "num_heads", 16),
            mlp_ratio=getattr(xvla_cfg, "mlp_ratio", 4.0),
            num_domains=getattr(xvla_cfg, "num_domains", 30),
            len_soft_prompts=getattr(xvla_cfg, "len_soft_prompts", 32),
            dim_time=getattr(xvla_cfg, "dim_time", 32),
            max_len_seq=getattr(xvla_cfg, "max_len_seq", 512),
            use_hetero_proj=getattr(xvla_cfg, "use_hetero_proj", False),
            action_mode=getattr(xvla_cfg, "action_mode", "ee6d"),
            num_denoising_steps=getattr(xvla_cfg, "num_denoising_steps", 10),
            chunk_size=getattr(xvla_cfg, "chunk_size", 32),
            n_action_steps=getattr(xvla_cfg, "n_action_steps", 32),
            max_action_dim=getattr(xvla_cfg, "max_action_dim", 20),
            use_proprio=getattr(xvla_cfg, "use_proprio", True),
            max_state_dim=getattr(xvla_cfg, "max_state_dim", 20),
            dtype=getattr(xvla_cfg, "dtype", "bfloat16"),
            freeze_vision_encoder=getattr(xvla_cfg, "freeze_vision_encoder", True),
            freeze_language_encoder=getattr(xvla_cfg, "freeze_language_encoder", True),
            train_policy_transformer=getattr(xvla_cfg, "train_policy_transformer", True),
            train_soft_prompts=getattr(xvla_cfg, "train_soft_prompts", True),
        )
        lerobot_policy = XVLAPolicy(xvla_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lerobot_policy = lerobot_policy.to(device).eval()

    adapter = create_xvla_adapter(xvla_cfg)

    model = XVLAForRLActionPrediction.from_lerobot_policy(
        lerobot_policy=lerobot_policy,
        config_name=config_name,
        add_value_head=getattr(xvla_cfg, "add_value_head", False),
        adapter=adapter,
    )

    override_tokenizer_max_length = getattr(xvla_cfg, "tokenizer_max_length", None)
    if override_tokenizer_max_length is not None:
        model.tokenizer_max_length = int(override_tokenizer_max_length)

    override_domain_id = getattr(xvla_cfg, "domain_id", None)
    if override_domain_id is not None:
        model.domain_id = int(override_domain_id)

    return model
