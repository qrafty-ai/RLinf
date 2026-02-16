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

"""
XVLA model integration for RLinf.

This module provides the model loading and configuration functions for XVLA,
LeRobot's soft-prompted VLA model using flow-matching for action prediction.
"""

import os
from typing import Any, Optional

import torch
from omegaconf import DictConfig


def get_model_config_and_input_processor(cfg: DictConfig):
    """
    Get model configuration and input processor for XVLA.

    Args:
        cfg: Model configuration DictConfig

    Returns:
        Tuple of (model_config, input_processor)
    """
    # Try to import LeRobot components
    try:
        from lerobot.policies.xvla.configuration_xvla import XVLAConfig
        from lerobot.policies.factory import make_pre_post_processors

        # Load XVLA config from pretrained model
        model_config = XVLAConfig.from_pretrained(
            cfg.model_path,
            trust_remote_code=cfg.trust_remote_code,
        )

        # Create pre/post processors for XVLA
        # These handle image normalization and domain_id injection
        preprocessors, postprocessors = make_pre_post_processors(
            model_config, cfg.dataset_name if hasattr(cfg, "dataset_name") else None
        )

        # Return config and processors
        return model_config, {
            "preprocessors": preprocessors,
            "postprocessors": postprocessors,
        }

    except ImportError:
        # If LeRobot is not available, return placeholder
        # This allows the code to be imported even without LeRobot installed
        return None, None


def get_model(cfg: DictConfig, torch_dtype: torch.dtype = torch.bfloat16):
    """
    Load XVLA model for RLinf training.

    Args:
        cfg: Model configuration DictConfig
        torch_dtype: Data type for model weights

    Returns:
        XVLA model instance
    """
    try:
        from lerobot.policies.xvla.configuration_xvla import XVLAConfig
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    except ImportError as e:
        raise ImportError(
            "LeRobot XVLA is not available. Install with: pip install 'lerobot[xvla]'"
        ) from e

    # Load model config
    actor_model_config = XVLAConfig.from_pretrained(
        cfg.model_path,
        trust_remote_code=cfg.trust_remote_code,
    )

    # Override config with user-specified values
    if hasattr(cfg, "action_dim"):
        actor_model_config.action_dim = cfg.action_dim
    if hasattr(cfg, "action_mode"):
        actor_model_config.action_mode = cfg.action_mode
    if hasattr(cfg, "domain_id"):
        actor_model_config.domain_id = cfg.domain_id

    # Import the RLinf wrapper
    from rlinf.models.embodiment.xvla.xvla_action_model import (
        XVLAForRLActionPrediction,
    )

    # Determine hidden size from config or use default
    hidden_size = cfg.get("hidden_size", actor_model_config.hidden_size)

    # Load the base XVLAPolicy
    model = XVLAPolicy.from_pretrained(
        cfg.model_path,
        config=actor_model_config,
        torch_dtype=torch_dtype,
        trust_remote_code=cfg.trust_remote_code,
    )

    # Wrap with RLinf-compatible wrapper
    xvla_model = XVLAForRLActionPrediction(
        config=actor_model_config,
        hidden_size=hidden_size,
        action_dim=cfg.get("action_dim", 20),
        num_action_chunks=cfg.get("num_action_chunks", 1),
        add_value_head=cfg.get("add_value_head", False),
        action_mode=cfg.get("action_mode", "auto"),
        domain_id=cfg.get("domain_id", 0),
    )

    # Copy weights from loaded model to wrapper
    # This is needed because XVLAForRLActionPrediction wraps XVLAPolicy
    xvla_model.model = model.model if hasattr(model, "model") else model
    xvla_model.config = model.config if hasattr(model, "config") else actor_model_config

    # Move to dtype
    xvla_model.to(torch_dtype)

    return xvla_model
