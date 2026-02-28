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

"""XVLA data configuration registry."""

from dataclasses import dataclass
from typing import Any


@dataclass
class XVLADataConfig:
    """Configuration for XVLA training data."""
    name: str
    model_config: dict[str, Any]
    data_transforms: dict[str, Any]
    assets: dict[str, str] | None = None


# Registry of XVLA configs
XVLACONFIGS: dict[str, XVLADataConfig] = {}


def get_xvla_config(config_name: str, model_path: str | None = None) -> Any:
    """Get XVLA training configuration by name.
    
    Args:
        config_name: Name of the config (e.g., "xvla_libero")
        model_path: Path to model checkpoint (optional)
        
    Returns:
        Training configuration object
    """
    if config_name not in XVLACONFIGS:
        raise ValueError(
            f"Unknown XVLA config: {config_name}. "
            f"Available: {list(XVLACONFIGS.keys())}"
        )
    
    config = XVLACONFIGS[config_name]
    
    # TODO: Return actual training config object
    # This should create the full config with data transforms, etc.
    raise NotImplementedError("Config creation not implemented")


# TODO: Register configs
# XVLACONFIGS["xvla_libero"] = XVLADataConfig(...)
# XVLACONFIGS["xvla_maniskill"] = XVLADataConfig(...)
