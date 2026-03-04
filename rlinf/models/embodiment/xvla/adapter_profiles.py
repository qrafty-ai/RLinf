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

"""Adapter profiles for XVLA model environment integration.

This module provides configuration profiles for different simulators/environments
that define how observations and actions should be transformed for XVLA models.
Each profile specifies:
- Image preprocessing (target size, view mapping)
- Proprioception handling (state dimensions, padding)
- Action handling (model output format, environment input format, gripper range)
"""

from dataclasses import dataclass


@dataclass
class AdapterProfile:
    """High-level configuration for XVLA adapter for a specific simulator.

    This profile encapsulates all the necessary configurations to adapt the XVLA
    model to work with a specific robotics simulator (e.g., LIBERO, ManiSkill).

    Attributes:
        name: Name of the simulator/environment.
        image: Image preprocessing configuration (resize, view_mapping, rotation).
        proprio: Proprioception handling configuration (state_dim, pad_to, state_keys).
        action: Action space handling configuration (model_output, env_input, gripper_range).
    """

    name: str
    image: dict[str, object]
    proprio: dict[str, object]
    action: dict[str, object]


# =============================================================================
# Pre-defined adapter profiles
# =============================================================================

ADAPTER_PROFILES: dict[str, AdapterProfile] = {
    "libero": AdapterProfile(
        name="libero",
        image={
            "resize": [224, 224],
            "view_mapping": {"main_images": "primary", "wrist_images": "wrist"},
            "rotation": 0,
        },
        proprio={
            "state_dim": 7,
            "pad_to": 20,
            "state_keys": ["states"],
        },
        action={
            "model_output": "ee6d",  # 10D: [dx, dy, dz, rx1, ry1, rz1, rx2, ry2, rz2, gripper]
            "env_input": "axis_angle",  # 7D: [dx, dy, dz, droll, dpitch, dyaw, gripper]
            "gripper_range": [-1, 1],
        },
    ),
}


# =============================================================================
# Profile retrieval
# =============================================================================


def get_adapter_profile(name: str) -> AdapterProfile:
    """Retrieve an adapter profile by simulator name.

    Args:
        name: Name of the simulator/environment (e.g., 'libero').

    Returns:
        The AdapterProfile for the specified simulator.

    Raises:
        KeyError: If no profile exists for the given name.
    """
    name_lower = name.lower()
    if name_lower not in ADAPTER_PROFILES:
        available = ", ".join(ADAPTER_PROFILES.keys())
        raise KeyError(
            f"No adapter profile found for '{name}'. "
            f"Available profiles: {available}"
        )
    return ADAPTER_PROFILES[name_lower]
