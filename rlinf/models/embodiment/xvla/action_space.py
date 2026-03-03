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

"""Action space definitions for XVLA.

Handles action preprocessing, postprocessing, and normalization
for different robot control modes.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class ActionSpace(ABC):
    """Base class for action spaces."""
    
    def __init__(self, action_dim: int):
        self.action_dim = action_dim
    
    @abstractmethod
    def preprocess(self, action: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Preprocess action before feeding to model."""
        pass
    
    @abstractmethod
    def postprocess(self, action: torch.Tensor) -> np.ndarray:
        """Postprocess model output to environment action."""
        pass
    
    @abstractmethod
    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Get action bounds (low, high)."""
        pass


class EE6DActionSpace(ActionSpace):
    """LeRobot/XVLA ee6d action space with packed 20D layout.

    Internal XVLA action layout per timestep (20D):
      [0:3]   position_1 (x, y, z)
      [3:9]   rotation_6d_1
      [9]     gripper_1 (logit)
      [10:13] position_2 (x, y, z)
      [13:19] rotation_6d_2
      [19]    gripper_2 (logit)

    For single-arm LIBERO handoff, we use arm-1 and convert to 7D:
      [pos(3), axis_angle(3), gripper(1)]
    where gripper is discretized to {-1, +1}.
    """
    
    def __init__(
        self,
        pos_bounds: tuple[float, float] = (-1.0, 1.0),
        rot_bounds: tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__(action_dim=20)
        self.pos_bounds = pos_bounds
        self.rot_bounds = rot_bounds

        # Packed indices (LeRobot xvla/action_hub.py compatible)
        self.pos_idx_1 = slice(0, 3)
        self.rot_idx_1 = slice(3, 9)
        self.gripper_idx_1 = 9
        self.pos_idx_2 = slice(10, 13)
        self.rot_idx_2 = slice(13, 19)
        self.gripper_idx_2 = 19
    
    def preprocess(self, action: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Preprocess actions to packed 20D ee6d format.

        Accepts either:
        - 20D packed ee6d actions (pass-through), or
        - 7D LIBERO actions [pos(3), axis_angle(3), gripper(1)] and packs to 20D.
        """
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()

        if action.shape[-1] == 20:
            return action

        if action.shape[-1] != 7:
            raise ValueError(f"EE6DActionSpace expects last dim 7 or 20, got {action.shape[-1]}")

        from rlinf.models.embodiment.xvla.rotation_utils import axis_angle_to_rotation_6d

        pos = torch.clamp(action[..., 0:3], -1.0, 1.0)
        axis_angle = action[..., 3:6]
        rot6d = axis_angle_to_rotation_6d(axis_angle)
        gripper = action[..., 6:7]

        # Convert {-1,+1} or [0,1] to logit-like scalar for packed representation.
        # Keep simple bounded value in [-1,1] for compatibility.
        gripper = torch.clamp(gripper, -1.0, 1.0)

        arm1 = torch.cat([pos, rot6d, gripper], dim=-1)  # 10D
        arm2 = torch.zeros_like(arm1)
        return torch.cat([arm1, arm2], dim=-1)  # 20D
    
    def postprocess(self, action: torch.Tensor) -> np.ndarray:
        """Convert packed 20D ee6d output to LIBERO-compatible 7D.

        Input shapes supported:
        - [B, T, 20]
        - [B, 20]

        Output:
        - [B, T, 7] or [B, 7]
          [pos(3), axis_angle(3), gripper in (0,1)]
        
        Matches LeRobot ee6d action space: gripper is sigmoid(logit) -> continuous (0, 1).
        """
        from rlinf.models.embodiment.xvla.rotation_utils import rotation_6d_to_axis_angle

        if action.shape[-1] != 20:
            raise ValueError(f"EE6DActionSpace.postprocess expects last dim 20, got {action.shape[-1]}")

        # Arm-1 extraction for single-arm LIBERO
        pos = torch.clamp(action[..., self.pos_idx_1], -1.0, 1.0)
        rot6d = action[..., self.rot_idx_1]
        gripper_logit = action[..., self.gripper_idx_1 : self.gripper_idx_1 + 1]

        axis_angle = rotation_6d_to_axis_angle(rot6d)
        axis_angle = torch.clamp(axis_angle, -1.0, 1.0)

        # LeRobot convention: sigmoid -> continuous (0, 1)
        gripper = torch.sigmoid(gripper_logit)

        out = torch.cat([pos, axis_angle, gripper], dim=-1)
        return out.detach().cpu().numpy()
    
    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Get bounds for packed 20D representation."""
        low = np.array([-1.0] * 20)
        high = np.array([1.0] * 20)
        return low, high


class EE7DActionSpace(ActionSpace):
    """7D end-effector pose with gripper.
    
    Actions: [x, y, z, roll, pitch, yaw, gripper] 
    where gripper is in [0, 1] (open/close)
    """
    
    def __init__(
        self,
        pos_bounds: tuple[float, float] = (-1.0, 1.0),
        rot_bounds: tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__(action_dim=7)
        self.pos_bounds = pos_bounds
        self.rot_bounds = rot_bounds
    
    def preprocess(self, action: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Normalize action."""
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()
        return action
    
    def postprocess(self, action: torch.Tensor) -> np.ndarray:
        """Clip pose to [-1, 1], gripper to [0, 1]."""
        # First 6 dims: pose
        pose = torch.clamp(action[..., :6], -1.0, 1.0)
        # Last dim: gripper
        gripper = torch.clamp(action[..., 6:], 0.0, 1.0)
        action = torch.cat([pose, gripper], dim=-1)
        return action.detach().cpu().numpy()
    
    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Get bounds."""
        low = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0])
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        return low, high


class JointActionSpace(ActionSpace):
    """Joint position control action space."""
    
    def __init__(self, num_joints: int = 7):
        super().__init__(action_dim=num_joints)
        self.num_joints = num_joints
    
    def preprocess(self, action: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Normalize joint positions."""
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()
        return action
    
    def postprocess(self, action: torch.Tensor) -> np.ndarray:
        """Clip to [-1, 1]."""
        action = torch.clamp(action, -1.0, 1.0)
        return action.detach().cpu().numpy()
    
    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Get bounds."""
        low = np.array([-1.0] * self.num_joints)
        high = np.array([1.0] * self.num_joints)
        return low, high


class ActionHub:
    """Factory for creating action spaces."""
    
    @staticmethod
    def build(mode: str, **kwargs) -> ActionSpace:
        """Build action space by mode.
        
        Args:
            mode: Action mode ("ee6d", "ee7d", "joint", etc.)
            **kwargs: Additional arguments for specific action space
            
        Returns:
            ActionSpace instance
        """
        if mode == "ee6d":
            return EE6DActionSpace(**kwargs)
        elif mode == "ee7d":
            return EE7DActionSpace(**kwargs)
        elif mode == "joint":
            return JointActionSpace(**kwargs)
        else:
            raise ValueError(f"Unknown action mode: {mode}")
    
    @staticmethod
    def infer_mode_from_data(actions: np.ndarray) -> str:
        """Infer action mode from data shape.
        
        Args:
            actions: Action array
            
        Returns:
            Inferred mode string
        """
        action_dim = actions.shape[-1]
        
        if action_dim == 6:
            return "ee6d"
        elif action_dim == 7:
            return "ee7d"
        else:
            return "joint"
