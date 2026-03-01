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
    """6D end-effector pose action space.
    
    Actions: [x, y, z, roll, pitch, yaw] in normalized space [-1, 1]
    """
    
    def __init__(
        self,
        pos_bounds: tuple[float, float] = (-1.0, 1.0),
        rot_bounds: tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__(action_dim=6)
        self.pos_bounds = pos_bounds
        self.rot_bounds = rot_bounds
    
    def preprocess(self, action: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Normalize action to [-1, 1]."""
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()
        
        # Assume input is already in normalized space for training
        # In practice, you might denormalize based on dataset stats
        return action
    
    def postprocess(self, action: torch.Tensor) -> np.ndarray:
        """Convert to numpy and clip."""
        action = torch.clamp(action, -1.0, 1.0)
        return action.detach().cpu().numpy()
    
    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Get normalized bounds."""
        low = np.array([-1.0] * 6)
        high = np.array([1.0] * 6)
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
