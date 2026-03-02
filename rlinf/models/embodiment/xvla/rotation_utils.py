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

"""Rotation utilities for 6D rotation representation.

Implements conversion between:
- Quaternion (x, y, z, w)
- Rotation matrix (3x3)
- Axis-angle (3D)
- 6D rotation representation (6D)

Based on "On the Continuity of Rotation Representations in Neural Networks"
by Zhou et al. (CVPR 2019).
"""

import numpy as np
import torch
import torch.nn.functional as F


def quaternion_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (x, y, z, w) to rotation matrix (3x3).
    
    Args:
        quat: Quaternion tensor [..., 4] (x, y, z, w)
        
    Returns:
        Rotation matrix [..., 3, 3]
    """
    quat = F.normalize(quat, dim=-1)
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w
    
    R = torch.stack([
        torch.stack([1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)], dim=-1),
        torch.stack([2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)], dim=-1),
        torch.stack([2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)], dim=-1),
    ], dim=-2)
    
    return R


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix (3x3) to 6D representation.
    
    Takes first two columns of rotation matrix.
    
    Args:
        matrix: Rotation matrix [..., 3, 3]
        
    Returns:
        6D rotation representation [..., 6]
    """
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def rotation_6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to rotation matrix (3x3).
    
    Uses Gram-Schmidt to orthonormalize the first two columns,
    then computes the third as their cross product.
    
    Args:
        rot6d: 6D rotation representation [..., 6]
        
    Returns:
        Rotation matrix [..., 3, 3]
    """
    a1 = rot6d[..., :3]  # First column
    a2 = rot6d[..., 3:]  # Second column
    
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    
    return torch.stack([b1, b2, b3], dim=-1)


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle to rotation matrix (3x3).
    
    Uses Rodrigues' rotation formula.
    
    Args:
        axis_angle: Axis-angle [..., 3]
        
    Returns:
        Rotation matrix [..., 3, 3]
    """
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / (angle + 1e-8)
    
    # Rodrigues' rotation formula
    K = torch.zeros(*axis.shape[:-1], 3, 3, device=axis.device, dtype=axis.dtype)
    K[..., 0, 1] = -axis[..., 2]
    K[..., 0, 2] = axis[..., 1]
    K[..., 1, 0] = axis[..., 2]
    K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]
    K[..., 2, 1] = axis[..., 0]
    
    I = torch.eye(3, device=axis.device, dtype=axis.dtype)
    R = I + torch.sin(angle).unsqueeze(-1) * K + (1 - torch.cos(angle)).unsqueeze(-1) * (K @ K)
    
    return R


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to axis-angle representation.
    
    Args:
        matrix: Rotation matrix [..., 3, 3]
        
    Returns:
        Axis-angle [..., 3] (direction * angle)
    """
    trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))
    
    axis = torch.stack([
        matrix[..., 2, 1] - matrix[..., 1, 2],
        matrix[..., 0, 2] - matrix[..., 2, 0],
        matrix[..., 1, 0] - matrix[..., 0, 1],
    ], dim=-1)
    
    axis_norm = torch.norm(axis, dim=-1, keepdim=True)
    axis = torch.where(
        axis_norm > 1e-6,
        axis / axis_norm,
        torch.tensor([1.0, 0.0, 0.0], device=axis.device, dtype=axis.dtype).unsqueeze(0)
    )
    
    return axis * angle.unsqueeze(-1)


def quaternion_to_rotation_6d(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion directly to 6D rotation.
    
    Args:
        quat: Quaternion [..., 4] (x, y, z, w)
        
    Returns:
        6D rotation [..., 6]
    """
    matrix = quaternion_to_matrix(quat)
    return matrix_to_rotation_6d(matrix)


def axis_angle_to_rotation_6d(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle directly to 6D rotation.
    
    Args:
        axis_angle: Axis-angle [..., 3]
        
    Returns:
        6D rotation [..., 6]
    """
    matrix = axis_angle_to_matrix(axis_angle)
    return matrix_to_rotation_6d(matrix)


def rotation_6d_to_axis_angle(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation directly to axis-angle.
    
    Args:
        rot6d: 6D rotation [..., 6]
        
    Returns:
        Axis-angle [..., 3]
    """
    matrix = rotation_6d_to_matrix(rot6d)
    return matrix_to_axis_angle(matrix)


# NumPy versions
def quaternion_to_matrix_np(quat: np.ndarray) -> np.ndarray:
    """NumPy version: quaternion to rotation matrix."""
    quat = quat / (np.linalg.norm(quat) + 1e-8)
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w
    
    R = np.stack([
        np.stack([1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)], axis=-1),
        np.stack([2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)], axis=-1),
        np.stack([2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)], axis=-1),
    ], axis=-2)
    
    return R


def matrix_to_rotation_6d_np(matrix: np.ndarray) -> np.ndarray:
    """NumPy version: rotation matrix to 6D."""
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def quaternion_to_rotation_6d_np(quat: np.ndarray) -> np.ndarray:
    """NumPy version: quaternion to 6D rotation."""
    matrix = quaternion_to_matrix_np(quat)
    return matrix_to_rotation_6d_np(matrix)


def rotation_6d_to_matrix_np(rot6d: np.ndarray) -> np.ndarray:
    """NumPy version: 6D rotation to rotation matrix."""
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:]
    
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    
    return np.stack([b1, b2, b3], axis=-1)


def matrix_to_axis_angle_np(matrix: np.ndarray) -> np.ndarray:
    """NumPy version: rotation matrix to axis-angle."""
    trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    
    axis = np.stack([
        matrix[..., 2, 1] - matrix[..., 1, 2],
        matrix[..., 0, 2] - matrix[..., 2, 0],
        matrix[..., 1, 0] - matrix[..., 0, 1],
    ], axis=-1)
    
    axis_norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    axis = np.where(
        axis_norm > 1e-6,
        axis / axis_norm,
        np.array([1.0, 0.0, 0.0])
    )
    
    return axis * angle[..., np.newaxis]


def rotation_6d_to_axis_angle_np(rot6d: np.ndarray) -> np.ndarray:
    """NumPy version: 6D rotation to axis-angle."""
    matrix = rotation_6d_to_matrix_np(rot6d)
    return matrix_to_axis_angle_np(matrix)
