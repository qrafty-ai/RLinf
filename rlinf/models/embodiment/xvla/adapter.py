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

"""XVLA environment adapter."""

from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from lerobot.policies.xvla.utils import rotate6d_to_axis_angle

from .adapter_profiles import AdapterProfile, get_adapter_profile


class XVLAAdapter:
    """Profile-driven adapter between env observations/actions and XVLA IO."""

    def __init__(self, simulator: str, overrides: dict[str, object] | None = None) -> None:
        """Load simulator profile and optional override values."""
        self.profile: AdapterProfile = get_adapter_profile(simulator)

        self.image_cfg: dict[str, object] = deepcopy(self.profile.image)
        self.proprio_cfg: dict[str, object] = deepcopy(self.profile.proprio)
        self.action_cfg: dict[str, object] = deepcopy(self.profile.action)
        self.task_description_key = "task_descriptions"

        if overrides is not None:
            self._apply_overrides(overrides)

    def transform_input(self, env_obs: dict[str, object]) -> dict[str, object]:
        """Transform environment observations to XVLA-ready observations."""
        view_mapping = self._get_view_mapping()

        image_views: list[torch.Tensor] = []
        batch_size: int | None = None
        for obs_key in view_mapping:
            image = env_obs.get(obs_key)
            if image is None:
                continue

            image_tensor = self._to_bvchw(image)
            if batch_size is None:
                batch_size = int(image_tensor.shape[0])
            elif image_tensor.shape[0] != batch_size:
                raise ValueError(
                    f"Inconsistent batch size for key '{obs_key}': "
                    f"expected {batch_size}, got {image_tensor.shape[0]}"
                )
            image_views.append(image_tensor)

        if len(image_views) == 0:
            raise ValueError(f"No image found for configured views: {list(view_mapping)}")

        pixel_values = torch.cat(image_views, dim=1)
        pixel_values = self._resize(pixel_values)
        pixel_values = self._normalize_image(pixel_values)

        if batch_size is None:
            batch_size = int(pixel_values.shape[0])

        image_mask = torch.ones(
            batch_size,
            pixel_values.shape[1],
            dtype=torch.bool,
            device=pixel_values.device,
        )
        proprio = self._process_proprio(env_obs, batch_size)
        task_descriptions = self._process_task_descriptions(env_obs, batch_size)

        return {
            "pixel_values": pixel_values,
            "image_mask": image_mask,
            "proprio": proprio,
            "task_descriptions": task_descriptions,
        }

    def transform_output(self, model_action: torch.Tensor) -> torch.Tensor:
        """Transform XVLA output action to environment action format."""
        action = model_action.to(dtype=torch.float32)

        model_output = str(self.action_cfg.get("model_output", "ee6d"))
        env_input = str(self.action_cfg.get("env_input", "axis_angle"))
        if model_output == "ee6d" and env_input == "axis_angle" and action.shape[-1] >= 10:
            action = self._convert_ee6d_to_axis_angle(action)

        action = self._normalize_gripper(action)
        target_dim = self._target_action_dim(action)
        action = self._trim_or_pad(action, target_dim)
        return action

    def _apply_overrides(self, overrides: dict[str, object]) -> None:
        image_override = overrides.get("image")
        if isinstance(image_override, dict):
            self.image_cfg.update(image_override)

        proprio_override = overrides.get("proprio")
        if isinstance(proprio_override, dict):
            self.proprio_cfg.update(proprio_override)

        action_override = overrides.get("action")
        if isinstance(action_override, dict):
            self.action_cfg.update(action_override)

        task_key = overrides.get("task_description_key")
        if isinstance(task_key, str):
            self.task_description_key = task_key

        view_mapping = overrides.get("view_mapping")
        if view_mapping is not None:
            self.image_cfg["view_mapping"] = view_mapping

    def _get_view_mapping(self) -> dict[str, str]:
        mapping = self.image_cfg.get("view_mapping")
        if isinstance(mapping, dict):
            return {str(key): str(value) for key, value in mapping.items()}

        fallback = {}
        for key in ("main_images", "wrist_images"):
            fallback[key] = key
        return fallback

    def _to_bvchw(self, image: object) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            tensor = torch.from_numpy(image)
        elif isinstance(image, torch.Tensor):
            tensor = image
        else:
            tensor = torch.as_tensor(image)

        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() == 4:
            tensor = tensor.unsqueeze(1)

        if tensor.dim() != 5:
            raise ValueError(f"Expected image rank 5 after reshape, got {tuple(tensor.shape)}")

        if tensor.shape[-1] == 3:
            tensor = tensor.permute(0, 1, 4, 2, 3)

        if tensor.shape[2] != 3:
            raise ValueError(f"Expected 3 channels, got {tuple(tensor.shape)}")

        return tensor.to(dtype=torch.float32)

    def _resize(self, pixel_values: torch.Tensor) -> torch.Tensor:
        resize = self.image_cfg.get("resize", [224, 224])
        if isinstance(resize, (list, tuple)) and len(resize) == 2:
            target_h = int(resize[0])
            target_w = int(resize[1])
        else:
            target_h, target_w = 224, 224

        if tuple(pixel_values.shape[-2:]) == (target_h, target_w):
            return pixel_values

        batch_size, num_views = pixel_values.shape[:2]
        resized = F.interpolate(
            pixel_values.flatten(0, 1),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        return resized.unflatten(0, (batch_size, num_views))

    def _normalize_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet normalization to match XVLA training."""
        # First scale to [0, 1] if needed
        if pixel_values.max() > 1.0:
            pixel_values = pixel_values / 255.0
        pixel_values = pixel_values.clamp(0.0, 1.0)

        # ImageNet normalization: (x - mean) / std
        mean = torch.tensor([0.485, 0.456, 0.406], device=pixel_values.device)
        std = torch.tensor([0.229, 0.224, 0.225], device=pixel_values.device)

        # Reshape for broadcasting: [C] -> [1, 1, C, 1, 1]
        mean = mean.view(1, 1, 3, 1, 1)
        std = std.view(1, 1, 3, 1, 1)

        return (pixel_values - mean) / std

    def _process_proprio(self, env_obs: dict[str, object], batch_size: int) -> torch.Tensor:
        """Process proprioception into 20D XVLA format: [pos(3), rot6d(6), pad(1), zeros(10)]."""
        # Extract raw state components from env
        # LIBERO provides: pos(3), axis_angle(3), gripper_qpos(2) = 8D
        state = env_obs.get("states")

        assert state is not None

        if isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).float()
        elif isinstance(state, torch.Tensor):
            state_tensor = state.float()
        else:
            state_tensor = torch.as_tensor(state).float()

        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        # Ensure batch size
        if state_tensor.shape[0] != batch_size:
            if state_tensor.shape[0] == 1:
                state_tensor = state_tensor.expand(batch_size, -1)
            else:
                raise ValueError(
                    f"Inconsistent state batch size: {state_tensor.shape[0]} vs expected {batch_size}"
                )

        # Extract components: LIBERO state is [pos(3), axis_angle(3), gripper(2)]
        eef_pos = state_tensor[:, :3]  # (B, 3)
        axis_angle = state_tensor[:, 3:6]  # (B, 3)

        # Convert axis-angle -> rotation matrix -> 6D rotation
        rot_mat = self._axis_angle_to_rotation_matrix(axis_angle)  # (B, 3, 3)
        rot6d = self._rotation_matrix_to_6d(rot_mat)  # (B, 6)

        # Build 10D proprio: [pos(3), rot6d(6), pad(1)]
        proprio_10d = torch.cat([
            eef_pos,
            rot6d,
            torch.zeros(batch_size, 1, dtype=torch.float32, device=state_tensor.device)
        ], dim=-1)  # (B, 10)

        # Pad to 20D: [10D | 10D zeros]
        zeros_10d = torch.zeros_like(proprio_10d)
        state_20d = torch.cat([proprio_10d, zeros_10d], dim=-1)  # (B, 20)

        return state_20d

    def _axis_angle_to_rotation_matrix(self, axis_angle: torch.Tensor) -> torch.Tensor:
        """Convert axis-angle to rotation matrix using Rodrigues formula."""
        batch_size = axis_angle.shape[0]
        angle = torch.norm(axis_angle, dim=-1, keepdim=True)  # (B, 1)

        # Handle zero rotation
        mask = angle.squeeze(-1) > 1e-6

        # Normalize axis
        axis = axis_angle / (angle + 1e-8)  # (B, 3)

        # Skew-symmetric matrix
        x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]

        K = torch.zeros(batch_size, 3, 3, dtype=axis_angle.dtype, device=axis_angle.device)
        K[:, 0, 1] = -z
        K[:, 0, 2] = y
        K[:, 1, 0] = z
        K[:, 1, 2] = -x
        K[:, 2, 0] = -y
        K[:, 2, 1] = x

        # Rodrigues formula: R = I + sin(angle) * K + (1 - cos(angle)) * K^2
        I = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device).unsqueeze(0)

        sin_angle = torch.sin(angle).unsqueeze(-1)  # (B, 1, 1)
        cos_angle = torch.cos(angle).unsqueeze(-1)  # (B, 1, 1)

        R = I + sin_angle * K + (1 - cos_angle) * torch.bmm(K, K)

        # For zero rotations, return identity
        identity = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device).unsqueeze(0).expand(batch_size, -1, -1)
        R = torch.where(mask.unsqueeze(-1).unsqueeze(-1), R, identity)

        return R

    def _rotation_matrix_to_6d(self, rot_mat: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrix to 6D representation (first 2 columns)."""
        # Take first 2 columns: (B, 3, 2) -> (B, 6)
        return rot_mat[:, :, :2].reshape(rot_mat.shape[0], 6)

    def _process_task_descriptions(
        self,
        env_obs: dict[str, object],
        batch_size: int,
    ) -> list[str]:
        task = env_obs.get(self.task_description_key, "")
        if isinstance(task, str):
            return [task] * batch_size
        if isinstance(task, list):
            if len(task) == 0:
                return [""] * batch_size
            if len(task) == 1 and batch_size > 1:
                return [str(task[0])] * batch_size
            if len(task) == batch_size:
                return [str(item) for item in task]
            raise ValueError(
                f"Task description length mismatch: {len(task)} vs batch size {batch_size}"
            )
        return [str(task)] * batch_size

    def _convert_ee6d_to_axis_angle(self, action: torch.Tensor) -> torch.Tensor:
        pos = action[..., :3]
        rot6d = action[..., 3:9]

        original_shape = rot6d.shape
        rot6d_flat = rot6d.reshape(-1, 6)
        rot6d_np = rot6d_flat.detach().cpu().to(torch.float32).numpy()
        axis_angle_np = rotate6d_to_axis_angle(rot6d_np)

        axis_angle = torch.from_numpy(axis_angle_np).to(action.device, action.dtype)
        axis_angle = axis_angle.reshape(*original_shape[:-1], 3)
        gripper = action[..., 9:10]
        return torch.cat([pos, axis_angle, gripper], dim=-1)

    def _normalize_gripper(self, action: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid and binarize gripper to match XVLA output format."""
        if action.shape[-1] < 1:
            return action

        gripper = action[..., -1:]

        # Apply sigmoid if gripper appears to be logits (unbounded)
        if torch.any(torch.abs(gripper) > 5.0):
            gripper = torch.sigmoid(gripper)

        # Binarize: >0.5 -> 1.0, else -> -1.0 (LIBERO format)
        gripper = torch.where(gripper > 0.5, torch.ones_like(gripper), -torch.ones_like(gripper))

        # Clamp to configured range
        gripper_range = self.action_cfg.get("gripper_range", [-1.0, 1.0])
        if isinstance(gripper_range, (list, tuple)) and len(gripper_range) == 2:
            min_v = float(gripper_range[0])
            max_v = float(gripper_range[1])
        else:
            min_v, max_v = -1.0, 1.0

        gripper = gripper.clamp(min_v, max_v)
        return torch.cat([action[..., :-1], gripper], dim=-1)

    def _target_action_dim(self, action: torch.Tensor) -> int:
        configured = self.action_cfg.get("action_dim")
        if isinstance(configured, (int, float)):
            return int(configured)

        env_input = str(self.action_cfg.get("env_input", "axis_angle"))
        if env_input == "axis_angle":
            return 7
        return int(action.shape[-1])

    def _trim_or_pad(self, action: torch.Tensor, target_dim: int) -> torch.Tensor:
        if target_dim <= 0:
            return action
        if action.shape[-1] == target_dim:
            return action
        if action.shape[-1] > target_dim:
            return action[..., :target_dim]

        pad_shape = (*action.shape[:-1], target_dim - action.shape[-1])
        return torch.cat([action, action.new_zeros(pad_shape)], dim=-1)
