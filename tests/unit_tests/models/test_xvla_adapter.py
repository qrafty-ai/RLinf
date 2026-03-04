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

"""Unit tests for XVLA adapter profiles and transformations."""

import numpy as np
from numpy.typing import NDArray
import pytest
from typing import TYPE_CHECKING, cast

torch = pytest.importorskip("torch")

pytest.importorskip("lerobot")

if TYPE_CHECKING:
    import torch as torch_typing

from rlinf.models.embodiment.xvla.adapter import XVLAAdapter  # pyright: ignore[reportMissingImports]
from rlinf.models.embodiment.xvla.adapter_profiles import (
    ADAPTER_PROFILES,
    get_adapter_profile,
)


@pytest.fixture
def libero_profile():
    return ADAPTER_PROFILES["libero"]


@pytest.fixture
def libero_adapter():
    return XVLAAdapter("libero")


@pytest.fixture
def image_obs_batch2():
    return {
        "main_images": np.zeros((2, 8, 10, 3), dtype=np.uint8),
        "wrist_images": np.full((2, 8, 10, 3), 255, dtype=np.uint8),
    }


def test_get_adapter_profile_libero(libero_profile):
    profile = get_adapter_profile("LiBeRo")

    assert profile is libero_profile
    assert profile.name == "libero"
    assert profile.image["resize"] == [224, 224]
    assert profile.proprio["pad_to"] == 20
    assert profile.action["env_input"] == "axis_angle"


def test_get_adapter_profile_unknown():
    with pytest.raises(KeyError, match="No adapter profile found"):
        get_adapter_profile("unknown_sim")


def test_adapter_init_libero(libero_profile):
    adapter = XVLAAdapter("libero")

    assert adapter.profile is libero_profile
    assert adapter.image_cfg == libero_profile.image
    assert adapter.proprio_cfg == libero_profile.proprio
    assert adapter.action_cfg == libero_profile.action
    assert adapter.image_cfg is not libero_profile.image
    assert adapter.proprio_cfg is not libero_profile.proprio
    assert adapter.action_cfg is not libero_profile.action


def test_adapter_init_with_overrides():
    overrides = {
        "image": {"resize": [64, 64], "rotation": 90},
        "proprio": {"pad_to": 12, "state_keys": ["robot_state"]},
        "action": {"action_dim": 6, "gripper_range": [-0.5, 0.5]},
        "task_description_key": "instruction",
        "view_mapping": {"front_cam": "primary"},
    }

    adapter = XVLAAdapter("libero", overrides=overrides)

    assert adapter.image_cfg["resize"] == [64, 64]
    assert adapter.image_cfg["rotation"] == 90
    assert adapter.image_cfg["view_mapping"] == {"front_cam": "primary"}
    assert adapter.proprio_cfg["pad_to"] == 12
    assert adapter.proprio_cfg["state_keys"] == ["robot_state"]
    assert adapter.action_cfg["action_dim"] == 6
    assert adapter.action_cfg["gripper_range"] == [-0.5, 0.5]
    assert adapter.task_description_key == "instruction"

    assert ADAPTER_PROFILES["libero"].image["resize"] == [224, 224]
    assert ADAPTER_PROFILES["libero"].proprio["pad_to"] == 20


def test_transform_input_images(image_obs_batch2):
    adapter = XVLAAdapter("libero", overrides={"image": {"resize": [16, 16]}})
    env_obs = {
        **image_obs_batch2,
        "states": np.zeros((2, 7), dtype=np.float32),
        "task_descriptions": "do task",
    }

    transformed = adapter.transform_input(env_obs)
    pixel_values = cast("torch_typing.Tensor", transformed["pixel_values"])
    image_mask = cast("torch_typing.Tensor", transformed["image_mask"])

    assert pixel_values.shape == (2, 2, 3, 16, 16)
    assert image_mask.shape == (2, 2)
    assert image_mask.dtype == torch.bool
    assert torch.all(pixel_values[:, 0] == -1.0)
    assert torch.all(pixel_values[:, 1] == 1.0)


def test_transform_input_proprio(libero_adapter):
    env_obs = {
        "main_images": np.zeros((2, 8, 8, 3), dtype=np.uint8),
        "states": np.arange(14, dtype=np.float32).reshape(2, 7),
    }

    transformed = libero_adapter.transform_input(env_obs)
    proprio = transformed["proprio"]

    assert proprio.shape == (2, 20)
    assert torch.allclose(
        proprio[:, :7],
        torch.tensor(np.arange(14, dtype=np.float32).reshape(2, 7)),
    )
    assert torch.all(proprio[:, 7:] == 0)


def test_transform_input_task_descriptions(libero_adapter):
    env_obs: dict[str, object] = {
        "main_images": np.zeros((2, 8, 8, 3), dtype=np.uint8),
        "states": np.zeros((2, 7), dtype=np.float32),
        "task_descriptions": "open drawer",
    }
    transformed = libero_adapter.transform_input(env_obs)
    assert transformed["task_descriptions"] == ["open drawer", "open drawer"]

    env_obs["task_descriptions"] = ["single description"]
    transformed = libero_adapter.transform_input(env_obs)
    assert transformed["task_descriptions"] == [
        "single description",
        "single description",
    ]

    env_obs["task_descriptions"] = ["task-a", "task-b"]
    transformed = libero_adapter.transform_input(env_obs)
    assert transformed["task_descriptions"] == ["task-a", "task-b"]


def test_transform_input_batch_consistency(libero_adapter):
    env_obs = {
        "main_images": np.zeros((2, 8, 8, 3), dtype=np.uint8),
        "wrist_images": np.zeros((1, 8, 8, 3), dtype=np.uint8),
    }

    with pytest.raises(ValueError, match="Inconsistent batch size"):
        libero_adapter.transform_input(env_obs)


def test_transform_output_ee6d_to_axis_angle(monkeypatch, libero_adapter):
    expected_axis_angle = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)

    def fake_rotate6d_to_axis_angle(rot6d: NDArray[np.float32]) -> NDArray[np.float32]:
        assert rot6d.shape == (2, 6)
        return expected_axis_angle

    monkeypatch.setattr(
        "rlinf.models.embodiment.xvla.adapter.rotate6d_to_axis_angle",
        fake_rotate6d_to_axis_angle,
    )

    model_action = torch.tensor(
        [
            [1.0, 2.0, 3.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, -0.5],
            [4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -0.2],
        ],
        dtype=torch.float32,
    )

    transformed = libero_adapter.transform_output(model_action)

    assert transformed.shape == (2, 7)
    assert torch.allclose(transformed[:, :3], model_action[:, :3])
    assert torch.allclose(transformed[:, 3:6], torch.from_numpy(expected_axis_angle))
    assert torch.allclose(transformed[:, 6], model_action[:, 9])


def test_transform_output_gripper_normalization(libero_adapter):
    model_action = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    transformed = libero_adapter.transform_output(model_action)

    assert torch.allclose(transformed[:, -1], torch.tensor([-1.0, 0.0, 1.0]))


def test_transform_output_padding():
    action = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5]], dtype=torch.float32)

    pad_adapter = XVLAAdapter(
        "libero",
        overrides={
            "action": {
                "model_output": "axis_angle",
                "env_input": "axis_angle",
                "action_dim": 9,
            }
        },
    )
    padded = pad_adapter.transform_output(action)
    assert padded.shape == (1, 9)
    assert torch.allclose(padded[:, :7], torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0]]))
    assert torch.all(padded[:, 7:] == 0)

    trim_adapter = XVLAAdapter(
        "libero",
        overrides={
            "action": {
                "model_output": "axis_angle",
                "env_input": "axis_angle",
                "action_dim": 5,
            }
        },
    )
    trimmed = trim_adapter.transform_output(action)
    assert trimmed.shape == (1, 5)
    assert torch.allclose(trimmed, torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]))


def test_adapter_end_to_end_libero(monkeypatch, image_obs_batch2):
    def fake_rotate6d_to_axis_angle(rot6d: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.zeros((rot6d.shape[0], 3), dtype=np.float32)

    monkeypatch.setattr(
        "rlinf.models.embodiment.xvla.adapter.rotate6d_to_axis_angle",
        fake_rotate6d_to_axis_angle,
    )

    adapter = XVLAAdapter("libero")
    env_obs = {
        **image_obs_batch2,
        "states": np.arange(14, dtype=np.float32).reshape(2, 7),
        "task_descriptions": ["open drawer", "close drawer"],
    }
    model_action = torch.tensor(
        [
            [0.1, 0.2, 0.3, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 0.0],
            [0.4, 0.5, 0.6, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0],
        ],
        dtype=torch.float32,
    )

    transformed_input = adapter.transform_input(env_obs)
    transformed_output = adapter.transform_output(model_action)
    pixel_values = cast("torch_typing.Tensor", transformed_input["pixel_values"])
    image_mask = cast("torch_typing.Tensor", transformed_input["image_mask"])
    proprio = cast("torch_typing.Tensor", transformed_input["proprio"])

    assert pixel_values.shape == (2, 2, 3, 224, 224)
    assert image_mask.shape == (2, 2)
    assert proprio.shape == (2, 20)
    assert transformed_input["task_descriptions"] == ["open drawer", "close drawer"]

    assert transformed_output.shape == (2, 7)
    assert torch.allclose(transformed_output[:, :3], model_action[:, :3])
    assert torch.allclose(transformed_output[:, 3:6], torch.zeros((2, 3)))
    assert torch.allclose(transformed_output[:, 6], torch.tensor([-1.0, 1.0]))
