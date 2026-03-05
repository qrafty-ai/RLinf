"""XVLA (Flow-Matching Vision-Language-Action) model for embodied RL.

Builds on top of `lerobot.policies.xvla.modeling_xvla.XVLAModel` for core
XVLA model construction and flow-matching generation, while adding RLinf
interfaces (input/output transform, RL log-prob/value computation, and
checkpoint compatibility helpers).
"""

from copy import deepcopy
from typing import Any, Literal, Mapping, Optional, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import from lerobot directly
from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.policies.xvla.modeling_xvla import XVLAModel, XVLAPolicy
from lerobot.policies.xvla.utils import rotate6d_to_axis_angle
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_TOKENS, OBS_STATE
from transformers.models.bart import BartTokenizerFast
from uni_transform import rotvec_to_matrix

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.utils.logging import get_logger


class ValueHead(nn.Module):
    """Value head for PPO (predicts scalar value from features).

    Simple MLP that takes pooled visual features and outputs a value estimate.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            [batch_size, input_dim] pooled features

        Returns
        -------
        Tensor
            [batch_size, 1] value estimates
        """
        return self.network(x)


class XVLAForRLActionPrediction(nn.Module, BasePolicy):
    """XVLA model for reinforcement learning action prediction using flow-matching.

    Architecture:
    1. Florence2 VLM (frozen): DaViT vision + BART language encoder
    2. SoftPromptedTransformer (trainable): Policy head with domain soft prompts
    3. Flow-matching sampler: Generate actions via iterative denoising

    Uses lerobot's XVLA implementation directly for checkpoint compatibility.
    """

    def __init__(
        self,
        config: XVLAConfig,
        proprio_dim: int = 0,
        config_name: Optional[str] = None,
        domain_id: int = 0,
        add_value_head: bool = False,
    ):
        """Initialize XVLA model.

        Args:
            config: XVLA configuration (from lerobot)
            proprio_dim: Proprioception dimension (0 if not used)
            config_name: Config name for logging
            domain_id: Domain ID for multi-domain training
            add_value_head: Whether to add value head for PPO
            adapter: Optional adapter for input/output transformation
        """
        super().__init__()

        self.config = config
        self.logger = get_logger()
        self.proprio_dim = proprio_dim
        self.chunk_size = config.chunk_size
        self.use_proprio = config.use_proprio
        self.config_name = config_name or "xvla"
        self.domain_id = domain_id
        self.num_denoising_steps = config.num_denoising_steps
        self.add_value_head = add_value_head

        lerobot_policy = XVLAPolicy(config)
        xvla_model = lerobot_policy.model
        self._lerobot_policy = lerobot_policy
        self._xvla_model: XVLAModel
        object.__setattr__(self, "_xvla_model", xvla_model)

        # Keep historical attribute names for checkpoint compatibility.
        self.vlm = xvla_model.vlm
        self.policy_head = xvla_model.transformer
        self.action_space = xvla_model.action_space
        self.dim_action = xvla_model.dim_action
        projection_dim = xvla_model.vlm.config.projection_dim

        # 5. Optional value head for PPO
        if add_value_head:
            self.value_head = ValueHead(
                input_dim=projection_dim,
                hidden_dim=config.hidden_size,
            )
        else:
            self.value_head = None

        # Initialize BART tokenizer for language instructions.
        self.tokenizer = BartTokenizerFast.from_pretrained(
            config.tokenizer_name,
            padding_side=config.tokenizer_padding_side,
        )
        self.tokenizer_max_length = config.tokenizer_max_length
        self._apply_dtype()

        self.logger.info(f"Initialized XVLA model with config: {self.config_name}")
        self.logger.info(f"  Florence2 projection dim: {projection_dim}")
        self.logger.info(f"  Policy head hidden size: {config.hidden_size}")
        self.logger.info(f"  Action dimension: {self.dim_action}")
        self.logger.info(f"  Proprio dimension: {proprio_dim}")
        self.logger.info(f"  Domain ID: {domain_id}")
        self.logger.info(
            f"  Tokenizer: {config.tokenizer_name} (max_length={config.tokenizer_max_length})"
        )

    @classmethod
    def from_lerobot_policy(
        cls,
        lerobot_policy,
        config_name: str = "xvla",
        add_value_head: bool = False,
        io_config: Optional[dict[str, Any]] = None,
    ):
        """Create XVLAForRLActionPrediction from a LeRobot XVLAPolicy.

        This allows loading LeRobot checkpoints directly without conversion.

        Args:
            lerobot_policy: LeRobot XVLAPolicy instance (loaded via from_pretrained)
            config_name: Config name for logging
            add_value_head: Whether to add value head for PPO
            adapter: Optional adapter for input/output transformation

        Returns:
            XVLAForRLActionPrediction instance with loaded weights from LeRobot policy
        """
        # Create instance without calling __init__ to avoid re-initializing components
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)

        # Copy config and core components from LeRobot policy
        if not hasattr(lerobot_policy, "model") or not isinstance(
            lerobot_policy.model, XVLAModel
        ):
            raise ValueError(
                "Expected a LeRobot XVLAPolicy with a valid XVLAModel in `model`."
            )

        xvla_model = lerobot_policy.model
        instance.config = lerobot_policy.config
        instance.logger = get_logger()
        instance.proprio_dim = (
            lerobot_policy.config.max_state_dim
            if lerobot_policy.config.use_proprio
            else 0
        )
        instance.chunk_size = lerobot_policy.config.chunk_size
        instance.use_proprio = lerobot_policy.config.use_proprio
        instance.config_name = config_name
        instance.num_denoising_steps = lerobot_policy.config.num_denoising_steps
        instance.add_value_head = add_value_head
        instance._lerobot_policy = lerobot_policy

        object.__setattr__(instance, "_xvla_model", xvla_model)
        instance.vlm = xvla_model.vlm
        instance.policy_head = xvla_model.transformer
        instance.action_space = xvla_model.action_space
        instance.dim_action = xvla_model.dim_action

        if hasattr(lerobot_policy, "tokenizer"):
            instance.tokenizer = lerobot_policy.tokenizer
        else:
            instance.tokenizer = BartTokenizerFast.from_pretrained(
                lerobot_policy.config.tokenizer_name,
                padding_side=lerobot_policy.config.tokenizer_padding_side,
            )
        instance.tokenizer_max_length = lerobot_policy.config.tokenizer_max_length

        # Set domain_id from config or default
        instance.domain_id = getattr(lerobot_policy.config, "domain_id", 3)

        # Add value head if requested
        if add_value_head:
            projection_dim = xvla_model.vlm.config.projection_dim
            instance.value_head = ValueHead(
                input_dim=projection_dim,
                hidden_dim=lerobot_policy.config.hidden_size,
            )
            instance.value_head.to(dtype=xvla_model._get_target_dtype())
        else:
            instance.value_head = None

        instance.logger.info(
            f"Initialized XVLA model from LeRobot policy: {config_name}"
        )
        instance.logger.info(
            f"  Florence2 projection dim: {lerobot_policy.config.get_florence_config().projection_dim}"
        )
        instance.logger.info(
            f"  Policy head hidden size: {lerobot_policy.config.hidden_size}"
        )
        instance.logger.info(f"  Action dimension: {instance.dim_action}")
        instance.logger.info(f"  Proprio dimension: {instance.proprio_dim}")
        instance.logger.info(f"  Domain ID: {instance.domain_id}")
        instance.logger.info(
            f"  Tokenizer: {lerobot_policy.config.tokenizer_name} (max_length={instance.tokenizer_max_length})"
        )

        return instance

    def _get_target_dtype(self) -> torch.dtype:
        """Get target dtype from core XVLA model config."""
        return self._xvla_model._get_target_dtype()

    def _apply_dtype(self) -> None:
        """Apply dtype casting to XVLA model and value head."""
        self._xvla_model._apply_dtype()
        if self.value_head is not None:
            self.value_head.to(dtype=self._get_target_dtype())

    def _apply_freezing(self) -> None:
        """Apply freezing to core XVLA model."""
        self._xvla_model._apply_freezing()

    @property
    def _no_split_modules(self) -> list[str]:
        """Modules that should not be split during FSDP wrapping."""
        return [
            # Florence2 components
            "Florence2VisionModel",
            "Florence2Encoder",
            "Florence2VisionEncoder",
            "Florence2EncoderLayer",
            # Policy head
            "SoftPromptedTransformer",
            "SoftPromptHub",
            "TransformerBlock",
        ]

    @property
    def _no_split_names(self) -> list[str]:
        """Parameter names that should not be split."""
        return [
            # Action projections
            "action_encoder",
            "action_decoder",
            "vlm_proj",
            "aux_visual_proj",
            # Soft prompts
            "soft_prompts",
            "soft_prompt_hub",
            # Value head
            "value_head",
        ]

    def _get_vlm_features(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        image_mask: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Get visual-language features from Florence2."""
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(1)
        image_mask = self._prepare_image_mask(image_mask, pixel_values)
        input_ids_long = cast(torch.LongTensor, input_ids.to(dtype=torch.long))
        pixel_values_float = cast(
            torch.FloatTensor, pixel_values.to(dtype=self._get_target_dtype())
        )
        return self._xvla_model.forward_vlm(
            input_ids=input_ids_long,
            pixel_values=pixel_values_float,
            image_mask=image_mask,
        )

    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, forward_type: ForwardType = ForwardType.DEFAULT, **kwargs
    ) -> dict[str, Any]:
        """Main forward method dispatching to specific forward implementations."""
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        elif forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        elif forward_type == ForwardType.SAC:
            return self.sac_forward(**kwargs)
        else:
            raise NotImplementedError(f"Forward type {forward_type} not supported")

    def sft_forward(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Supervised fine-tuning forward pass.

        Args:
            data: Dictionary containing:
                - observations: Processed observations with either:
                    - input_ids: Pre-tokenized text [B, seq_len]
                    - task_descriptions: Raw text strings (list[str]) to be tokenized
                - actions: Ground truth actions

        Returns:
            Dictionary with loss and metrics
        """
        if "batch" in data:
            loss, log_dict = self._lerobot_policy.forward(data["batch"])
            metrics = dict(log_dict)
            metrics["sft_loss"] = float(log_dict.get("loss", loss.detach().item()))
            return {"loss": loss, "metrics": metrics}

        observations = data["observations"]
        actions = data["actions"]

        target_dtype = self._get_target_dtype()
        pixel_values = observations["pixel_values"].to(dtype=target_dtype)
        image_mask = observations.get("image_mask")

        # Handle both pre-tokenized input_ids and raw task_descriptions
        if "input_ids" in observations:
            input_ids = observations["input_ids"]
        elif "task_descriptions" in observations:
            # Tokenize task descriptions
            task_desc = observations["task_descriptions"]
            batch_size = actions.shape[0]

            if isinstance(task_desc, str):
                task_desc = [task_desc] * batch_size
            elif isinstance(task_desc, list):
                if len(task_desc) == 1 and batch_size > 1:
                    task_desc = task_desc * batch_size
            else:
                task_desc = [str(task_desc)] * batch_size

            tokenized = self.tokenizer(
                task_desc,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer_max_length,
                return_tensors="pt",
            )
            input_ids = tokenized.input_ids.to(actions.device)
        else:
            raise ValueError(
                "observations must contain either 'input_ids' or 'task_descriptions'"
            )

        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(1)

        batch_size = actions.shape[0]
        image_mask = self._prepare_image_mask(image_mask, pixel_values)
        proprio = self._prepare_proprio(
            observations.get("proprio"),
            batch_size=batch_size,
            device=actions.device,
            dtype=target_dtype,
        )
        domain_id = self._prepare_domain_id(
            observations.get("domain_id"), batch_size, actions.device
        )

        action_target = actions.to(dtype=target_dtype)
        if action_target.ndim == 2:
            action_target = action_target.unsqueeze(1)

        if action_target.shape[1] != self.chunk_size:
            if action_target.shape[1] > self.chunk_size:
                action_target = action_target[:, : self.chunk_size]
            else:
                pad_shape = (
                    action_target.shape[0],
                    self.chunk_size - action_target.shape[1],
                    action_target.shape[2],
                )
                action_target = torch.cat(
                    [action_target, action_target.new_zeros(pad_shape)], dim=1
                )

        if action_target.shape[-1] != self.dim_action:
            action_space_name = str(getattr(self.action_space, "name", "")).lower()
            if (
                action_space_name in {"ee6d", "agibot_ee6d"}
                and action_target.shape[-1] == 7
            ):
                action_target = self._convert_axis_angle_action_to_ee6d(
                    action_target, self.dim_action
                )
            else:
                action_target = self._convert_action_to_target_dim(
                    action_target, self.dim_action
                )

        image_feature_keys = list(self._lerobot_policy.config.image_features)
        batch: dict[str, torch.Tensor] = {
            OBS_LANGUAGE_TOKENS: cast(
                torch.Tensor, input_ids.to(dtype=torch.long, device=actions.device)
            ),
            ACTION: action_target,
            OBS_STATE: proprio,
            "domain_id": cast(
                torch.Tensor, domain_id.to(dtype=torch.long, device=actions.device)
            ),
        }
        for idx, key in enumerate(image_feature_keys):
            if idx < pixel_values.shape[1]:
                batch[key] = pixel_values[:, idx]

        loss, log_dict = self._lerobot_policy.forward(batch)
        metrics = dict(log_dict)
        metrics["sft_loss"] = float(log_dict.get("loss", loss.detach().item()))
        return {"loss": loss, "metrics": metrics}

    def default_forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, forward_inputs: dict[str, Any], **kwargs
    ) -> dict[str, Any]:
        """Default RL forward pass for computing log-probs and values.

        Args:
            forward_inputs: Dictionary containing:
                - chains: Generated action sequences [batch, num_steps, action_chunk, action_dim]
                - timesteps: Denoising timesteps [batch, num_steps]
                - observations: Processed observations

        Returns:
            Dictionary with logprobs, values, entropy
        """
        chains = forward_inputs["chains"]
        timesteps = forward_inputs["timesteps"]
        observations = forward_inputs["observations"]

        target_dtype = self._get_target_dtype()
        batch_size = chains.shape[0]
        num_steps = chains.shape[1]

        pixel_values = observations["pixel_values"].to(dtype=target_dtype)
        input_ids = observations["input_ids"].to(dtype=torch.long)
        image_mask = self._prepare_image_mask(
            observations.get("image_mask"), pixel_values
        )
        proprio = self._prepare_proprio(
            observations.get("proprio"),
            batch_size=batch_size,
            device=chains.device,
            dtype=target_dtype,
        )
        domain_id = self._prepare_domain_id(
            observations.get("domain_id"), batch_size, chains.device
        )

        feature_dict = self._get_vlm_features(pixel_values, input_ids, image_mask)
        vlm_features = feature_dict["vlm_features"]
        aux_visual_inputs = feature_dict["aux_visual_inputs"]

        logprobs_list = []
        for i in range(num_steps):
            action_noisy = chains[:, i].to(dtype=target_dtype)
            t = timesteps[:, i].to(dtype=target_dtype)

            proprio_m, action_noisy_m = self.action_space.preprocess(
                proprio, action_noisy
            )
            pred_action = self.policy_head(
                domain_id=domain_id,
                action_with_noise=action_noisy_m,
                proprio=proprio_m,
                t=t,
                vlm_features=vlm_features,
                aux_visual_inputs=aux_visual_inputs,
            )

            residual = pred_action - action_noisy_m
            logprob = -0.5 * (residual**2).sum(dim=-1)
            logprobs_list.append(logprob)

        logprobs = torch.stack(logprobs_list, dim=1)

        values = None
        if self.value_head is not None:
            pooled_features = vlm_features.mean(dim=1)
            values = self.value_head(pooled_features).squeeze(-1)

        entropy = torch.zeros(batch_size, device=chains.device)

        return {
            "logprobs": logprobs,
            "values": values,
            "entropy": entropy,
        }

    def sac_forward(self, **kwargs) -> dict[str, Any]:
        """Soft Actor-Critic forward pass (placeholder)."""
        raise NotImplementedError("SAC forward not implemented for XVLA")

    def forward_vlm(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Encode text and multi-view images via Florence2 encoder.
        """
        return self._xvla_model.forward_vlm(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_mask=image_mask,
        )

    def _io_get_view_mapping(self) -> list[str]:
        mapping = self.io_image_cfg.get("view_mapping")
        if isinstance(mapping, Mapping):
            return [str(key) for key in mapping.keys()]
        return ["main_images", "wrist_images"]

    def _io_to_bvchw(self, image: object) -> torch.Tensor:
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
            raise ValueError(
                f"Expected image rank 5 after reshape, got {tuple(tensor.shape)}"
            )
        if tensor.shape[-1] == 3:
            tensor = tensor.permute(0, 1, 4, 2, 3)
        if tensor.shape[2] != 3:
            raise ValueError(f"Expected 3 channels, got {tuple(tensor.shape)}")
        return tensor.to(dtype=torch.float32)

    def _io_resize(self, pixel_values: torch.Tensor) -> torch.Tensor:
        resize = self.io_image_cfg.get("resize", [224, 224])
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

    def _io_normalize_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.max() > 1.0:
            pixel_values = pixel_values / 255.0
        pixel_values = pixel_values.clamp(0.0, 1.0)
        mean = torch.tensor([0.485, 0.456, 0.406], device=pixel_values.device).view(
            1, 1, 3, 1, 1
        )
        std = torch.tensor([0.229, 0.224, 0.225], device=pixel_values.device).view(
            1, 1, 3, 1, 1
        )
        return (pixel_values - mean) / std

    def _io_process_task_descriptions(
        self, env_obs: Mapping[str, Any], batch_size: int
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

    def _io_prepare_proprio(
        self, env_obs: Mapping[str, Any], batch_size: int, device: torch.device
    ) -> torch.Tensor:
        state = env_obs.get("states")
        if state is None:
            return torch.zeros((batch_size, 20), dtype=torch.float32, device=device)

        if isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).float()
        elif isinstance(state, torch.Tensor):
            state_tensor = state.float()
        else:
            state_tensor = torch.as_tensor(state).float()

        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        if state_tensor.shape[0] != batch_size:
            if state_tensor.shape[0] == 1:
                state_tensor = state_tensor.expand(batch_size, -1)
            else:
                raise ValueError(
                    f"Inconsistent state batch size: {state_tensor.shape[0]} vs expected {batch_size}"
                )

        state_tensor = state_tensor.to(device=device)
        eef_pos = state_tensor[:, :3]
        axis_angle = state_tensor[:, 3:6]
        rot_mat = self._axis_angle_to_rotation_matrix(axis_angle)
        rot6d = self._rotation_matrix_to_6d(rot_mat)
        proprio_10d = torch.cat(
            [
                eef_pos,
                rot6d,
                torch.zeros(batch_size, 1, dtype=torch.float32, device=device),
            ],
            dim=-1,
        )
        return torch.cat([proprio_10d, torch.zeros_like(proprio_10d)], dim=-1)

    def _convert_ee6d_to_axis_angle(self, action: torch.Tensor) -> torch.Tensor:
        pos = action[..., :3]
        rot6d = action[..., 3:9]
        original_shape = rot6d.shape
        rot6d_np = rot6d.reshape(-1, 6).detach().cpu().to(torch.float32).numpy()
        axis_angle_np = rotate6d_to_axis_angle(rot6d_np)
        axis_angle = torch.from_numpy(axis_angle_np).to(action.device, action.dtype)
        axis_angle = axis_angle.reshape(*original_shape[:-1], 3)
        gripper = action[..., 9:10]
        return torch.cat([pos, axis_angle, gripper], dim=-1)

    def _io_normalize_gripper(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape[-1] < 1:
            return action
        gripper = action[..., -1:]
        if torch.any(torch.abs(gripper) > 5.0):
            gripper = torch.sigmoid(gripper)
        gripper = torch.where(
            gripper > 0.5, torch.ones_like(gripper), -torch.ones_like(gripper)
        )
        gripper_range = self.io_action_cfg.get("gripper_range", [-1.0, 1.0])
        if isinstance(gripper_range, (list, tuple)) and len(gripper_range) == 2:
            min_v = float(gripper_range[0])
            max_v = float(gripper_range[1])
        else:
            min_v, max_v = -1.0, 1.0
        gripper = gripper.clamp(min_v, max_v)
        return torch.cat([action[..., :-1], gripper], dim=-1)

    def _io_target_action_dim(self, action: torch.Tensor) -> int:
        configured = self.io_action_cfg.get("action_dim")
        if isinstance(configured, (int, float)):
            return int(configured)
        env_input = str(self.io_action_cfg.get("env_input", "axis_angle"))
        if env_input == "axis_angle":
            return 7
        return int(action.shape[-1])

    def _io_transform_output(self, action: torch.Tensor) -> torch.Tensor:
        transformed = action.to(dtype=torch.float32)
        model_output = str(self.io_action_cfg.get("model_output", "ee6d"))
        env_input = str(self.io_action_cfg.get("env_input", "axis_angle"))
        if (
            model_output == "ee6d"
            and env_input == "axis_angle"
            and transformed.shape[-1] >= 10
        ):
            transformed = self._convert_ee6d_to_axis_angle(transformed)
        transformed = self._io_normalize_gripper(transformed)
        target_dim = self._io_target_action_dim(transformed)
        return self._convert_action_to_target_dim(transformed, target_dim)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._lerobot_policy.predict_action_chunk(batch)

    def predict_action_batch(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "eval",
        sampling_params: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        """Generate actions for rollout (inference).

        Args:
            env_obs: Observation dictionary with images, states, prompts
            mode: "train" or "eval" mode
            sampling_params: Optional sampling parameters (e.g., num_steps)

        Returns:
            Tuple of (actions, None)
        """
        del mode
        del sampling_params
        batch = self.convert_obs(env_obs)
        if OBS_LANGUAGE_TOKENS not in batch:
            raise ValueError(
                "env_obs must be a LeRobot XVLAPolicy batch with 'observation.language.tokens'"
            )
        if not any(
            feature_key in batch
            for feature_key in self._lerobot_policy.config.image_features
        ):
            raise ValueError(
                "env_obs must include at least one configured image feature key for XVLAPolicy"
            )

        with torch.no_grad():
            actions = self.predict_action_chunk(batch)

        self.logger.info(
            f"shape after processing: {actions.shape}, dtype={actions.dtype}"
        )
        return actions.to(dtype=torch.float32), None

    def libero_state_to_lerobot(self, state: torch.Tensor) -> torch.Tensor:
        """Convert Libero state to LeRobot state."""
        eef_pos = state[..., :3]
        axis_angle = state[..., 3:6]
        rot_mat = rotvec_to_matrix(axis_angle)
        rot6d = rot_mat[:, :, :2].reshape(rot_mat.shape[0], 6)
        return torch.cat(
            [
                eef_pos,
                rot6d,
                state[..., 6:],
            ],
            dim=-1,
        )

    def convert_obs(self, env_obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Convert raw environment observation to model input format."""
        lerobot_obs: dict[str, torch.Tensor] = {
            f"observation.images.{key}": env_obs[key]
            for key in env_obs
            if "image" in key
        }
        lerobot_obs["observation.state"] = self.libero_state_to_lerobot(
            env_obs["states"]
        )
        lerobot_obs["task"] = env_obs["task_descriptions"]

        lerobot_obs["observation.language.tokens"] = self.tokenizer(
            env_obs["task_descriptions"],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_max_length,
            return_tensors="pt",
        ).input_ids.to(device=next(self.parameters()).device)

        return self.input_transform(env_obs)

    def sample_actions(
        self,
        observation: Any,
        num_steps: Optional[int] = None,
        noise_level: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Sample actions using flow-matching.

        This is a simplified interface for direct action sampling.
        """
        actions, _ = self.predict_action_batch(env_obs=observation)
        return actions

    def get_log_prob_value(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        chains: torch.Tensor,
        timesteps: torch.Tensor,
        compute_values: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute log-probabilities of actions under current policy.

        Args:
            images: List of image tensors [batch, C, H, W]
            img_masks: List of image masks [batch, H, W]
            lang_tokens: Language token IDs [batch, seq_len]
            lang_masks: Language attention masks [batch, seq_len]
            state: Proprioceptive state [batch, state_dim]
            chains: Action sequences [batch, num_steps, action_chunk, action_dim]
            timesteps: Denoising timesteps [batch, num_steps]
            compute_values: Whether to compute value estimates

        Returns:
            Tuple of (log_probs, values, entropy)
        """
        stacked_images = torch.stack(images, dim=1)
        if len(img_masks) > 0:
            stacked_mask = torch.stack(img_masks, dim=1)
        else:
            stacked_mask = self._get_default_image_mask(stacked_images)

        observations = {
            "pixel_values": stacked_images,
            "image_mask": stacked_mask,
            "input_ids": lang_tokens,
            "attention_mask": lang_masks,
            "proprio": state,
            "domain_id": torch.full(
                (lang_tokens.shape[0],),
                self.domain_id,
                dtype=torch.long,
                device=lang_tokens.device,
            ),
        }

        forward_inputs = {
            "chains": chains,
            "timesteps": timesteps,
            "observations": observations,
        }

        outputs = self.default_forward(forward_inputs)

        return (
            outputs["logprobs"],
            outputs["values"]
            if outputs["values"] is not None
            else torch.zeros(chains.shape[0], device=chains.device),
            outputs["entropy"],
        )

    def compute_flow_matching_loss(
        self, actions: torch.Tensor, observations: Any, **kwargs
    ) -> torch.Tensor:
        """Compute flow-matching loss for training."""
        data = {
            "observations": observations,
            "actions": actions,
        }
        outputs = self.sft_forward(data)
        return outputs["loss"]

    def input_transform(
        self,
        env_obs: dict[str, Any],
    ) -> dict[str, Any]:
        """Transform environment observations to model format.

        When adapter is used, it returns pre-processed tensors directly.
        Otherwise, processes raw environment observations.

        Args:
            env_obs: Environment observation dictionary with images, states, task_descriptions

        Returns:
            Dictionary with pixel_values, input_ids, attention_mask, proprio, domain_id
        """
        device = next(self.parameters()).device
        target_dtype = self._get_target_dtype()

        view_keys = self._io_get_view_mapping()
        image_views: list[torch.Tensor] = []
        batch_size: int | None = None
        for obs_key in view_keys:
            image = env_obs.get(obs_key)
            if image is None:
                continue
            image_tensor = self._io_to_bvchw(image)
            if batch_size is None:
                batch_size = int(image_tensor.shape[0])
            elif image_tensor.shape[0] != batch_size:
                raise ValueError(
                    f"Inconsistent batch size for key '{obs_key}': expected {batch_size}, got {image_tensor.shape[0]}"
                )
            image_views.append(image_tensor)

        if len(image_views) == 0:
            raise ValueError(f"No image found for configured views: {view_keys}")

        pixel_values = torch.cat(image_views, dim=1)
        pixel_values = self._io_resize(pixel_values)
        pixel_values = self._io_normalize_image(pixel_values)
        if batch_size is None:
            batch_size = int(pixel_values.shape[0])
        image_mask = torch.ones(
            batch_size,
            pixel_values.shape[1],
            dtype=torch.bool,
            device=pixel_values.device,
        )
        proprio = self._io_prepare_proprio(env_obs, batch_size, device)
        batch_size = pixel_values.shape[0]

        # Only tokenization and device transfer needed
        task_desc = self._io_process_task_descriptions(env_obs, batch_size)

        tokenized = self.tokenizer(
            task_desc,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_max_length,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids.to(device)
        attention_mask = tokenized.attention_mask.to(device)

        domain_id = self._prepare_domain_id(
            env_obs.get("domain_id"), batch_size, device
        )

        return {
            "pixel_values": pixel_values.to(device=device, dtype=target_dtype),
            "image_mask": image_mask.to(device=device),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "proprio": proprio.to(device=device, dtype=target_dtype),
            "domain_id": domain_id,
        }

    def output_transform(self, outputs: dict[str, Any]) -> dict[str, Any]:
        """Transform model outputs to environment format."""
        return outputs

    def precision_processor(self, processed_obs: dict[str, Any]) -> dict[str, Any]:
        """Process precision/dtype of observations."""
        dtype = torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float32
        device = next(self.parameters()).device

        for key, value in processed_obs.items():
            if isinstance(value, torch.Tensor):
                if key in {"input_ids", "attention_mask", "domain_id"}:
                    processed_obs[key] = value.to(device=device)
                else:
                    processed_obs[key] = value.to(dtype=dtype, device=device)

        return processed_obs
