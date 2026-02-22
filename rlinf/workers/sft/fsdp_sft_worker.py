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

import os
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils import _pytree
from torch.utils.data import DataLoader

import rlinf.algorithms  # noqa: F401
from rlinf.config import SupportedModel
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.pytree import register_pytree_dataclasses
from rlinf.utils.utils import clear_memory


class FSDPSftWorker(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)

        self.cfg = cfg
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.cuda.current_device()

        self._component_placement = HybridComponentPlacement(cfg, Cluster())
        self.data_loader, self.data_config = self.build_dataloader()
        self.data_iter = iter(self.data_loader)
        self.global_step = 0

    def init_worker(self):
        self.setup_model_and_optimizer()

        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()

    def model_provider_func(self):
        model = get_model(self.cfg.actor.model)
        if model is not None:
            return model
        return super().model_provider_func()

    def _resolve_lerobot_dataset_args(self) -> tuple[str, str | None]:
        data_cfg = self.cfg.get("data", {})
        data_path = data_cfg.get("data_path") if data_cfg is not None else None

        dataset_repo_id = self.cfg.actor.model.get("repo_id")
        if dataset_repo_id is None:
            dataset_repo_id = self.cfg.actor.model.get("dataset_name")
        if dataset_repo_id is None:
            dataset_repo_id = "lerobot/xvla"

        dataset_root = None
        if isinstance(data_path, str):
            expanded_path = os.path.expanduser(data_path)
            if data_path.startswith(("~", "/", ".")) or os.path.isdir(expanded_path):
                dataset_root = expanded_path

        return str(dataset_repo_id), dataset_root

    def _extract_sft_batch(
        self, batch: Any
    ) -> tuple[dict[str, Any], torch.Tensor]:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            return batch[0], batch[1]

        if not isinstance(batch, dict):
            raise TypeError(
                f"Unsupported SFT batch type {type(batch)}. Expected tuple/list of (observation, action) or dict batch."
            )

        actions = batch.get("action")
        if actions is None:
            actions = batch.get("actions")
        if actions is None:
            raise KeyError(
                f"Unable to find action tensor in LeRobot batch. Available keys: {sorted(batch.keys())}"
            )

        observation = {}
        nested_observation = batch.get("observation")
        if isinstance(nested_observation, dict):
            observation.update(nested_observation)

        for key, value in batch.items():
            if isinstance(key, str) and key.startswith("observation."):
                observation[key] = value

        if "domain_ids" not in observation and "domain_id" in batch:
            observation["domain_ids"] = batch["domain_id"]
        if "domain_ids" not in observation:
            observation["domain_ids"] = torch.full(
                (actions.shape[0],),
                int(self.cfg.actor.model.get("domain_id", 0)),
                dtype=torch.long,
            )

        if "input_ids" not in observation and "observation.language.tokens" in observation:
            observation["input_ids"] = observation["observation.language.tokens"]
        if (
            "attention_mask" not in observation
            and "observation.language.attention_mask" in observation
        ):
            observation["attention_mask"] = observation[
                "observation.language.attention_mask"
            ]

        if "pixel_values" not in observation:
            if "observation.image" in observation:
                observation["pixel_values"] = observation["observation.image"]
            else:
                image_keys = sorted(
                    key
                    for key in observation
                    if isinstance(key, str) and key.startswith("observation.images.")
                )
                if image_keys:
                    observation["pixel_values"] = observation[image_keys[0]]

        return observation, actions

    def build_dataloader(self):
        if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.OPENPI]:
            import openpi.training.data_loader as openpi_data_loader

            from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

            config = get_openpi_config(
                self.cfg.actor.model.openpi.config_name,
                model_path=self.cfg.actor.model.model_path,
                batch_size=self.cfg.actor.micro_batch_size * self._world_size,
            )
            data_loader = openpi_data_loader.create_data_loader(
                config, framework="pytorch", shuffle=True
            )
            return data_loader, data_loader.data_config()
        elif SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.XVLA]:
            # Import LeRobot dataset components
            try:
                from lerobot.datasets.lerobot_dataset import LeRobotDataset
            except ImportError:
                raise ImportError(
                    "LeRobot is not installed. Install with: pip install 'lerobot[xvla]'"
                )

            # Get dataset configuration
            dataset_repo_id, dataset_root = self._resolve_lerobot_dataset_args()
            batch_size = self.cfg.actor.micro_batch_size * self._world_size
            video_backend = self.cfg.actor.model.get("video_backend", "torchcodec")

            # Create LeRobot dataset
            dataset = LeRobotDataset(
                repo_id=dataset_repo_id,
                root=dataset_root,
                video_backend=video_backend,
            )

            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.cfg.actor.get("num_workers", 4),
                pin_memory=torch.cuda.is_available(),
                drop_last=True,
            )

            # Return dataset and config
            return data_loader, {
                "preprocessors": None,
                "postprocessors": None,
                "dataset_repo_id": dataset_repo_id,
                "dataset_root": dataset_root,
                "video_backend": video_backend,
            }
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def run_training(self):
        with self.worker_timer():
            if self.cfg.actor.get("enable_offload", False):
                with self.device_lock:
                    self.load_param_and_grad(self.device)
                    self.load_optimizer(self.device)

            self.model.train()
            if hasattr(self.model, "gradient_checkpointing_disable"):
                self.model.gradient_checkpointing_disable()

            assert (
                self.cfg.actor.global_batch_size
                % (self.cfg.actor.micro_batch_size * self._world_size)
                == 0
            ), "global_batch_size is not divisible by micro_batch_size * world_size"

            self.gradient_accumulation = (
                self.cfg.actor.global_batch_size
                // self.cfg.actor.micro_batch_size
                // self._world_size
            )

            metrics = {}

            avg_loss = 0.0
            for idx in range(self.gradient_accumulation):
                backward_ctx = self.before_micro_batch(
                    self.model,
                    is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                )
                observation, actions = self._extract_sft_batch(next(self.data_iter))

                register_pytree_dataclasses(observation)
                observation = _pytree.tree_map(
                    lambda x: (
                        torch.as_tensor(x, device=self.device).contiguous().clone()
                        if x is not None
                        and isinstance(
                            x,
                            (
                                torch.Tensor,
                                np.ndarray,
                                list,
                                tuple,
                                int,
                                float,
                                bool,
                                np.number,
                            ),
                        )
                        else x
                    ),
                    observation,
                )
                actions = torch.as_tensor(actions).to(torch.float32)
                actions = actions.to(self.device)

                with self.amp_context:
                    losses = self.model(
                        forward_type=ForwardType.SFT,
                        data={"observation": observation, "actions": actions},
                    )
                    if isinstance(losses, dict):
                        if "loss" in losses:
                            losses = losses["loss"]
                        elif "losses" in losses:
                            losses = losses["losses"]
                        else:
                            raise KeyError(
                                f"SFT forward output dict missing loss key. Available keys: {sorted(losses.keys())}"
                            )
                    if isinstance(losses, (list, tuple)):
                        losses = torch.stack(losses)
                    elif not isinstance(losses, torch.Tensor):
                        losses = torch.tensor(
                            losses, device=self.device, dtype=torch.float32
                        )
                    loss = losses.mean()

                loss = loss / self.gradient_accumulation
                avg_loss += loss.item()
                with backward_ctx:
                    self.grad_scaler.scale(loss).backward()

            grad_norm, lr_list = self.optimizer_step()
            self.optimizer.zero_grad(set_to_none=True)

            # Collect stats
            lr_value = (
                lr_list[0] if len(lr_list) > 0 else self.optimizer.param_groups[0]["lr"]
            )
            grad_norm_value = (
                float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm
            )
            append_to_dict(
                metrics,
                {
                    "loss": avg_loss,
                    "learning_rate": lr_value,
                    "grad_norm": grad_norm_value,
                },
            )

            self.lr_scheduler.step()

            if self.global_step > 0 and self.global_step % 1000 == 0:
                clear_memory()

            train_metrics = {key: np.mean(value) for key, value in metrics.items()}
            train_metrics = all_reduce_dict(
                train_metrics, op=torch.distributed.ReduceOp.AVG
            )

            return train_metrics

    def set_global_step(self, global_step):
        self.global_step = global_step
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)
