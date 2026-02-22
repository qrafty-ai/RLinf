#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=0
export EMBODIED_PATH="$REPO_ROOT/examples/embodiment"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

PYTHON_BIN="$REPO_ROOT/.venv/bin/python"

"$PYTHON_BIN" examples/sft/train_embodied_sft.py \
  --config-name xvla_sft \
  cluster.num_nodes=1 \
  actor.micro_batch_size=1 \
  actor.global_batch_size=8 \
  actor.model.domain_id=3 \
  actor.model.model_path="lerobot/xvla-base" \
  actor.model.repo_id="lerobot/libero_10" \
  data.data_path="$HOME/.cache/huggingface/lerobot"
