#!/bin/bash
# Copyright 2026 The RLinf Authors.
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

# Run script for XVLA SFT training on LIBERO dataset
#
# Usage:
#   bash run_xvla_sft.sh <config_name>
#   bash run_xvla_sft.sh libero_sft_xvla
#
# The config file should be located at: examples/sft/config/<config_name>.yaml

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Config name (default to libero_sft_xvla if not provided)
CONFIG_NAME="${1:-libero_sft_xvla}"
if [ "$#" -gt 0 ]; then
  shift
fi
EXTRA_ARGS=("$@")

echo "========================================"
echo "XVLA SFT Training"
echo "========================================"
echo "Config: ${CONFIG_NAME}"
echo "Script directory: ${SCRIPT_DIR}"
echo ""

# Set environment variables
export EMBODIED_PATH="${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}/../..:${PYTHONPATH}"

PYTHON_BIN="python"
if ! "${PYTHON_BIN}" -c "import hydra" >/dev/null 2>&1 && [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
fi

# Optional: Set CUDA devices
# export CUDA_VISIBLE_DEVICES=0

echo "Starting SFT training..."
echo ""

# Run the training script
"${PYTHON_BIN}" "${SCRIPT_DIR}/train_vla_sft.py" \
    --config-name "${CONFIG_NAME}" \
    --config-dir "${SCRIPT_DIR}/config" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "Training completed!"
