# OpenVLA LIBERO Evaluation Setup Guide

This document provides a complete guide for setting up the environment and running OpenVLA-OFT evaluation on the LIBERO benchmark using RLinf.

---

## Overview

**Purpose**: Run evaluation of OpenVLA-OFT (Orthogonal Finetuned Vision-Language-Action model) on LIBERO-10 manipulation tasks.

**What you'll get**:
- Fully configured Python environment with all dependencies
- Pre-trained OpenVLA-OFT model (RLinf/RLinf-OpenVLAOFT-LIBERO-130)
- LIBERO simulation environment (10 tasks)
- Evaluation pipeline with metrics logging

**Expected results** (single RTX 3090):
- Success@Once: ~87.5%
- Success@End: ~75%
- Evaluation time: ~3 minutes

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA RTX 3090 (24GB) | Multiple RTX 3090/A100 |
| VRAM | 24 GB | 40+ GB |
| RAM | 32 GB | 64 GB |
| Storage | 50 GB free | 100 GB SSD |
| CUDA | 12.0+ | 12.4+ |

### System Dependencies

Install system-level dependencies (requires sudo):

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    libgl1-mesa-glx \
    libosmesa6-dev \
    freeglut3-dev \
    ffmpeg \
    libglfw3-dev \
    libglew-dev \
    libomp-dev \
    libglfw3 \
    libglm-dev \
    libffi-dev \
    libjpeg-dev \
    libpng-dev \
    libvulkan1 \
    vulkan-tools
```

### Software Requirements

- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.11 (managed by uv)
- **Git**: For repository cloning
- **wget**: For downloading assets
- **NVIDIA Driver**: 525.60.11+ (for CUDA 12)

---

## Quick Start

```bash
# Navigate to repository
cd /path/to/RLinf

# Run installation (15-30 minutes)
bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero

# Activate environment
source .venv/bin/activate

# Download evaluation model (5-15 minutes)
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='RLinf/RLinf-OpenVLAOFT-LIBERO-130')
"

# Run evaluation (3-5 minutes)
bash examples/embodiment/eval_embodiment.sh libero_10_grpo_openvlaoft_eval
```

---

## Detailed Installation Flow

### Step 1: Run Installation Script

```bash
bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero
```

**Arguments explained**:
- `embodied`: Target installation type (embodied AI experiments)
- `--model openvla-oft`: Select OpenVLA-Orthogonal-Finetuning model
- `--env maniskill_libero`: Select LIBERO environment on ManiSkill simulator

**What happens**:

1. **Creates virtual environment**
   ```bash
   uv venv .venv --python 3.11.14
   ```

2. **Installs common embodied dependencies**
   ```bash
   uv sync --extra embodied --active
   uv pip install -r requirements/embodied/envs/common.txt
   ```
   Packages installed:
   - PyTorch 2.6+ with CUDA support
   - Ray (distributed execution)
   - Hydra-core 1.3 (configuration)
   - Gymnasium (RL interface)
   - SAPIEN (physics engine)
   - robosuite (robot simulation)

3. **Clones and installs LIBERO**
   ```bash
   git clone https://github.com/RLinf/LIBERO.git .venv/libero
   uv pip install -e .venv/libero
   export PYTHONPATH=/path/to/LIBERO:$PYTHONPATH
   ```

4. **Installs ManiSkill3**
   ```bash
   uv pip install git+https://github.com/haosulab/ManiSkill.git@v3.0.0b22
   ```

5. **Downloads ManiSkill assets** (~2GB)
   ```bash
   bash requirements/embodied/download_assets.sh --assets maniskill
   ```
   Assets include:
   - Robot URDF models
   - Object meshes
   - Textures and materials
   - Pre-trained assets

6. **Installs OpenVLA-OFT**
   ```bash
   uv pip install git+https://github.com/moojink/openvla-oft.git --no-build-isolation
   ```

7. **Installs Flash Attention 2.7.4**
   ```bash
   # Tries pre-built wheel first, falls back to source build
   install_flash_attn
   ```

8. **Cleanup**
   ```bash
   uv pip uninstall pynvml || true
   ```

### Step 2: Activate Environment

```bash
source .venv/bin/activate
```

**Verify environment variables are set**:
```bash
echo $NVIDIA_DRIVER_CAPABILITIES  # Should be 'all'
echo $PYTHONPATH                   # Should include LIBERO path
```

### Step 3: Download Pre-trained Model

The evaluation uses **RLinf/RLinf-OpenVLAOFT-LIBERO-130**, an 8B parameter model finetuned with GRPO.

```bash
python -c "
from huggingface_hub import snapshot_download
import os

model_id = 'RLinf/RLinf-OpenVLAOFT-LIBERO-130'
local_dir = os.path.expanduser('~/.cache/huggingface/hub/models--RLinf--RLinf-OpenVLAOFT-LIBERO-130')

print(f'Downloading {model_id}...')
print(f'Saving to: {local_dir}')

snapshot_download(repo_id=model_id, local_dir=local_dir)
print('Download complete!')
"
```

**Model details**:
- **Size**: ~18 GB (4 safetensors shards)
- **Architecture**: Prismatic VLA with OFT adapters
- **Training**: GRPO on LIBERO-130 (all tasks)
- **Performance**: 97.85% on LIBERO-130 benchmark
- **Normalization**: `libero_130_no_noops_trajall`

---

## Configuration

### Evaluation Config File

Location: `examples/embodiment/config/libero_10_grpo_openvlaoft_eval.yaml`

**Key settings**:

```yaml
# Task configuration
env:
  eval:
    task_suite_name: libero_10     # LIBERO-10 benchmark
    total_num_envs: 4              # Parallel environments (reduce for low VRAM)
    max_episode_steps: 512         # Max steps per episode
    is_eval: true

# Model configuration
actor:
  model:
    model_path: "/home/cc/.cache/huggingface/hub/models--RLinf--RLinf-OpenVLAOFT-LIBERO-130"
    model_type: "openvla_oft"
    unnorm_key: "libero_130_no_noops_trajall"  # Must match model's norm_stats
    is_lora: false

# Algorithm configuration
algorithm:
  sampling_params:
    do_sample: false    # Deterministic for evaluation
    temperature_eval: 1.0
  eval_rollout_epoch: 2  # Number of evaluation passes

# Resource configuration
rollout:
  enable_offload: true   # Enable CPU offload for low VRAM
```

### Adjusting for Your Hardware

**For GPUs with < 24GB VRAM**:
```yaml
env:
  eval:
    total_num_envs: 2    # Reduce parallel envs
```

**For multi-GPU setup**:
```yaml
cluster:
  num_nodes: 1
  component_placement:
    actor,env,rollout: all

# Increase environments for better throughput
env:
  eval:
    total_num_envs: 50
```

**Enable video recording** (requires additional ~10GB VRAM):
```yaml
env:
  eval:
    video_cfg:
      save_video: true
      info_on_video: true
```

---

## Running Evaluation

### Single Node Evaluation

```bash
# Method 1: Using eval script
bash examples/embodiment/eval_embodiment.sh libero_10_grpo_openvlaoft_eval

# Method 2: Direct Python command
source .venv/bin/activate
export EMBODIED_PATH=$(pwd)/examples/embodiment
python examples/embodiment/eval_embodied_agent.py \
    --config-path $EMBODIED_PATH/config/ \
    --config-name libero_10_grpo_openvlaoft_eval
```

### Expected Output

```
[INFO] RLinf is running on a cluster with 1 node and 1 accelerator
[INFO] Starting a local Ray instance
[INFO] Loading checkpoint shards: 100%|██████████| 4/4
[INFO] Evaluating Rollout Epochs: 100%|██████████| 2/2 [01:32<00:00]
[INFO] {'eval/success_once': 0.875, 'eval/success_at_end': 0.75, ...}
```

### Output Files

Results are saved to:
```
logs_eval/YYYYMMDD-HH:MM:SS/
├── eval.log                    # Full evaluation log
├── events.out.tfevents.*       # TensorBoard logs
└── video/eval/                 # Videos (if enabled)
    ├── task_0/
    ├── task_1/
    └── ...
```

---

## Verification

### Verify Installation

```bash
source .venv/bin/activate

# Check Python version
python --version  # Should be 3.11.14

# Check key packages
pip list | grep -E "torch|openvla|libero|mani_skill"

# Test imports
python -c "import libero; print('LIBERO ✓')"
python -c "import openvla_oft; print('OpenVLA-OFT ✓')"
python -c "import mani_skill; print('ManiSkill ✓')"
python -c "import torch; print(f'PyTorch {torch.__version__} ✓')"

# Check CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

### Verify Model Download

```bash
ls -lh ~/.cache/huggingface/hub/models--RLinf--RLinf-OpenVLAOFT-LIBERO-130/

# Should show:
# - config.json (7KB)
# - model-00001-of-00004.safetensors (~4.6GB)
# - model-00002-of-00004.safetensors (~4.6GB)
# - model-00003-of-00004.safetensors (~4.6GB)
# - model-00004-of-00004.safetensors (~250MB)
# - tokenizer.json, tokenizer.model, etc.
```

### Test Environment

```bash
source .venv/bin/activate
python -c "
from libero.libero import get_libero_suite
suite = get_libero_suite('libero_10')
env = suite.get_env(0)
print(f'Environment: {env}')
print(f'Task: {env.task_name}')
obs = env.reset()
print(f'Observation keys: {obs.keys()}')
"
```

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Error**: `RuntimeError: CUDA out of memory`

**Solution**: Reduce parallel environments
```yaml
env:
  eval:
    total_num_envs: 2  # Reduce from 4
```

Also disable videos:
```yaml
env:
  eval:
    video_cfg:
      save_video: false
```

#### 2. LIBERO Import Error

**Error**: `ModuleNotFoundError: No module named 'libero'`

**Solution**: Verify PYTHONPATH
```bash
source .venv/bin/activate
echo $PYTHONPATH  # Should include path to LIBERO
```

If missing, manually add:
```bash
export PYTHONPATH=/path/to/.venv/libero:$PYTHONPATH
```

#### 3. Model Loading Error

**Error**: `AssertionError: Action un-norm key not found in VLA norm_stats`

**Solution**: Check `unnorm_key` in config matches model's `norm_stats`:
```yaml
actor:
  model:
    unnorm_key: "libero_130_no_noops_trajall"  # For LIBERO-130 model
```

Verify model config:
```bash
python -c "
import json
with open('~/.cache/huggingface/hub/models--RLinf--RLinf-OpenVLAOFT-LIBERO-130/config.json') as f:
    config = json.load(f)
    print('Available norm_stats keys:', list(config['norm_stats'].keys()))
    print('Model unnorm_key:', config['unnorm_key'])
"
```

#### 4. Flash Attention Build Failure

**Error**: `flash-attn build failed`

**Solution**: Install from pre-built wheel
```bash
source .venv/bin/activate
uv pip install flash-attn==2.7.4.post1 --no-build-isolation
```

Or skip if not critical (slower but functional):
```bash
# Set in config
actor:
  model:
    attn_implementation: "sdpa"  # Use PyTorch SDPA instead
```

#### 5. Vulkan/OpenGL Errors

**Error**: `libGL error: failed to load driver`

**Solution**: Set environment variables
```bash
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export NVIDIA_DRIVER_CAPABILITIES=all
```

Add to `.venv/bin/activate` for persistence.

#### 6. Ray Initialization Failure

**Error**: `Ray failed to initialize`

**Solution**: Clear Ray temp files
```bash
rm -rf /tmp/ray/
ray stop --force
```

Then re-run evaluation.

---

## Environment Details

### Installed Packages (Key)

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.6.x | Deep learning framework |
| torchvision | 0.21.x | Computer vision utilities |
| openvla-oft | 0.0.1 | VLA model implementation |
| transformers | 4.40.1 | Model architecture |
| diffusers | 0.x | Diffusion model utilities |
| ManiSkill | 3.0.0b22 | Robot simulation |
| robosuite | 1.4.x | Robot control |
| gymnasium | 0.29.x | RL environment interface |
| ray | 2.x | Distributed execution |
| hydra-core | 1.3.2 | Configuration management |
| omegaconf | 2.3.x | Config parsing |
| flash-attn | 2.7.4 | Efficient attention |
| peft | 0.x | Parameter-efficient finetuning |
| accelerate | 0.x | Model acceleration |
| datasets | 2.x | Dataset utilities |

### Directory Structure

```
RLinf/
├── .venv/                          # Virtual environment
│   ├── bin/
│   │   ├── activate                # Source to activate
│   │   ├── python -> python3.11
│   │   └── [executables]
│   ├── lib/
│   │   └── python3.11/
│   │       └── site-packages/
│   │           ├── openvla_oft/    # OpenVLA-OFT code
│   │           ├── libero/         # LIBERO (symlink)
│   │           ├── mani_skill/     # ManiSkill3
│   │           └── [packages]
│   └── pyvenv.cfg
├── requirements/
│   ├── install.sh                  # Main installation script
│   ├── embodied/
│   │   ├── envs/
│   │   │   └── common.txt          # Common dependencies
│   │   └── download_assets.sh      # Asset downloader
│   └── README.md
├── examples/embodiment/
│   ├── config/
│   │   ├── libero_10_grpo_openvlaoft_eval.yaml
│   │   ├── model/
│   │   │   └── openvla_oft.yaml
│   │   └── env/
│   │       └── libero_10.yaml
│   ├── eval_embodiment.sh          # Evaluation script
│   └── eval_embodied_agent.py      # Evaluation entry point
├── rlinf/                          # RLinf package
│   ├── models/embodiment/
│   │   └── openvla_oft/            # Model implementation
│   ├── envs/
│   │   └── libero.py               # LIBERO wrapper
│   └── runners/
│       └── embodied_eval_runner.py # Evaluation runner
├── logs_eval/                      # Evaluation output
│   └── YYYYMMDD-HH:MM:SS/
└── README.Setup.OpenVLA.LIBERO.md  # This document
```

### Environment Variables

Set automatically by `.venv/bin/activate`:

```bash
NVIDIA_DRIVER_CAPABILITIES=all
VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
PYTHONPATH=/path/to/LIBERO:$PYTHONPATH
PATH=/path/to/.venv/bin:$PATH
VIRTUAL_ENV=/path/to/.venv
```

---

## Performance Optimization

### Single GPU Optimization

1. **Reduce parallel environments**: `total_num_envs: 2-4`
2. **Enable offloading**: `enable_offload: true`
3. **Disable videos**: `save_video: false`
4. **Use BF16 precision**: `precision: "bf16"`
5. **Gradient checkpointing**: `gradient_checkpointing: false` (for eval)

### Multi-GPU Optimization

1. **Increase environments**: `total_num_envs: 50-100`
2. **Enable FSDP sharding**: Adjust `fsdp_config.sharding_strategy`
3. **Pipeline parallelism**: `pipeline_stage_num: >1`

### Memory Profiling

```bash
# Monitor GPU memory during evaluation
watch -n1 nvidia-smi

# Profile with PyTorch
python -c "
import torch
torch.cuda.memory_summary()
"
```

---

## Next Steps

After successful evaluation:

1. **Analyze results**: Check TensorBoard logs
   ```bash
   tensorboard --logdir logs_eval/
   ```

2. **Watch videos**: If enabled, view in `logs_eval/.../video/eval/`

3. **Compare models**: Try different checkpoints
   - `RLinf/RLinf-OpenVLAOFT-LIBERO-90`
   - `RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-spatial`

4. **Run training**: Modify config for GRPO/PPO training

5. **Try other environments**:
   ```bash
   # Behavior-1K
   bash requirements/install.sh embodied --model openvla-oft --env behavior
   
   # RoboTwin
   bash requirements/install.sh embodied --model openvla-oft --env robotwin
   ```

---

## References

- [RLinf Documentation](https://rlinf.readthedocs.io/)
- [LIBERO Benchmark](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [OpenVLA-OFT](https://github.com/moojink/openvla-oft)
- [ManiSkill3](https://github.com/haosulab/ManiSkill)
- [RLinf/RLinf-OpenVLAOFT-LIBERO-130 Model](https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130)

---

## Support

For issues or questions:

1. Check [FAQ](https://rlinf.readthedocs.io/en/latest/rst_source/faq.html)
2. Review [CONTRIBUTING.md](CONTRIBUTING.md)
3. Open GitHub issue with:
   - Full error message
   - Environment details (`pip list`, `nvidia-smi`)
   - Config file used
   - Steps to reproduce

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-24  
**Tested On**: Ubuntu 22.04, RTX 3090, CUDA 12.4
