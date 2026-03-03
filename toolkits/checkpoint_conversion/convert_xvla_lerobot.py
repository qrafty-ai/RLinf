#!/usr/bin/env python3
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

"""Convert LeRobot XVLA checkpoint to RLinf format.

LeRobot checkpoint structure:
- model.safetensors: Contains VLM + policy weights
- config.json: XVLA configuration

RLinf checkpoint structure:
- model.safetensors: Contains mapped weights
- config.json: RLinf-compatible config

Usage:
    python convert_xvla_lerobot.py \
        /path/to/lerobot/xvla-libero \
        /path/to/output/xvla-libero-rlinf \
        --verify
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch


def load_lerobot_config(checkpoint_dir: Path) -> dict:
    """Load LeRobot config.json."""
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, "r") as f:
        return json.load(f)


def load_lerobot_weights(checkpoint_dir: Path) -> Dict[str, torch.Tensor]:
    """Load model.safetensors."""
    weights_path = checkpoint_dir / "model.safetensors"
    if not weights_path.exists():
        # Try pytorch bin
        weights_path = checkpoint_dir / "model.bin"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Weights not found: {checkpoint_dir}/model.safetensors or model.bin"
            )
        return torch.load(weights_path, map_location="cpu")
    
    try:
        from safetensors.torch import load_file
        return load_file(weights_path)
    except ImportError:
        raise ImportError(
            "safetensors is required to load .safetensors files. "
            "Install with: pip install safetensors"
        )


def _map_weight_name(old_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Map a LeRobot checkpoint key to RLinf XVLA key.

    Returns:
        (new_key, skip_reason)
        - new_key is None when key should be skipped.
        - skip_reason is set when key is skipped.
    """
    new_name = old_name

    # Remove top-level "model." prefix
    if new_name.startswith("model."):
        new_name = new_name[6:]

    # VLM keys are structurally aligned
    if new_name.startswith("vlm."):
        return new_name, None

    # LeRobot transformer -> RLinf policy_head
    if not new_name.startswith("transformer."):
        return None, "unsupported_prefix"

    key = new_name[len("transformer.") :]

    # Direct module remaps (handles both nn.Linear and DomainAwareLinear)
    direct_map = {
        # DomainAwareLinear case (use_hetero_proj=True)
        "action_encoder.fc.weight": "policy_head.action_encoder.fc.weight",
        "action_encoder.bias.weight": "policy_head.action_encoder.bias.weight",
        "action_decoder.fc.weight": "policy_head.action_decoder.fc.weight",
        "action_decoder.bias.weight": "policy_head.action_decoder.bias.weight",
        "vlm_proj.fc.weight": "policy_head.vlm_proj.fc.weight",
        "vlm_proj.bias.weight": "policy_head.vlm_proj.bias.weight",
        "aux_visual_proj.fc.weight": "policy_head.aux_visual_proj.fc.weight",
        "aux_visual_proj.bias.weight": "policy_head.aux_visual_proj.bias.weight",
        # nn.Linear case (use_hetero_proj=False)
        "vlm_proj.weight": "policy_head.vlm_proj.weight",
        "vlm_proj.bias": "policy_head.vlm_proj.bias",
        "aux_visual_proj.weight": "policy_head.aux_visual_proj.weight",
        "aux_visual_proj.bias": "policy_head.aux_visual_proj.bias",
        # Shared params
        "pos_emb": "policy_head.pos_emb",
        "norm.weight": "policy_head.norm.weight",
        "norm.bias": "policy_head.norm.bias",
        "soft_prompt_hub.weight": "policy_head.soft_prompt_hub.weight",
    }
    if key in direct_map:
        return direct_map[key], None

    # Block-wise remap:
    # blocks.i.attn.(qkv|proj).{weight,bias} -> policy_head.blocks.i.attn...
    # blocks.i.norm1|norm2.{weight,bias}      -> policy_head.blocks.i.norm...
    # blocks.i.mlp.fc1|fc2.{weight,bias}      -> policy_head.blocks.i.mlp.fc1|fc2...
    m = re.match(r"^blocks\.(\d+)\.(.+)$", key)
    if not m:
        return None, "unrecognized_transformer_key"

    block_idx = m.group(1)
    tail = m.group(2)

    if tail.startswith("attn.") or tail.startswith("norm1.") or tail.startswith("norm2."):
        return f"policy_head.blocks.{block_idx}.{tail}", None

    if tail.startswith("mlp.fc1."):
        return f"policy_head.blocks.{block_idx}.mlp.fc1.{tail[len('mlp.fc1.'):]}", None
    if tail.startswith("mlp.fc2."):
        return f"policy_head.blocks.{block_idx}.mlp.fc2.{tail[len('mlp.fc2.'):]}", None

    return None, "unrecognized_block_tail"


def convert_weight_mapping(lerobot_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map LeRobot weight names to RLinf names.
    
    LeRobot naming convention:
    - model.vlm.* -> Florence2 weights (same in RLinf)
    - model.transformer.* -> Policy transformer
    - model.action_* -> Action projections
    - model.time_mlp_* -> Time embeddings
    - model.soft_prompt_hub.* -> Soft prompts
    
    RLinf naming convention:
    - vlm.* -> Florence2 (same)
    - policy_head.* -> All policy components
    """
    rlinf_weights: Dict[str, torch.Tensor] = {}
    skipped: Dict[str, int] = {}

    for old_name, tensor in lerobot_weights.items():
        new_name, skip_reason = _map_weight_name(old_name)
        if new_name is None:
            reason = skip_reason or "unknown"
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        rlinf_weights[new_name] = tensor

    # Florence2 shared embedding can be absent in source safetensors.
    shared_key = "vlm.language_model.model.shared.weight"
    encoder_embed_key = "vlm.language_model.model.encoder.embed_tokens.weight"
    if shared_key not in rlinf_weights and encoder_embed_key in rlinf_weights:
        rlinf_weights[shared_key] = rlinf_weights[encoder_embed_key].clone()

    if skipped:
        print("Skipped keys summary:")
        for reason, count in sorted(skipped.items()):
            print(f"  - {reason}: {count}")

    return rlinf_weights


def convert_config(lerobot_config: dict, lerobot_weights: Optional[Dict[str, torch.Tensor]] = None) -> dict:
    """Convert LeRobot config to RLinf format.
    
    LeRobot config has nested structure that needs to be adapted.
    """
    # Flatten nested policy config when loading from newer LeRobot checkpoints
    if "policy" in lerobot_config and isinstance(lerobot_config["policy"], dict):
        policy_cfg = lerobot_config["policy"]
    else:
        policy_cfg = lerobot_config

    chunk_size = policy_cfg.get("chunk_size", 32)
    max_action_dim = policy_cfg.get("max_action_dim", 20)

    detected_hetero_proj = False
    if lerobot_weights is not None:
        detected_hetero_proj = any(
            k.endswith("transformer.vlm_proj.fc.weight") or k.endswith("transformer.aux_visual_proj.fc.weight")
            for k in lerobot_weights.keys()
        )

    # Extract relevant fields
    rlinf_config = {
        "model_type": "xvla",
        "config_name": policy_cfg.get("config_name", "xvla_libero"),
        "num_action_chunks": chunk_size,
        "action_dim": max_action_dim,
        "precision": "bfloat16",
        "is_lora": False,
        "xvla": {
            # Florence2 config (pass through)
            "florence_config": policy_cfg.get("florence_config", {}),
            
            # Tokenizer
            "tokenizer_name": policy_cfg.get("tokenizer_name", "facebook/bart-base"),
            "tokenizer_max_length": policy_cfg.get("tokenizer_max_length", 96),
            "tokenizer_padding_side": policy_cfg.get("tokenizer_padding_side", "right"),
            "domain_id": policy_cfg.get("domain_id", 3),
            
            # SoftPromptedTransformer
            "hidden_size": policy_cfg.get("hidden_size", 1024),
            "depth": policy_cfg.get("depth", 24),
            "num_heads": policy_cfg.get("num_heads", 16),
            "mlp_ratio": policy_cfg.get("mlp_ratio", 4.0),
            "num_domains": policy_cfg.get("num_domains", 30),
            "len_soft_prompts": policy_cfg.get("len_soft_prompts", 32),
            "dim_time": policy_cfg.get("dim_time", 32),
            "max_len_seq": policy_cfg.get("max_len_seq", 512),
            "use_hetero_proj": detected_hetero_proj or policy_cfg.get("use_hetero_proj", False),
            
            # Flow-matching
            "noise_method": "flow_matching",
            "num_steps": policy_cfg.get("num_denoising_steps", 10),
            "sigma_min": 0.001,
            "sigma_max": 1.0,
            "rho": 7.0,
            "time_schedule": "lognorm",
            "chunk_size": chunk_size,
            "n_action_steps": policy_cfg.get("n_action_steps", chunk_size),
            
            # Action space
            "action_mode": policy_cfg.get("action_mode", "ee6d"),
            "max_action_dim": max_action_dim,
            
            # Observation
            "num_images_in_input": policy_cfg.get("num_images_in_input", 2),
            "use_proprio": policy_cfg.get("use_proprio", True),
            "max_state_dim": policy_cfg.get("max_state_dim", 20),
            
            # Training
            "dtype": "bfloat16",
            "freeze_vision_encoder": True,
            "freeze_language_encoder": True,
            "train_policy_transformer": True,
            "train_soft_prompts": True,
            
            # RL
            "add_value_head": False,
        }
    }
    
    return rlinf_config


def verify_conversion(
    lerobot_weights: Dict[str, torch.Tensor],
    rlinf_weights: Dict[str, torch.Tensor],
    checkpoint_dir: Path,
) -> bool:
    """Verify conversion by checking all weights were mapped.
    
    Args:
        lerobot_weights: Original LeRobot weights
        rlinf_weights: Converted RLinf weights
        checkpoint_dir: Checkpoint directory for info
        
    Returns:
        True if verification passed
    """
    print("\n=== Verification Report ===")
    
    # Check all LeRobot weights were converted
    converted_count = 0
    missing_keys = []
    
    for key in lerobot_weights.keys():
        mapped_key, skip_reason = _map_weight_name(key)
        if mapped_key is None:
            continue

        if mapped_key in rlinf_weights:
            converted_count += 1
        else:
            missing_keys.append(key)
    
    print(f"Total weights: {len(lerobot_weights)}")
    print(f"Successfully converted: {converted_count}")
    print(f"Missing: {len(missing_keys)}")
    
    if missing_keys:
        print(f"\nMissing keys (first 10):")
        for key in missing_keys[:10]:
            print(f"  - {key}")
    
    # Check shapes match
    shape_mismatches = []
    for old_key, old_tensor in lerobot_weights.items():
        mapped_key, _ = _map_weight_name(old_key)
        if mapped_key is None or mapped_key not in rlinf_weights:
            continue
        new_shape = rlinf_weights[mapped_key].shape
        if old_tensor.shape != new_shape:
            shape_mismatches.append((old_key, old_tensor.shape, new_shape))
    
    if shape_mismatches:
        print(f"\nShape mismatches: {len(shape_mismatches)}")
        for old_key, old_shape, new_shape in shape_mismatches[:5]:
            print(f"  {old_key}: {old_shape} -> {new_shape}")
    
    success = len(missing_keys) == 0 and len(shape_mismatches) == 0
    
    if success:
        print("\n✓ Verification PASSED")
    else:
        print("\n✗ Verification FAILED")
    
    return success


def save_checkpoint(
    output_dir: Path,
    weights: Dict[str, torch.Tensor],
    config: dict,
    use_safetensors: bool = True,
):
    """Save converted checkpoint.
    
    Args:
        output_dir: Output directory
        weights: Converted weights
        config: RLinf config
        use_safetensors: Use safetensors format
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save weights
    if use_safetensors:
        try:
            from safetensors.torch import save_file
            save_file(weights, output_dir / "model.safetensors")
            print(f"Saved weights to {output_dir / 'model.safetensors'}")
        except ImportError:
            print("safetensors not available, saving as .bin")
            torch.save(weights, output_dir / "model.bin")
            print(f"Saved weights to {output_dir / 'model.bin'}")
    else:
        torch.save(weights, output_dir / "model.bin")
        print(f"Saved weights to {output_dir / 'model.bin'}")
    
    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {output_dir / 'config.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot XVLA checkpoint to RLinf format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic conversion
    python convert_xvla_lerobot.py \
        /path/to/lerobot/xvla-libero \
        /path/to/output/xvla-libero-rlinf
    
    # With verification
    python convert_xvla_lerobot.py \
        /path/to/lerobot/xvla-libero \
        /path/to/output/xvla-libero-rlinf \
        --verify
    
    # Save as .bin instead of .safetensors
    python convert_xvla_lerobot.py \
        /path/to/lerobot/xvla-libero \
        /path/to/output/xvla-libero-rlinf \
        --no-safetensors
        """,
    )
    parser.add_argument(
        "checkpoint_dir",
        type=Path,
        help="Path to LeRobot checkpoint directory (containing model.safetensors and config.json)",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for RLinf checkpoint",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify conversion by checking weight mappings",
    )
    parser.add_argument(
        "--no-safetensors",
        action="store_true",
        help="Save as .bin instead of .safetensors",
    )
    
    args = parser.parse_args()
    
    print(f"XVLA Checkpoint Converter")
    print(f"=" * 50)
    print(f"Input:  {args.checkpoint_dir}")
    print(f"Output: {args.output_dir}")
    print()
    
    # 1. Load LeRobot checkpoint
    print("Step 1: Loading LeRobot checkpoint...")
    try:
        lerobot_config = load_lerobot_config(args.checkpoint_dir)
        lerobot_weights = load_lerobot_weights(args.checkpoint_dir)
        print(f"  ✓ Loaded {len(lerobot_weights)} weights")
        print(f"  ✓ Config: {lerobot_config.get('config_name', 'unknown')}")
    except Exception as e:
        print(f"  ✗ Error loading checkpoint: {e}")
        sys.exit(1)
    
    # 2. Convert weights
    print("\nStep 2: Converting weight names...")
    try:
        rlinf_weights = convert_weight_mapping(lerobot_weights)
        print(f"  ✓ Converted {len(rlinf_weights)} weights")
    except Exception as e:
        print(f"  ✗ Error converting weights: {e}")
        sys.exit(1)
    
    # 3. Convert config
    print("\nStep 3: Converting config...")
    try:
        rlinf_config = convert_config(lerobot_config, lerobot_weights)
        print(f"  ✓ Converted config")
    except Exception as e:
        print(f"  ✗ Error converting config: {e}")
        sys.exit(1)
    
    # 4. Verify (optional)
    if args.verify:
        print("\nStep 4: Verifying conversion...")
        success = verify_conversion(lerobot_weights, rlinf_weights, args.checkpoint_dir)
        if not success:
            print("\n⚠ Warning: Verification found issues")
    
    # 5. Save
    print("\nStep 5: Saving RLinf checkpoint...")
    try:
        save_checkpoint(
            args.output_dir,
            rlinf_weights,
            rlinf_config,
            use_safetensors=not args.no_safetensors,
        )
        print(f"  ✓ Saved to {args.output_dir}")
    except Exception as e:
        print(f"  ✗ Error saving checkpoint: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("Conversion complete!")
    print(f"  Input:  {args.checkpoint_dir}")
    print(f"  Output: {args.output_dir}")
    
    # Print usage hint
    print("\nNext steps:")
    print(f"  1. Copy the converted checkpoint to your model directory:")
    print(f"     cp -r {args.output_dir} /path/to/your/models/")
    print(f"  2. Update your training config to use:")
    print(f"     actor:")
    print(f"       model:")
    print(f"         model_type: \"xvla\"")
    print(f"         model_path: \"/path/to/your/models/{args.output_dir.name}\"")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
