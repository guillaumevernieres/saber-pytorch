#!/usr/bin/env python3
"""Build a TorchScript ML balance surface emulator from an aibalance checkpoint.

This script bridges the aibalance training output format and the SABER
TorchBalance surface emulator contract.  It reads an aibalance checkpoint
(best_model.pt + normalization.pt), reconstructs the FFNN weights, and
exports a TorchScript module loadable by SABER C++ via torch::jit::load().

Aibalance checkpoint format (produced by aibalance/ufsemulator/training.py)
---------------------------------------------------------------------------
best_model.pt:
  {
    'model_state_dict': { ... },   # FFNN weights / biases
    'config': {
        'model': {
            'hidden_size':   int,
            'hidden_layers': int,
            'activation':    str,
        },
        'variables': {
            'input_variables':  [short_name, ...],
            'output_variables': [short_name, ...],
        },
        # Optional CF mappings (added during export):
        'metadata': {
            'input_cf_mapping':  {short_name: cf_name, ...},
            'output_cf_mapping': {short_name: cf_name, ...},
        },
    },
  }

normalization.pt (in the same directory as best_model.pt):
  {
    'input_mean':  Tensor[input_size],
    'input_std':   Tensor[input_size],
    'output_mean': Tensor[output_size],
    'output_std':  Tensor[output_size],
  }

Variable name resolution
------------------------
SABER field names must be CF-standard.  The script resolves names in this
priority order:

  1. Explicit --input-cf / --output-cf mappings supplied on the command line
     (format: "short_name:cf_name,short_name:cf_name")
  2. CF mappings recorded in checkpoint['config']['metadata']
  3. Short names as-is (fallback; prints a warning)

Usage
-----
    python build_ml_balance_emulator.py \\
        --checkpoint  /path/to/best_model.pt \\
        --output      ml_balance.ts \\
        [--input-cf   "sst:sea_water_potential_temperature,sss:sea_water_salinity"] \\
        [--output-cf  "aice:sea_ice_area_fraction"] \\
        [--input-levels  "0,0,0"]   \\   # comma-sep int per input feature
        [--output-levels "0"]            # comma-sep int per output feature
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from saber_pytorch.ml.ffnn import FFNN
from saber_pytorch.ml.ml_balance import FFNNSurfaceEmulator


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_kv_mapping(raw: Optional[str]) -> Dict[str, str]:
    """Parse 'k1:v1,k2:v2' into {'k1': 'v1', 'k2': 'v2'}."""
    if not raw:
        return {}
    result: Dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if ":" not in pair:
            raise ValueError(
                f"Invalid mapping entry '{pair}'. Expected 'short_name:cf_name'."
            )
        k, v = pair.split(":", 1)
        result[k.strip()] = v.strip()
    return result


def _parse_int_list(raw: Optional[str], length: int, label: str) -> List[int]:
    """Parse a comma-separated int list; default to all-zeros if omitted."""
    if not raw:
        return [0] * length
    vals = [int(x.strip()) for x in raw.split(",")]
    if len(vals) != length:
        raise ValueError(
            f"--{label}: expected {length} values, got {len(vals)}"
        )
    return vals


def _resolve_names(
    short_names: List[str],
    cli_mapping: Dict[str, str],
    checkpoint_mapping: Dict[str, str],
    label: str,
) -> List[str]:
    """Resolve short variable names to CF names."""
    resolved: List[str] = []
    for name in short_names:
        if name in cli_mapping:
            resolved.append(cli_mapping[name])
        elif name in checkpoint_mapping:
            resolved.append(checkpoint_mapping[name])
        else:
            print(
                f"Warning: no CF mapping for {label} variable '{name}'. "
                "Using short name as-is. SABER may not find this field."
            )
            resolved.append(name)
    return resolved


# ------------------------------------------------------------------
# Core build logic
# ------------------------------------------------------------------

def build_and_save(
    checkpoint_path: str,
    output_path: str,
    input_cf_raw: Optional[str] = None,
    output_cf_raw: Optional[str] = None,
    input_levels_raw: Optional[str] = None,
    output_levels_raw: Optional[str] = None,
) -> None:
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    # --- Load checkpoint ---
    checkpoint = torch.load(ckpt_file, weights_only=False)
    config = checkpoint["config"]
    model_cfg = config["model"]
    var_cfg = config["variables"]

    short_inputs: List[str] = list(var_cfg["input_variables"])
    short_outputs: List[str] = list(var_cfg["output_variables"])

    hidden_size: int = int(model_cfg["hidden_size"])
    hidden_layers: int = int(model_cfg["hidden_layers"])
    activation: str = str(model_cfg.get("activation", "gelu"))

    # --- Load normalization ---
    norm_file = ckpt_file.parent / "normalization.pt"
    if not norm_file.exists():
        raise FileNotFoundError(
            f"normalization.pt not found next to checkpoint: {norm_file}"
        )
    moments = torch.load(norm_file, weights_only=False)
    input_mean: torch.Tensor = moments["input_mean"].float()
    input_std: torch.Tensor = moments["input_std"].float()
    output_mean: torch.Tensor = moments["output_mean"].float()
    output_std: torch.Tensor = moments["output_std"].float()

    input_size = input_mean.shape[0]
    output_size = output_mean.shape[0]

    # --- Resolve CF names ---
    cli_in = _parse_kv_mapping(input_cf_raw)
    cli_out = _parse_kv_mapping(output_cf_raw)
    ckpt_meta = config.get("metadata", {})
    ckpt_in_cf: Dict[str, str] = dict(ckpt_meta.get("input_cf_mapping", {}))
    ckpt_out_cf: Dict[str, str] = dict(ckpt_meta.get("output_cf_mapping", {}))

    input_names = _resolve_names(short_inputs, cli_in, ckpt_in_cf, "input")
    output_names = _resolve_names(short_outputs, cli_out, ckpt_out_cf, "output")

    # --- Resolve level indices ---
    input_levels = _parse_int_list(input_levels_raw, input_size, "input-levels")
    output_levels = _parse_int_list(output_levels_raw, output_size, "output-levels")

    # --- Build emulator ---
    emulator = FFNNSurfaceEmulator(
        input_names=input_names,
        output_names=output_names,
        input_levels=input_levels,
        output_levels=output_levels,
        hidden_size=hidden_size,
        hidden_layers=hidden_layers,
        activation=activation,
    )

    # Load weights — map keys from UfsEmulatorFFNN to FFNN sub-module
    # UfsEmulatorFFNN stores weights as "network.*"
    # FFNNSurfaceEmulator wraps FFNN as "ffnn", which also uses "network.*"
    raw_state = checkpoint["model_state_dict"]
    mapped: Dict[str, torch.Tensor] = {}
    prefix_src = "network."
    prefix_dst = "ffnn.network."
    for k, v in raw_state.items():
        if k.startswith(prefix_src):
            mapped[prefix_dst + k[len(prefix_src):]] = v.float()
        # Normalization buffers are loaded separately via init_norm; skip them.
        # Skip conv1d keys if present (not supported in FFNNSurfaceEmulator).
        elif k.startswith("conv1d") or k.startswith("conv_activation"):
            print(f"Warning: skipping unsupported key '{k}' (conv1d not supported)")

    missing, unexpected = emulator.load_state_dict(mapped, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in state dict: {unexpected}")
    if missing:
        # Only FFNN normalization buffers are missing (loaded next); warn otherwise.
        non_norm = [
            k for k in missing
            if not any(k.endswith(s) for s in ("_mean", "_std"))
        ]
        if non_norm:
            raise RuntimeError(f"Missing keys in state dict: {non_norm}")

    emulator.init_norm(input_mean, input_std, output_mean, output_std)
    emulator.eval()

    # --- Export to TorchScript ---
    scripted = torch.jit.script(emulator)
    scripted.save(output_path)

    # --- Verification ---
    loaded = torch.jit.load(output_path)
    test_x = torch.randn(4, input_size)
    test_mask = torch.ones(4, 1)
    jac = loaded.jac_physical(test_x, test_mask)
    expected_shape = (4, output_size, input_size)
    if tuple(jac.shape) != expected_shape:
        raise RuntimeError(
            f"Verification failed: expected jac shape {expected_shape}, "
            f"got {tuple(jac.shape)}"
        )

    print(f"Saved: {output_path}")
    print(f"  input_names  ({input_size}): {input_names}")
    print(f"  input_levels          : {input_levels}")
    print(f"  output_names ({output_size}): {output_names}")
    print(f"  output_levels         : {output_levels}")
    print(f"  architecture: {hidden_layers}×{hidden_size}, activation={activation}")
    print(f"  jac_physical shape at runtime: [nnodes, {output_size}, {input_size}]")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build TorchScript ML balance surface emulator from aibalance checkpoint"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to best_model.pt (normalization.pt must be in the same directory)",
    )
    parser.add_argument("--output", required=True, help="Output .ts path")
    parser.add_argument(
        "--input-cf",
        help=(
            "CF name overrides for input variables. "
            "Format: 'short1:cf_name1,short2:cf_name2'"
        ),
    )
    parser.add_argument(
        "--output-cf",
        help="CF name overrides for output variables. Same format as --input-cf",
    )
    parser.add_argument(
        "--input-levels",
        help=(
            "Comma-separated vertical level indices for each input feature. "
            "Defaults to all zeros. "
            "Example: '0,0,127' for two surface vars and one atmospheric level."
        ),
    )
    parser.add_argument(
        "--output-levels",
        help=(
            "Comma-separated vertical level indices for each output feature. "
            "Defaults to all zeros."
        ),
    )
    args = parser.parse_args()

    build_and_save(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        input_cf_raw=args.input_cf,
        output_cf_raw=args.output_cf,
        input_levels_raw=args.input_levels,
        output_levels_raw=args.output_levels,
    )


if __name__ == "__main__":
    main()
