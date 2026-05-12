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
    python emulators/ml_aice/scripts/build_surface_ml_balance_emulator.py \\
        --checkpoint  /path/to/best_model.pt \\
        --output      surface_ml_balance.ts \\
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

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from saber_pytorch.ml.ml_balance import FFNNSurfaceEmulator
from saber_pytorch.ml.cf_mappings import CF_ATM, CF_OCN

_BUILTIN_CF: Dict[str, str] = {**CF_ATM, **CF_OCN}


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
    """Resolve short variable names to CF names.

    Priority: CLI override > checkpoint metadata > built-in CF table > short name (warns).
    """
    resolved: List[str] = []
    for name in short_names:
        if name in cli_mapping:
            resolved.append(cli_mapping[name])
        elif name in checkpoint_mapping:
            resolved.append(checkpoint_mapping[name])
        elif name in _BUILTIN_CF:
            resolved.append(_BUILTIN_CF[name])
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
    mask_var_name_cli: Optional[str] = None,
    mask_min_cli: Optional[float] = None,
    mask_max_cli: Optional[float] = None,
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

    # --- Resolve domain mask ---
    # Priority: CLI args > checkpoint config > no masking
    if mask_var_name_cli is not None:
        mask_var_name: str = mask_var_name_cli
        mask_min: float = mask_min_cli if mask_min_cli is not None else 0.0
        mask_max: float = mask_max_cli if mask_max_cli is not None else 1.0
    else:
        domain_cfg = config.get("domain", {})
        mask_mode: str = str(domain_cfg.get("mask_mode", "none"))
        min_ice: float = float(domain_cfg.get("min_ice_concentration", 0.0))

        if mask_mode == "sea_ice":
            mask_var_name = _BUILTIN_CF.get("aice", "sea_ice_area_fraction")
            mask_min = min_ice
            mask_max = 1.0
        elif mask_mode == "ocean":
            mask_var_name = _BUILTIN_CF.get("aice", "sea_ice_area_fraction")
            mask_min = 0.0
            mask_max = min_ice
        else:
            mask_var_name = ""
            mask_min = 0.0
            mask_max = 1.0

    # --- Build emulator ---
    emulator = FFNNSurfaceEmulator(
        input_names=input_names,
        output_names=output_names,
        input_levels=input_levels,
        output_levels=output_levels,
        hidden_size=hidden_size,
        hidden_layers=hidden_layers,
        activation=activation,
        mask_var_name=mask_var_name,
        mask_min=mask_min,
        mask_max=mask_max,
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
    # Use mid-range mask_var so all 4 test nodes are within [mask_min, mask_max]
    if mask_var_name:
        test_mask_var = torch.full((4, 1), (mask_min + mask_max) / 2.0)
    else:
        test_mask_var = torch.ones(4, 1)
    jac = loaded.jac_physical(test_x, test_mask_var)
    expected_shape = (4, output_size, input_size)
    if tuple(jac.shape) != expected_shape:
        raise RuntimeError(
            f"Verification failed: expected jac shape {expected_shape}, "
            f"got {tuple(jac.shape)}"
        )

    # If masking is configured, also verify masked-out nodes return zero Jacobian.
    # When mask_var_name is one of the input features, the model reads the mask
    # value directly from the input column — so we must set that column too.
    if mask_var_name:
        tol = 1.0e-12
        mask_var_idx = loaded.mask_var_idx  # -1 if mask_var is not an input column

        def _make_inputs_with_mask(mask_val: float) -> torch.Tensor:
            x = test_x.clone()
            if mask_var_idx >= 0:
                x[:, mask_var_idx] = mask_val
            return x

        inputs_below = _make_inputs_with_mask(mask_min - 1.0e-3)
        inputs_above = _make_inputs_with_mask(mask_max + 1.0e-3)

        below = loaded.jac_physical(inputs_below, torch.full((4, 1), mask_min - 1.0e-3))
        above = loaded.jac_physical(inputs_above, torch.full((4, 1), mask_max + 1.0e-3))
        if float(below.abs().max()) > tol:
            raise RuntimeError(
                "Verification failed: Jacobian is non-zero below mask_min; "
                f"max abs = {float(below.abs().max()):.3e}"
            )
        if float(above.abs().max()) > tol:
            raise RuntimeError(
                "Verification failed: Jacobian is non-zero above mask_max; "
                f"max abs = {float(above.abs().max()):.3e}"
            )

    print(f"Saved: {output_path}")
    print(f"  input_names  ({input_size}): {input_names}")
    print(f"  input_levels          : {input_levels}")
    print(f"  output_names ({output_size}): {output_names}")
    print(f"  output_levels         : {output_levels}")
    print(f"  mask_var_name         : {mask_var_name or '(none)'}")
    if mask_var_name:
        print(f"  mask range            : [{mask_min}, {mask_max}]")
    print(f"  emulator type: surface ML balance")
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
    parser.add_argument(
        "--mask-var",
        help=(
            "CF name of the background variable used to gate the Jacobian "
            "(e.g. 'sea_ice_area_fraction').  Overrides the domain config "
            "stored in the checkpoint.  Omit to use the checkpoint config."
        ),
    )
    parser.add_argument(
        "--mask-min", type=float,
        help="Lower bound of the active range for --mask-var (inclusive).",
    )
    parser.add_argument(
        "--mask-max", type=float,
        help="Upper bound of the active range for --mask-var (inclusive).",
    )
    args = parser.parse_args()

    build_and_save(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        input_cf_raw=args.input_cf,
        output_cf_raw=args.output_cf,
        input_levels_raw=args.input_levels,
        output_levels_raw=args.output_levels,
        mask_var_name_cli=args.mask_var,
        mask_min_cli=args.mask_min,
        mask_max_cli=args.mask_max,
    )


if __name__ == "__main__":
    main()
