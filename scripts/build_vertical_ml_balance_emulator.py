#!/usr/bin/env python3
"""Build a TorchScript ML balance vertical emulator from a checkpoint.

This exports FFNNVerticalEmulator, which uses the TorchBalance vertical
contract:

  jac_physical(inputs, mask, row_indices, col_indices) -> [nnodes, nRequestedPairs]

Use this for profile-to-profile emulators such as Salt(z) from Temp(z).  Use
build_surface_ml_balance_emulator.py for single-level/surface ML balance
emulators.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from saber_pytorch.ml.ml_balance import FFNNSalinityProfileEmulator, FFNNVerticalEmulator


def _parse_kv_mapping(raw: Optional[str]) -> Dict[str, str]:
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


def _resolve_names(
    short_names: List[str],
    cli_mapping: Dict[str, str],
    checkpoint_mapping: Dict[str, str],
    label: str,
) -> List[str]:
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


def _infer_num_levels(
    raw_num_levels: Optional[int],
    input_size: int,
    output_size: int,
    n_inputs: int,
) -> int:
    if raw_num_levels is not None:
        num_levels = int(raw_num_levels)
    else:
        num_levels = int(output_size)
    if num_levels <= 0:
        raise ValueError("num_levels must be positive")
    if output_size != num_levels:
        raise ValueError(
            f"Vertical emulator expects output_size == num_levels; got "
            f"output_size={output_size}, num_levels={num_levels}"
        )
    if input_size != n_inputs * num_levels:
        raise ValueError(
            f"Vertical emulator expects input_size == n_inputs*num_levels; got "
            f"input_size={input_size}, n_inputs={n_inputs}, num_levels={num_levels}"
        )
    return num_levels


def build_and_save(
    checkpoint_path: str,
    output_path: str,
    input_cf_raw: Optional[str] = None,
    output_cf_raw: Optional[str] = None,
    num_levels_raw: Optional[int] = None,
) -> None:
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    checkpoint = torch.load(ckpt_file, weights_only=False)
    config = checkpoint["config"]
    model_cfg = config["model"]
    var_cfg = config["variables"]

    short_inputs: List[str] = list(var_cfg["input_variables"])
    short_outputs: List[str] = list(var_cfg["output_variables"])
    if len(short_outputs) != 1:
        raise ValueError(
            "Vertical ML balance builder requires exactly one output variable; "
            f"got {len(short_outputs)}"
        )

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

    input_size = int(input_mean.shape[0])
    output_size = int(output_mean.shape[0])
    emulator_type = str(model_cfg.get("emulator_type", "vertical")).lower()
    raw_num_levels = (
        num_levels_raw if num_levels_raw is not None else var_cfg.get("num_levels")
    )

    cli_in = _parse_kv_mapping(input_cf_raw)
    cli_out = _parse_kv_mapping(output_cf_raw)
    ckpt_meta = config.get("metadata", {})
    input_names = _resolve_names(
        short_inputs, cli_in, dict(ckpt_meta.get("input_cf_mapping", {})), "input"
    )
    output_names = _resolve_names(
        short_outputs, cli_out, dict(ckpt_meta.get("output_cf_mapping", {})), "output"
    )

    if emulator_type == "salinity_profile":
        if len(input_names) != 2:
            raise ValueError(
                "salinity_profile emulator expects exactly two inputs: "
                "temperature and layer thickness"
            )
        if raw_num_levels is None:
            raise ValueError("salinity_profile emulator requires variables.num_levels")
        source_num_levels = int(raw_num_levels)
        target_num_levels = int(var_cfg.get("target_num_levels", output_size))
        if input_size != target_num_levels or output_size != target_num_levels:
            raise ValueError(
                "salinity_profile checkpoint normalization must be on the reduced grid; "
                f"got input_size={input_size}, output_size={output_size}, "
                f"target_num_levels={target_num_levels}"
            )
        emulator = FFNNSalinityProfileEmulator(
            temperature_variable_name=input_names[0],
            thickness_variable_name=input_names[1],
            output_variable_name=output_names[0],
            source_num_levels=source_num_levels,
            target_num_levels=target_num_levels,
            hidden_size=int(model_cfg["hidden_size"]),
            hidden_layers=int(model_cfg.get("hidden_layers", 2)),
            activation=str(model_cfg.get("activation", "gelu")),
            use_conv1d=bool(model_cfg.get("use_conv1d", False)),
            conv_channels=int(model_cfg.get("conv_channels", 32)),
            conv_kernel_size=int(model_cfg.get("conv_kernel_size", 3)),
        )
        runtime_input_size = len(input_names) * source_num_levels
        num_levels = source_num_levels
    else:
        num_levels = _infer_num_levels(
            raw_num_levels,
            input_size,
            output_size,
            len(short_inputs),
        )
        emulator = FFNNVerticalEmulator(
            input_variable_names=input_names,
            output_variable_name=output_names[0],
            num_levels=num_levels,
            hidden_size=int(model_cfg["hidden_size"]),
            hidden_layers=int(model_cfg.get("hidden_layers", 2)),
            activation=str(model_cfg.get("activation", "gelu")),
            use_conv1d=bool(model_cfg.get("use_conv1d", False)),
            conv_channels=int(model_cfg.get("conv_channels", 32)),
            conv_kernel_size=int(model_cfg.get("conv_kernel_size", 3)),
        )
        runtime_input_size = input_size

    raw_state = checkpoint["model_state_dict"]
    mapped: Dict[str, torch.Tensor] = {}
    for k, v in raw_state.items():
        if k.startswith("network."):
            mapped["ffnn.network." + k[len("network."):]] = v.float()
        elif k.startswith("conv1d."):
            mapped["ffnn.conv1d." + k[len("conv1d."):]] = v.float()

    missing, unexpected = emulator.load_state_dict(mapped, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in state dict: {unexpected}")
    non_norm = [
        k for k in missing
        if not any(k.endswith(s) for s in ("_mean", "_std"))
    ]
    if non_norm:
        raise RuntimeError(f"Missing keys in state dict: {non_norm}")

    emulator.init_norm(input_mean, input_std, output_mean, output_std)
    emulator.eval()

    scripted = torch.jit.script(emulator)
    scripted.save(output_path)

    loaded = torch.jit.load(output_path)
    test_x = torch.randn(2, runtime_input_size)
    test_mask = torch.ones(2, 1)
    test_levels = min(output_size, num_levels)
    test_rows = torch.arange(test_levels, dtype=torch.long)
    test_cols = torch.arange(test_levels, dtype=torch.long)
    jac = loaded.jac_physical(test_x, test_mask, test_rows, test_cols)
    expected_shape = (2, test_levels)
    if tuple(jac.shape) != expected_shape:
        raise RuntimeError(
            f"Verification failed: expected jac shape {expected_shape}, "
            f"got {tuple(jac.shape)}"
        )

    print(f"Saved: {output_path}")
    print(f"  emulator type: {emulator_type}")
    print(
        f"  input_names  ({len(input_names)} variables, "
        f"{runtime_input_size} runtime features): {input_names}"
    )
    print(f"  output_names (1 variable, {output_size} levels): {output_names}")
    print(f"  num_levels: {num_levels}")
    if emulator_type == "salinity_profile":
        print(f"  target_num_levels: {output_size}")
    print(f"  jac_physical shape at runtime: [nnodes, nRequestedPairs]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build TorchScript ML balance vertical emulator from checkpoint"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to best_model.pt (normalization.pt must be in the same directory)",
    )
    parser.add_argument("--output", required=True, help="Output .ts path")
    parser.add_argument(
        "--input-cf",
        help="CF name overrides. Format: 'short1:cf_name1,short2:cf_name2'",
    )
    parser.add_argument(
        "--output-cf",
        help="CF name overrides. Format: 'short1:cf_name1,short2:cf_name2'",
    )
    parser.add_argument(
        "--num-levels",
        type=int,
        help="Number of vertical levels. Defaults to checkpoint config or output size.",
    )
    args = parser.parse_args()

    build_and_save(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        input_cf_raw=args.input_cf,
        output_cf_raw=args.output_cf,
        num_levels_raw=args.num_levels,
    )


if __name__ == "__main__":
    main()
