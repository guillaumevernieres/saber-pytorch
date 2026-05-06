#!/usr/bin/env python3
"""Write a TorchScript heave-salinity balance emulator.

The saved module follows the vertical TorchBalance contract used by SABER:
runtime inputs are packed as all levels of T, S, and layer thickness.  The
Jacobian maps temperature increments to balanced salinity increments; salinity
and thickness input blocks are zero for this physical balance.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from saber_pytorch.physics.heave_salinity import WeaverTSBalance


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return config


def _build_emulator(
    config: Dict[str, Any],
    temperature_name: str,
    salinity_name: str,
    thickness_name: str,
    output_name: str,
) -> WeaverTSBalance:
    model = config.get("model", {})
    input_names: List[str] = [temperature_name, salinity_name, thickness_name]
    output_names: List[str] = [output_name]

    return WeaverTSBalance(
        input_names=input_names,
        output_names=output_names,
        epsilon=float(model.get("epsilon", 1.0e-12)),
        epsilon_taper=float(model.get("epsilon_taper", 1.0e-3)),
        amplitude=float(model.get("amplitude", 1.0)),
        use_temperature_gradient_taper=bool(
            model.get("use_temperature_gradient_taper", True)
        ),
        suppress_shallow_weak_stratification=bool(
            model.get("suppress_shallow_weak_stratification", False)
        ),
        shallow_taper_depth_m=float(model.get("shallow_taper_depth_m", 50.0)),
        shallow_epsilon_taper=float(
            model.get("shallow_epsilon_taper", 1.0e-4)
        ),
    ).eval()


def _verify_saved_module(path: Path) -> None:
    loaded = torch.jit.load(str(path))
    nlevels = 4
    inputs = torch.randn(2, 3 * nlevels)
    mask = torch.ones(2, 1)
    rows = torch.arange(nlevels, dtype=torch.long)
    cols = torch.arange(nlevels, dtype=torch.long)
    jacobian = loaded.jac_physical(inputs, mask, rows, cols)

    expected_shape = (2, nlevels)
    if tuple(jacobian.shape) != expected_shape:
        raise RuntimeError(
            f"Verification failed: expected jacobian shape {expected_shape}, "
            f"got {tuple(jacobian.shape)}"
        )


def write_torchscript(
    config_path: Path,
    output_path: Path,
    temperature_name: str,
    salinity_name: str,
    thickness_name: str,
    output_name: str,
) -> None:
    config = _load_config(config_path)
    emulator = _build_emulator(
        config=config,
        temperature_name=temperature_name,
        salinity_name=salinity_name,
        thickness_name=thickness_name,
        output_name=output_name,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(emulator)
    scripted.save(str(output_path))
    _verify_saved_module(output_path)

    print(f"Saved: {output_path}")
    print(f"  config      : {config_path}")
    print(f"  input_names : {[temperature_name, salinity_name, thickness_name]}")
    print(f"  output_names: {[output_name]}")
    print("  inputs      : [nnodes, 3*nlevels] packed as [T, S, dz]")
    print("  jacobian    : compact [nnodes, nRequestedPairs]")


def main() -> None:
    default_config = REPO_ROOT / "emulators/heave_salinity/config.yaml"
    default_output = SCRIPT_DIR / "heave_salinity.ts"

    parser = argparse.ArgumentParser(
        description="Write the heave-salinity TorchScript balance file"
    )
    parser.add_argument(
        "--config",
        default=str(default_config),
        help="Heave-salinity YAML config",
    )
    parser.add_argument(
        "--output",
        default=str(default_output),
        help="Output .ts path",
    )
    parser.add_argument(
        "--T-name",
        default="sea_water_potential_temperature",
        help="Packed temperature input field name",
    )
    parser.add_argument(
        "--S-name",
        default="sea_water_salinity",
        help="Packed salinity input field name",
    )
    parser.add_argument(
        "--dz-name",
        default="sea_water_cell_thickness",
        help="Packed layer-thickness input field name",
    )
    parser.add_argument(
        "--output-name",
        default="sea_water_salinity",
        help="Balanced salinity output field name",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()

    write_torchscript(
        config_path=config_path,
        output_path=output_path,
        temperature_name=args.T_name,
        salinity_name=args.S_name,
        thickness_name=args.dz_name,
        output_name=args.output_name,
    )


if __name__ == "__main__":
    main()
