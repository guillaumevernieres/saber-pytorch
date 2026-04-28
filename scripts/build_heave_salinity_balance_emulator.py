#!/usr/bin/env python3
"""Build and save a TorchScript localized heave salinity balance emulator.

The saved module follows the vertical TorchBalance contract used by the steric
height physical balance.  Runtime inputs are packed as all levels of T, S, and
dz.  The returned Jacobian maps temperature increments to balanced salinity
increments; salinity and dz column blocks are zero.
"""

import argparse
import sys
from pathlib import Path
from typing import List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from saber_pytorch.physics.heave_salinity import LocalizedHeaveSalinityEmulator


def build_and_save(
    output_path: str,
    T_name: str = "sea_water_potential_temperature",
    S_name: str = "sea_water_salinity",
    dz_name: str = "ocean_layer_thickness",
    output_name: str = "sea_water_salinity",
    epsilon: float = 1.0e-12,
    epsilon_taper: float = 1.0e-10,
    localization_length_scale_levels: float = 3.0,
    use_temperature_gradient_taper: bool = True,
    output_taper: str = "temperature_gradient",
) -> None:
    input_names: List[str] = [T_name, S_name, dz_name]
    output_names: List[str] = [output_name]

    emulator = LocalizedHeaveSalinityEmulator(
        input_names=input_names,
        output_names=output_names,
        epsilon=epsilon,
        epsilon_taper=epsilon_taper,
        localization_length_scale_levels=localization_length_scale_levels,
        use_temperature_gradient_taper=use_temperature_gradient_taper,
        output_taper=output_taper,
    )
    emulator.eval()

    scripted = torch.jit.script(emulator)
    scripted.save(output_path)

    loaded = torch.jit.load(output_path)
    nlevels = 4
    test_inputs = torch.randn(2, 3 * nlevels)
    test_mask = torch.ones(2, 1)
    test_rows = torch.arange(nlevels, dtype=torch.long)
    test_cols = torch.arange(nlevels, dtype=torch.long)
    test_jac = loaded.jac_physical(test_inputs, test_mask, test_rows, test_cols)
    expected_shape = (2, nlevels)
    if tuple(test_jac.shape) != expected_shape:
        raise RuntimeError(
            f"Verification failed: expected jac shape {expected_shape}, "
            f"got {tuple(test_jac.shape)}"
        )

    print(f"Saved: {output_path}")
    print(f"  input_names : {input_names}")
    print(f"  output_names: {output_names}")
    print("  inputs tensor at runtime: [nnodes, 3*nlevels]")
    print(f"    [:, 0*n:1*n] = {T_name}")
    print(f"    [:, 1*n:2*n] = {S_name}")
    print(f"    [:, 2*n:3*n] = {dz_name}")
    print("  jac shape   : [nnodes, nRequestedPairs]")
    print("  nonzero cols: temperature block only")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build localized heave salinity TorchScript emulator"
    )
    parser.add_argument("--output", required=True, help="Output .ts path")
    parser.add_argument("--T-name", default="sea_water_potential_temperature")
    parser.add_argument("--S-name", default="sea_water_salinity")
    parser.add_argument("--dz-name", default="ocean_layer_thickness")
    parser.add_argument("--output-name", default="sea_water_salinity")
    parser.add_argument("--epsilon", type=float, default=1.0e-12)
    parser.add_argument("--epsilon-taper", type=float, default=1.0e-10)
    parser.add_argument("--localization-length-scale-levels", type=float, default=3.0)
    parser.add_argument(
        "--no-temperature-gradient-taper",
        action="store_true",
        help="Disable the temperature-gradient taper",
    )
    parser.add_argument(
        "--output-taper",
        choices=("temperature_gradient", "none"),
        default="temperature_gradient",
    )
    args = parser.parse_args()

    build_and_save(
        output_path=args.output,
        T_name=args.T_name,
        S_name=args.S_name,
        dz_name=args.dz_name,
        output_name=args.output_name,
        epsilon=args.epsilon,
        epsilon_taper=args.epsilon_taper,
        localization_length_scale_levels=args.localization_length_scale_levels,
        use_temperature_gradient_taper=not args.no_temperature_gradient_taper,
        output_taper=args.output_taper,
    )


if __name__ == "__main__":
    main()
