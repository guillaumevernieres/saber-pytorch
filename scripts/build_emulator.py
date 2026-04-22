#!/usr/bin/env python3
"""Build and save a TorchScript steric height emulator.

dz (layer thicknesses) is passed as a regular Atlas input field by SABER at
runtime, alongside T and S.  Mid-level depths are derived inside the emulator
from dz, so no separate depth field is needed.

Usage
-----
    python build_emulator.py \\
        --output   steric_height_emulator.ts \\
        [--T-name  sea_water_potential_temperature] \\
        [--S-name  sea_water_salinity] \\
        [--dz-name ocean_layer_thickness] \\
        [--ssh-name sea_surface_height_above_geoid] \\
        [--rho0    1025.0]

Variable name notes
-------------------
--dz-name must match the Atlas field name in your SABER configuration.
The CF standard name for this field varies across models; verify against
your JEDI/SABER YAML before building.

The saved .ts file is a TorchScript module loadable by SABER C++ via
  torch::jit::load(path)
and compatible with setupVerticalEmulator.
"""

import argparse
import sys
from pathlib import Path
from typing import List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from saber_pytorch.physics.steric_height import StericHeightEmulator


def build_and_save(
    output_path: str,
    T_name: str = "sea_water_potential_temperature",
    S_name: str = "sea_water_salinity",
    dz_name: str = "ocean_layer_thickness",
    ssh_name: str = "sea_surface_height_above_geoid",
    rho0: float = 1025.0,
) -> None:
    input_names: List[str] = [T_name, S_name, dz_name]
    output_names: List[str] = [ssh_name]

    emulator = StericHeightEmulator(
        input_names=input_names,
        output_names=output_names,
        rho0=rho0,
    )
    emulator.eval()

    scripted = torch.jit.script(emulator)
    scripted.save(output_path)

    print(f"Saved: {output_path}")
    print(f"  input_names : {input_names}")
    print(f"  output_names: {output_names}")
    print(f"  inputs tensor at runtime: [nnodes, 3*nlevels]")
    print(f"    [:, 0*n:1*n] = {T_name}")
    print(f"    [:, 1*n:2*n] = {S_name}")
    print(f"    [:, 2*n:3*n] = {dz_name}")
    print(f"  jac shape   : [nnodes, 1, 3*nlevels]")
    print(f"  Note: Jacobian columns for {dz_name} are zero (geometry).")
    print(f"  Note: depth_m is derived from {dz_name} as cumsum(dz) - dz/2.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build steric height TorchScript emulator"
    )
    parser.add_argument("--output", required=True, help="Output .ts path")
    parser.add_argument("--T-name", default="sea_water_potential_temperature")
    parser.add_argument("--S-name", default="sea_water_salinity")
    parser.add_argument("--dz-name", default="ocean_layer_thickness")
    parser.add_argument("--ssh-name", default="sea_surface_height_above_geoid")
    parser.add_argument("--rho0", type=float, default=1025.0)
    args = parser.parse_args()

    build_and_save(
        output_path=args.output,
        T_name=args.T_name,
        S_name=args.S_name,
        dz_name=args.dz_name,
        ssh_name=args.ssh_name,
        rho0=args.rho0,
    )


if __name__ == "__main__":
    main()
