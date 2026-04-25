#!/usr/bin/env python3
"""Prepare training data from CF-1 NetCDF atmosphere and ocean/ice files.

Reads one atmosphere file and one ocean/ice file, applies quality control
and domain masking (sea-ice / ocean / both), normalises, and saves the
result as a compressed .npz file suitable for train_ml_balance.py.

Usage
-----
    python scripts/prepare_training_data.py \\
        --config   emulators/aice/config.yaml        \\
        --atm      /path/to/atm.nc          \\
        --ocn      /path/to/ice_ocean.nc    \\
        --output   data/training_aice.npz

Atmosphere file is optional; omit --atm if your config uses ocean-only
inputs (e.g. emulators/aice/both_domain.yaml without atmospheric variables).

The output .npz contains:
    inputs       : float32 [N, input_size]   — raw physical values
    targets      : float32 [N, output_size]  — raw physical values
    lons, lats   : float32 [N]               — spatial coordinates
    input_mean   : float32 [input_size]
    input_std    : float32 [input_size]
    output_mean  : float32 [output_size]
    output_std   : float32 [output_size]
    metadata     : object  — variable names, CF mappings, sizes

Note: inputs and targets are stored in *physical* space.  The training
script normalises them on load using the bundled statistics.
"""

import argparse
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from saber_pytorch.ml.data import UFSEmulatorDataBuilder  # noqa: E402
from saber_pytorch.ml.training import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare ML balance training data from CF-1 NetCDF files"
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--atm", default=None, help="Path to atmosphere NetCDF file (optional)"
    )
    parser.add_argument(
        "--ocn", required=True, help="Path to ocean/ice NetCDF file"
    )
    parser.add_argument(
        "--output", required=True, help="Output .npz file path"
    )
    parser.add_argument(
        "--max-patterns",
        type=int,
        default=None,
        help="Maximum number of patterns to extract (overrides config)",
    )
    parser.add_argument(
        "--thin-fraction",
        type=float,
        default=None,
        help="Fraction to keep, e.g. 0.1 (overrides config)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config.get("data", {})

    max_patterns: int = (
        args.max_patterns or data_cfg.get("max_patterns", 400000)
    )
    thin_fraction: float = (
        args.thin_fraction or data_cfg.get("thin_fraction", 1.0)
    )

    preparer = UFSEmulatorDataBuilder(config)
    result = preparer.prepare_training_data(
        atm_file=args.atm,
        ocn_file=args.ocn,
        max_patterns=max_patterns,
        output_file=args.output,
        thin_fraction=thin_fraction,
    )

    n = result["metadata"]["n_patterns"]
    in_sz = result["metadata"]["input_size"]
    out_sz = result["metadata"]["output_size"]
    print(f"\nSaved {n} patterns to {args.output}")
    print(f"  inputs : {n} × {in_sz}")
    print(f"  targets: {n} × {out_sz}")
    print(f"  input variables : {result['metadata']['input_features']}")
    print(f"  output variables: {result['metadata']['output_features']}")


if __name__ == "__main__":
    main()
