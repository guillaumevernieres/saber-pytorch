#!/usr/bin/env python3
"""Prepare ML salinity-profile training data from WOA23 climatology.

Reads annual-mean potential temperature (t_an) and salinity (s_an) from a
WOA23 NetCDF file, interpolates each valid ocean column onto the model depth
grid, and writes a .npz training dataset in the format expected by
train_ml_balance.py.

Usage
-----
    python scripts/prepare_woa_training_data.py \\
        --woa-file   /path/to/woa23_B5C2_st00_01.nc \\
        --model-depth-file  /path/to/MOM.res.nc \\
        --config     tests/salinity/config.local.yaml \\
        --output     tests/salinity/data/salt_profile_training_woa.npz

The output .npz has the same structure as salt_profile_training.npz and can
be used directly with train_ml_balance.py by updating the data.data_path in
the config.

WOA depth levels below the ocean floor are handled by setting the
corresponding model layer thickness to zero; the existing QC in
UFSEmulatorDataBuilder.filter_data() then skips those levels automatically.

WOA t_an is treated as potential temperature (the in-situ vs. potential
temperature difference is negligible for the upper ocean and acceptable for
the deep ocean for training purposes).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from saber_pytorch.ml.data import UFSEmulatorDataBuilder  # noqa: E402
from saber_pytorch.ml.training import load_config  # noqa: E402
from saber_pytorch.observations.argo_profiles import (  # noqa: E402
    read_model_depth_grid,
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_woa(
    woa_file: Path,
    t_var: str = "t_an",
    s_var: str = "s_an",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (depth, lat, lon, T, S) from a WOA NetCDF file.

    T and S have shape [ndepth, nlat, nlon] with NaN for masked/land cells.
    Time index 0 is used (annual climatology file has a single time slice).
    """
    with nc.Dataset(woa_file) as ds:
        depth = np.asarray(ds.variables["depth"][:], dtype=np.float64)
        lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
        t = np.ma.filled(
            ds.variables[t_var][0].astype(np.float64), fill_value=np.nan
        )
        s = np.ma.filled(
            ds.variables[s_var][0].astype(np.float64), fill_value=np.nan
        )
    return depth, lat, lon, t, s


# ---------------------------------------------------------------------------
# Depth-grid helpers
# ---------------------------------------------------------------------------

def _model_thicknesses(model_depth: np.ndarray) -> np.ndarray:
    """Compute layer thicknesses from model layer-centre depths.

    Uses midpoint differences with a zero surface interface.
    """
    n = len(model_depth)
    interfaces = np.empty(n + 1, dtype=np.float64)
    interfaces[0] = max(
        model_depth[0] - 0.5 * (model_depth[1] - model_depth[0]), 0.0
    )
    interfaces[1:n] = 0.5 * (model_depth[:-1] + model_depth[1:])
    interfaces[n] = model_depth[-1] + 0.5 * (model_depth[-1] - model_depth[-2])
    return np.diff(interfaces)


# ---------------------------------------------------------------------------
# WOA → model-depth interpolation
# ---------------------------------------------------------------------------

def build_woa_profiles(
    woa_depth: np.ndarray,
    woa_lat: np.ndarray,
    woa_lon: np.ndarray,
    t_woa: np.ndarray,
    s_woa: np.ndarray,
    model_depth: np.ndarray,
    model_dz: np.ndarray,
    min_valid_woa_levels: int = 5,
) -> dict[str, np.ndarray]:
    """Interpolate WOA T/S columns to the model depth grid.

    For each valid WOA ocean column:
    - T and S are linearly interpolated to model layer centres that lie within
      the WOA valid depth range.
    - Model levels below the deepest valid WOA depth have their layer
      thickness set to zero so that QC in filter_data() skips them.

    Returns a data dict compatible with UFSEmulatorDataBuilder.filter_data().
    Keys: lat, lon, mask, sea_water_potential_temperature,
          sea_water_cell_thickness, sea_water_salinity.
    """
    ndepth_woa = len(woa_depth)
    nmodel = len(model_depth)
    nlat, nlon = len(woa_lat), len(woa_lon)
    n_total = nlat * nlon

    # Flatten WOA grid to [n_total, ndepth_woa]
    t_flat = t_woa.reshape(ndepth_woa, n_total).T
    s_flat = s_woa.reshape(ndepth_woa, n_total).T

    lat_grid, lon_grid = np.meshgrid(woa_lat, woa_lon, indexing="ij")
    lat_flat = lat_grid.ravel().astype(np.float32)
    lon_flat = lon_grid.ravel().astype(np.float32)

    fixed_dz = model_dz.astype(np.float32)

    T_out = np.zeros((n_total, nmodel), dtype=np.float32)
    S_out = np.zeros((n_total, nmodel), dtype=np.float32)
    dz_out = np.zeros((n_total, nmodel), dtype=np.float32)
    mask_out = np.zeros(n_total, dtype=np.float32)

    # Pre-filter: only iterate over columns with enough valid WOA levels.
    valid_count = (np.isfinite(t_flat) & np.isfinite(s_flat)).sum(axis=1)
    candidates = np.where(valid_count >= min_valid_woa_levels)[0]
    print(f"  Candidate ocean columns: {len(candidates)} / {n_total}")

    n_accepted = 0
    for i in candidates:
        t_col = t_flat[i]
        s_col = s_flat[i]

        valid = np.isfinite(t_col) & np.isfinite(s_col)
        valid_depth = woa_depth[valid]
        valid_t = t_col[valid]
        valid_s = s_col[valid]

        # Skip if the shallowest model level is already below the WOA range.
        if model_depth[0] > valid_depth[-1]:
            continue

        in_range = model_depth <= valid_depth[-1]

        # Interpolate within the WOA valid depth range.
        T_out[i, in_range] = np.interp(
            model_depth[in_range], valid_depth, valid_t
        ).astype(np.float32)
        S_out[i, in_range] = np.interp(
            model_depth[in_range], valid_depth, valid_s
        ).astype(np.float32)

        # Levels below WOA bathymetry: fill value (dz stays 0, so QC skips).
        if not np.all(in_range):
            T_out[i, ~in_range] = float(valid_t[-1])
            S_out[i, ~in_range] = float(valid_s[-1])

        dz_out[i, in_range] = fixed_dz[in_range]
        mask_out[i] = 1.0
        n_accepted += 1

    print(f"  Accepted ocean columns:  {n_accepted} / {n_total}")
    return {
        "lat": lat_flat,
        "lon": lon_flat,
        "mask": mask_out,
        "sea_water_potential_temperature": T_out,
        "sea_water_cell_thickness": dz_out,
        "sea_water_salinity": S_out,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--woa-file",
        type=Path,
        default=Path("/home/gvernier/data/woa/woa23_B5C2_st00_01.nc"),
        help="WOA23 NetCDF file with t_an and s_an",
    )
    parser.add_argument(
        "--model-depth-file",
        type=Path,
        default=_REPO.parent / "i-jedi/test-soca/geom100/MOM.res.nc",
        help="Model NetCDF file containing the vertical depth coordinate",
    )
    parser.add_argument(
        "--model-depth-variable",
        default=None,
        help="Name of the depth variable (auto-detected if omitted)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_REPO / "tests/salinity/config.local.yaml",
        help="Training config YAML (emulator_type, target_num_levels, etc.)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO / "tests/salinity/data/salt_profile_training_woa.npz",
        help="Output .npz path",
    )
    parser.add_argument(
        "--t-variable",
        default="t_an",
        help="WOA temperature variable name",
    )
    parser.add_argument(
        "--s-variable",
        default="s_an",
        help="WOA salinity variable name",
    )
    parser.add_argument(
        "--min-valid-woa-levels",
        type=int,
        default=5,
        help="Minimum valid WOA depth levels required to include a column",
    )
    parser.add_argument(
        "--max-patterns",
        type=int,
        default=None,
        help="Cap on training patterns (default: all valid)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print(f"Reading WOA climatology: {args.woa_file}")
    woa_depth, woa_lat, woa_lon, t_woa, s_woa = _read_woa(
        args.woa_file, args.t_variable, args.s_variable
    )
    print(
        f"  {len(woa_depth)} depth levels, "
        f"{len(woa_lat)} lats × {len(woa_lon)} lons"
    )
    print(f"  depth range: {woa_depth[0]:.0f}–{woa_depth[-1]:.0f} m")

    print(f"\nReading model depth grid: {args.model_depth_file}")
    model_depth = read_model_depth_grid(
        args.model_depth_file, args.model_depth_variable
    )
    model_dz = _model_thicknesses(model_depth)
    print(
        f"  {len(model_depth)} levels, "
        f"{model_depth[0]:.1f}–{model_depth[-1]:.1f} m"
    )

    print("\nInterpolating WOA columns to model depth grid ...")
    data = build_woa_profiles(
        woa_depth,
        woa_lat,
        woa_lon,
        t_woa,
        s_woa,
        model_depth,
        model_dz,
        min_valid_woa_levels=args.min_valid_woa_levels,
    )

    print("\nLoading config and building training dataset ...")
    config = load_config(str(args.config))
    max_patterns = args.max_patterns or 500_000
    builder = UFSEmulatorDataBuilder(config)

    result = builder.filter_data(data, max_patterns)
    if len(result) == 4:
        patterns, targets, lons, lats = result
    else:
        patterns, targets, lons, lats = result[:4]

    print("\nComputing normalization statistics ...")
    input_mean, input_std = builder.compute_normalization_stats(patterns)
    output_mean, output_std = builder.compute_normalization_stats(targets)

    vcfg = config.get("variables", {})
    npz_data = {
        "inputs": patterns,
        "targets": targets,
        "lons": lons,
        "lats": lats,
        "input_mean": input_mean,
        "input_std": input_std,
        "output_mean": output_mean,
        "output_std": output_std,
        "metadata": {
            "n_patterns": len(patterns),
            "input_features": str(builder.input_variables),
            "output_features": str(builder.output_variables),
            "input_size": str(patterns.shape[1]),
            "output_size": str(targets.shape[1]),
            "emulator_type": "salinity_profile",
            "target_num_levels": str(builder.target_num_levels),
            "reduced_grid": str({
                "method": builder.reduced_grid_method,
                "gradient_weight": builder.reduced_grid_gradient_weight,
            }),
            "source": "WOA23",
            "woa_file": str(args.woa_file),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **npz_data)

    n = len(patterns)
    print(f"\nSaved {n} patterns → {args.output}")
    print(f"  inputs:  {n} × {patterns.shape[1]}")
    print(f"  targets: {n} × {targets.shape[1]}")


if __name__ == "__main__":
    main()
