#!/usr/bin/env python3
"""Reconstruct one geom100 salinity profile with the heave balance.

Given a longitude/latitude, this script finds the nearest valid geom100 column
as truth, chooses a nearby valid column as background, and applies the localized
heave balance with:

    deltaT = T_truth - T_background
    S_heave = S_background + K_ST(background) deltaT

The comparison is by model level.  The CSV also writes truth and background
layer-center depths because neighboring MOM6 columns can have different layer
thicknesses.
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml
from netCDF4 import Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(REPO_ROOT / "emulators/heave_salinity/outputs/mplconfig"),
)

from saber_pytorch.physics.heave_salinity import WeaverTSBalance


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_path(path_text: str, config_path: Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    from_cwd = (Path.cwd() / path).resolve()
    if from_cwd.exists():
        return from_cwd
    return (config_path.parent / path).resolve()


def _read_ocean_fields(
    ocean_file: Path,
    temperature_variable: str,
    salinity_variable: str,
    thickness_variable: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with Dataset(ocean_file) as ds:
        temperature = np.asarray(ds.variables[temperature_variable][0], dtype=np.float32)
        salinity = np.asarray(ds.variables[salinity_variable][0], dtype=np.float32)
        thickness = np.asarray(ds.variables[thickness_variable][0], dtype=np.float32)
    return temperature, salinity, thickness


def _read_grid(
    grid_file: Path,
    longitude_variable: str,
    latitude_variable: str,
) -> Tuple[np.ndarray, np.ndarray]:
    with Dataset(grid_file) as ds:
        lon = np.asarray(ds.variables[longitude_variable][:], dtype=np.float64)
        lat = np.asarray(ds.variables[latitude_variable][:], dtype=np.float64)
    return lon, lat


def _valid_column_mask(
    temperature: np.ndarray,
    salinity: np.ndarray,
    thickness: np.ndarray,
    min_valid_thickness_m: float,
) -> np.ndarray:
    finite = np.isfinite(temperature).all(axis=0) & np.isfinite(salinity).all(axis=0)
    has_water = (thickness > min_valid_thickness_m).any(axis=0)
    return finite & has_water


def _longitude_delta(lon: np.ndarray, target_lon: float) -> np.ndarray:
    return ((lon - target_lon + 180.0) % 360.0) - 180.0


def _nearest_valid_column(
    lon: np.ndarray,
    lat: np.ndarray,
    valid: np.ndarray,
    target_lon: float,
    target_lat: float,
) -> Tuple[int, int]:
    dlon = _longitude_delta(lon, target_lon) * np.cos(np.deg2rad(target_lat))
    dlat = lat - target_lat
    distance2 = dlon * dlon + dlat * dlat
    distance2 = np.where(valid, distance2, np.inf)
    iy, ix = np.unravel_index(np.argmin(distance2), distance2.shape)
    if not np.isfinite(distance2[iy, ix]):
        raise RuntimeError("No valid ocean columns found")
    return int(iy), int(ix)


def _nearby_background_column(
    valid: np.ndarray,
    temperature: np.ndarray,
    truth_iy: int,
    truth_ix: int,
    offset_iy: int,
    offset_ix: int,
    search_radius: int,
    min_grid_distance: int,
    min_surface_temperature_difference: float,
) -> Tuple[int, int]:
    ny, nx = valid.shape
    if search_radius < min_grid_distance:
        print(
            "Warning: background_search_radius is smaller than "
            "background_min_grid_distance; expanding search radius from "
            f"{search_radius} to {min_grid_distance}."
        )
        search_radius = min_grid_distance

    truth_surface_temperature = float(temperature[0, truth_iy, truth_ix])
    candidate_iy = truth_iy + offset_iy
    candidate_ix = (truth_ix + offset_ix) % nx
    candidate_grid_distance = max(abs(offset_iy), abs(offset_ix))
    if (
        0 <= candidate_iy < ny
        and valid[candidate_iy, candidate_ix]
        and candidate_grid_distance >= min_grid_distance
        and abs(float(temperature[0, candidate_iy, candidate_ix]) - truth_surface_temperature)
        >= min_surface_temperature_difference
    ):
        return int(candidate_iy), int(candidate_ix)

    best: Tuple[int, int] | None = None
    best_distance2 = float("inf")
    fallback: Tuple[int, int] | None = None
    fallback_abs_surface_temperature_difference = -1.0
    fallback_distance2 = float("inf")
    for radius in range(max(1, min_grid_distance), search_radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy == 0 and dx == 0:
                    continue
                if max(abs(dy), abs(dx)) != radius:
                    continue
                iy = truth_iy + dy
                ix = (truth_ix + dx) % nx
                if 0 <= iy < ny and valid[iy, ix]:
                    surface_temperature_difference = abs(
                        float(temperature[0, iy, ix]) - truth_surface_temperature
                    )
                    distance2 = float(dy * dy + dx * dx)
                    if (
                        surface_temperature_difference
                        > fallback_abs_surface_temperature_difference
                        or (
                            surface_temperature_difference
                            == fallback_abs_surface_temperature_difference
                            and distance2 < fallback_distance2
                        )
                    ):
                        fallback = (int(iy), int(ix))
                        fallback_abs_surface_temperature_difference = (
                            surface_temperature_difference
                        )
                        fallback_distance2 = distance2
                    if surface_temperature_difference < min_surface_temperature_difference:
                        continue
                    if distance2 < best_distance2:
                        best = (int(iy), int(ix))
                        best_distance2 = distance2
        if best is not None:
            return best

    if fallback is not None:
        print(
            "Warning: no background column satisfied "
            f"surface temperature difference >= "
            f"{min_surface_temperature_difference:.3f} degC within radius "
            f"{search_radius}. Using best available difference "
            f"{fallback_abs_surface_temperature_difference:.3f} degC."
        )
        return fallback

    raise RuntimeError(
        f"No valid background column found within radius {search_radius} of "
        f"iy={truth_iy}, ix={truth_ix}"
    )


def _make_emulator(config: Dict[str, Any]) -> WeaverTSBalance:
    model = config["model"]
    return WeaverTSBalance(
        input_names=[
            "sea_water_potential_temperature",
            "sea_water_salinity",
            "ocean_layer_thickness",
        ],
        output_names=["sea_water_salinity"],
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


def _load_ts_emulator(checkpoint_path: Path) -> Any:
    return torch.jit.load(str(checkpoint_path), map_location="cpu")


def _center_depth(thickness: np.ndarray) -> np.ndarray:
    safe_thickness = np.where(np.isfinite(thickness), np.maximum(thickness, 0.0), 0.0)
    return np.cumsum(safe_thickness) - 0.5 * safe_thickness


def _source_valid_levels(
    temperature: np.ndarray,
    thickness: np.ndarray,
    min_layer_thickness: float,
    fill_value_threshold: float,
) -> np.ndarray:
    return (
        np.isfinite(temperature)
        & np.isfinite(thickness)
        & (thickness > min_layer_thickness)
        & (np.abs(temperature) < fill_value_threshold)
    )


def _interpolate_profile_to_depths(
    profile: np.ndarray,
    source_depths: np.ndarray,
    valid_levels: np.ndarray,
    target_depths: np.ndarray,
) -> np.ndarray:
    valid = valid_levels & np.isfinite(profile) & np.isfinite(source_depths)
    if int(valid.sum()) == 0:
        return np.full_like(target_depths, np.nan, dtype=np.float32)
    if int(valid.sum()) == 1:
        return np.full_like(target_depths, float(profile[valid][0]), dtype=np.float32)
    return np.interp(target_depths, source_depths[valid], profile[valid]).astype(np.float32)


def _apply_heave(
    emulator: WeaverTSBalance,
    truth_temperature: np.ndarray,
    background_temperature: np.ndarray,
    background_salinity: np.ndarray,
    background_thickness: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    T_bg = torch.from_numpy(background_temperature.reshape(1, -1).astype(np.float32))
    S_bg = torch.from_numpy(background_salinity.reshape(1, -1).astype(np.float32))
    h_bg = torch.from_numpy(background_thickness.reshape(1, -1).astype(np.float32))
    dT = torch.from_numpy(
        (truth_temperature - background_temperature).reshape(1, -1).astype(np.float32)
    )
    mask = torch.ones(1, 1, dtype=T_bg.dtype)
    delta_s = emulator.apply_from_state(
        {
            emulator.input_names[0]: T_bg,
            emulator.input_names[1]: S_bg,
            emulator.input_names[2]: h_bg,
        },
        dT,
        mask,
    )
    delta_s_np = delta_s[0].detach().cpu().numpy()
    return background_salinity + delta_s_np, delta_s_np, dT[0].detach().cpu().numpy()


def _apply_ml_jacobian(
    ts_model: Any,
    truth_temperature: np.ndarray,
    truth_salinity: np.ndarray,
    truth_thickness: np.ndarray,
    background_temperature: np.ndarray,
    background_salinity: np.ndarray,
    background_thickness: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_n = int(ts_model.source_num_levels)
    target_n = int(ts_model.target_num_levels)
    min_thickness = float(ts_model.min_layer_thickness)
    fill_threshold = float(ts_model.fill_value_threshold)

    T_bg = background_temperature[:source_n].astype(np.float32)
    h_bg = background_thickness[:source_n].astype(np.float32)
    inputs = torch.from_numpy(np.concatenate([T_bg, h_bg])).unsqueeze(0)  # [1, 2*source_n]
    mask = torch.ones(1, 1)

    # Reduced-grid depths for plotting (derived from reduced layer thicknesses)
    with torch.no_grad():
        h_reduced = ts_model.reduced_thickness_inputs(inputs)[0].numpy()
    safe_h = np.where(h_reduced > 0.0, h_reduced, 0.0).astype(np.float32)
    target_depths = np.cumsum(safe_h) - 0.5 * safe_h

    # Interpolate truth and background salinity onto the reduced grid
    background_depths = _center_depth(background_thickness)
    background_valid = _source_valid_levels(
        background_temperature, background_thickness, min_thickness, fill_threshold
    )
    truth_depths = _center_depth(truth_thickness)
    truth_valid = _source_valid_levels(
        truth_temperature, truth_thickness, min_thickness, fill_threshold
    )
    background_s_reduced = _interpolate_profile_to_depths(
        background_salinity, background_depths, background_valid, target_depths
    )
    truth_s_reduced = _interpolate_profile_to_depths(
        truth_salinity, truth_depths, truth_valid, target_depths
    )

    # Full T-Jacobian [target_n, source_n] via a single jac_physical call
    row_indices = torch.arange(target_n, dtype=torch.long).repeat_interleave(source_n)
    col_indices = torch.arange(source_n, dtype=torch.long).repeat(target_n)
    jac_flat = ts_model.jac_physical(inputs, mask, row_indices, col_indices)
    jac_T = jac_flat.view(target_n, source_n).detach().numpy()

    dT_source = (truth_temperature[:source_n] - T_bg)
    delta_s = jac_T @ dT_source
    return (
        background_s_reduced + delta_s,
        truth_s_reduced,
        background_s_reduced,
        delta_s,
        target_depths,
    )


def _write_profile_csv(
    output_path: Path,
    truth_depth: np.ndarray,
    background_depth: np.ndarray,
    truth_temperature: np.ndarray,
    background_temperature: np.ndarray,
    delta_temperature: np.ndarray,
    truth_salinity: np.ndarray,
    background_salinity: np.ndarray,
    heave_salinity: np.ndarray,
    delta_salinity: np.ndarray,
    valid_level: np.ndarray,
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "level",
                "truth_depth_m",
                "background_depth_m",
                "truth_temperature",
                "background_temperature",
                "delta_temperature",
                "truth_salinity",
                "background_salinity",
                "heave_salinity",
                "delta_salinity_heave",
                "background_error",
                "heave_error",
                "valid_level",
            ]
        )
        for level in range(truth_salinity.size):
            writer.writerow(
                [
                    level,
                    float(truth_depth[level]),
                    float(background_depth[level]),
                    float(truth_temperature[level]),
                    float(background_temperature[level]),
                    float(delta_temperature[level]),
                    float(truth_salinity[level]),
                    float(background_salinity[level]),
                    float(heave_salinity[level]),
                    float(delta_salinity[level]),
                    float(background_salinity[level] - truth_salinity[level]),
                    float(heave_salinity[level] - truth_salinity[level]),
                    bool(valid_level[level]),
                ]
            )


def _write_ml_profile_csv(
    output_path: Path,
    ml_depth: np.ndarray,
    ml_truth_salinity: np.ndarray,
    ml_background_salinity: np.ndarray,
    ml_salinity: np.ndarray,
    ml_delta_salinity: np.ndarray,
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "ml_level",
                "ml_depth_m",
                "truth_salinity_on_ml_grid",
                "background_salinity_on_ml_grid",
                "ml_jacobian_salinity",
                "ml_delta_salinity",
                "background_error",
                "ml_error",
            ]
        )
        for level in range(ml_depth.size):
            writer.writerow(
                [
                    level,
                    float(ml_depth[level]),
                    float(ml_truth_salinity[level]),
                    float(ml_background_salinity[level]),
                    float(ml_salinity[level]),
                    float(ml_delta_salinity[level]),
                    float(ml_background_salinity[level] - ml_truth_salinity[level]),
                    float(ml_salinity[level] - ml_truth_salinity[level]),
                ]
            )


def _plot_profiles(
    output_path: Path,
    truth_depth: np.ndarray,
    background_depth: np.ndarray,
    truth_temperature: np.ndarray,
    background_temperature: np.ndarray,
    truth_salinity: np.ndarray,
    background_salinity: np.ndarray,
    heave_salinity: np.ndarray,
    valid_level: np.ndarray,
    ml_depth: np.ndarray | None,
    ml_truth_salinity: np.ndarray | None,
    ml_salinity: np.ndarray | None,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 7), constrained_layout=True)
    axes[0].plot(truth_temperature, truth_depth, label="truth T", color="black")
    axes[0].plot(
        background_temperature,
        background_depth,
        label="background T",
        color="tab:blue",
        linestyle="--",
    )
    axes[0].invert_yaxis()
    axes[0].set_xlabel("temperature [degC]")
    axes[0].set_ylabel("layer-center depth [m]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(truth_salinity, truth_depth, label="truth S", color="black")
    axes[1].plot(
        background_salinity,
        background_depth,
        label="background S",
        color="tab:blue",
        linestyle="--",
    )
    axes[1].plot(
        heave_salinity,
        background_depth,
        label="heave S",
        color="tab:red",
    )
    if ml_depth is not None and ml_salinity is not None:
        axes[1].plot(
            ml_salinity,
            ml_depth,
            label="ML Jacobian S",
            color="tab:green",
            marker=".",
            markersize=3,
        )
    axes[1].invert_yaxis()
    axes[1].set_xlabel("salinity [PSU]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    background_error = np.where(
        valid_level,
        background_salinity - truth_salinity,
        np.nan,
    )
    heave_error = np.where(valid_level, heave_salinity - truth_salinity, np.nan)
    background_rmse = float(np.sqrt(np.nanmean(background_error * background_error)))
    heave_rmse = float(np.sqrt(np.nanmean(heave_error * heave_error)))
    background_bias = float(np.nanmean(background_error))
    heave_bias = float(np.nanmean(heave_error))
    if ml_depth is not None and ml_truth_salinity is not None and ml_salinity is not None:
        ml_error = ml_salinity - ml_truth_salinity
        ml_rmse = float(np.sqrt(np.nanmean(ml_error * ml_error)))
        ml_bias = float(np.nanmean(ml_error))
    else:
        ml_error = None
        ml_rmse = float("nan")
        ml_bias = float("nan")
    axes[2].axvline(0.0, color="0.4", linewidth=1)
    axes[2].plot(
        background_error,
        truth_depth,
        label=(
            "background - truth "
            f"(RMSE={background_rmse:.4f}, bias={background_bias:+.4f})"
        ),
        color="tab:blue",
    )
    axes[2].plot(
        heave_error,
        truth_depth,
        label=f"heave - truth (RMSE={heave_rmse:.4f}, bias={heave_bias:+.4f})",
        color="tab:red",
    )
    if ml_error is not None and ml_depth is not None:
        axes[2].plot(
            ml_error,
            ml_depth,
            label=f"ML - truth (RMSE={ml_rmse:.4f}, bias={ml_bias:+.4f})",
            color="tab:green",
            marker=".",
            markersize=3,
        )
    axes[2].invert_yaxis()
    axes[2].set_xlabel("salinity error [PSU]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.suptitle(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="emulators/heave_salinity/config.yaml")
    parser.add_argument("--lon", type=float, default=None, help="Target longitude")
    parser.add_argument("--lat", type=float, default=None, help="Target latitude")
    parser.add_argument("--background-offset-iy", type=int, default=None)
    parser.add_argument("--background-offset-ix", type=int, default=None)
    parser.add_argument("--background-search-radius", type=int, default=None)
    parser.add_argument("--background-min-grid-distance", type=int, default=None)
    parser.add_argument("--background-min-surface-temp-diff", type=float, default=None)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = _load_config(config_path)
    input_cfg = config["input"]
    model_cfg = config["model"]
    ml_cfg = config.get("ml_salinity", {})
    diag_cfg = config["diagnostics"]
    output_dir = _resolve_path(config["output"]["directory"], config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_lon = float(
        args.lon
        if args.lon is not None
        else diag_cfg.get("target_lon", diag_cfg.get("reconstruction_lon", -119.5))
    )
    target_lat = float(
        args.lat
        if args.lat is not None
        else diag_cfg.get("target_lat", diag_cfg.get("reconstruction_lat", 4.36))
    )
    offset_iy = int(
        args.background_offset_iy
        if args.background_offset_iy is not None
        else diag_cfg.get("background_offset_iy", 0)
    )
    offset_ix = int(
        args.background_offset_ix
        if args.background_offset_ix is not None
        else diag_cfg.get("background_offset_ix", 1)
    )
    search_radius = int(
        args.background_search_radius
        if args.background_search_radius is not None
        else diag_cfg.get("background_search_radius", 5)
    )
    min_grid_distance = int(
        args.background_min_grid_distance
        if args.background_min_grid_distance is not None
        else diag_cfg.get("background_min_grid_distance", 1)
    )
    min_surface_temp_diff = float(
        args.background_min_surface_temp_diff
        if args.background_min_surface_temp_diff is not None
        else diag_cfg.get("background_min_surface_temperature_difference", 0.0)
    )

    ocean_file = _resolve_path(input_cfg["ocean_file"], config_path)
    grid_file = _resolve_path(input_cfg["grid_file"], config_path)
    temperature, salinity, thickness = _read_ocean_fields(
        ocean_file,
        input_cfg.get("temperature_variable", "Temp"),
        input_cfg.get("salinity_variable", "Salt"),
        input_cfg.get("thickness_variable", "h"),
    )
    lon, lat = _read_grid(
        grid_file,
        input_cfg.get("longitude_variable", "lon"),
        input_cfg.get("latitude_variable", "lat"),
    )
    valid_columns = _valid_column_mask(
        temperature,
        salinity,
        thickness,
        float(model_cfg.get("min_valid_thickness_m", 0.1)),
    )

    truth_iy, truth_ix = _nearest_valid_column(
        lon,
        lat,
        valid_columns,
        target_lon,
        target_lat,
    )
    bg_iy, bg_ix = _nearby_background_column(
        valid_columns,
        temperature,
        truth_iy,
        truth_ix,
        offset_iy,
        offset_ix,
        search_radius,
        min_grid_distance,
        min_surface_temp_diff,
    )

    truth_temperature = temperature[:, truth_iy, truth_ix]
    truth_salinity = salinity[:, truth_iy, truth_ix]
    truth_thickness = thickness[:, truth_iy, truth_ix]
    bg_temperature = temperature[:, bg_iy, bg_ix]
    bg_salinity = salinity[:, bg_iy, bg_ix]
    bg_thickness = thickness[:, bg_iy, bg_ix]
    surface_temperature_difference = float(bg_temperature[0] - truth_temperature[0])

    emulator = _make_emulator(config)
    heave_salinity, delta_salinity, delta_temperature = _apply_heave(
        emulator,
        truth_temperature,
        bg_temperature,
        bg_salinity,
        bg_thickness,
    )
    ml_salinity = None
    ml_truth_salinity = None
    ml_background_salinity = None
    ml_delta_salinity = None
    ml_depth = None
    ml_rmse = None
    ml_mae = None
    ml_bias = None
    if bool(ml_cfg.get("enabled", False)):
        ml_checkpoint = _resolve_path(str(ml_cfg["checkpoint"]), config_path)
        ml_emulator = _load_ts_emulator(ml_checkpoint)
        (
            ml_salinity,
            ml_truth_salinity,
            ml_background_salinity,
            ml_delta_salinity,
            ml_depth,
        ) = _apply_ml_jacobian(
            ml_emulator,
            truth_temperature,
            truth_salinity,
            truth_thickness,
            bg_temperature,
            bg_salinity,
            bg_thickness,
        )
        ml_error = ml_salinity - ml_truth_salinity
        ml_rmse = float(np.sqrt(np.nanmean(ml_error * ml_error)))
        ml_mae = float(np.nanmean(np.abs(ml_error)))
        ml_bias = float(np.nanmean(ml_error))

    min_thickness = float(model_cfg.get("min_valid_thickness_m", 0.1))
    valid_level = (
        np.isfinite(truth_salinity)
        & np.isfinite(bg_salinity)
        & (truth_thickness > min_thickness)
        & (bg_thickness > min_thickness)
    )
    background_rmse = float(
        np.sqrt(np.nanmean((bg_salinity[valid_level] - truth_salinity[valid_level]) ** 2))
    )
    heave_rmse = float(
        np.sqrt(
            np.nanmean((heave_salinity[valid_level] - truth_salinity[valid_level]) ** 2)
        )
    )
    background_mae = float(
        np.nanmean(np.abs(bg_salinity[valid_level] - truth_salinity[valid_level]))
    )
    heave_mae = float(
        np.nanmean(np.abs(heave_salinity[valid_level] - truth_salinity[valid_level]))
    )

    truth_depth = _center_depth(truth_thickness)
    bg_depth = _center_depth(bg_thickness)
    stem = f"geom100_heave_reconstruction_lon{target_lon:.2f}_lat{target_lat:.2f}"
    safe_stem = stem.replace("-", "m").replace(".", "p")
    csv_path = output_dir / f"{safe_stem}.csv"
    png_path = output_dir / f"{safe_stem}.png"
    npz_path = output_dir / f"{safe_stem}.npz"

    _write_profile_csv(
        csv_path,
        truth_depth,
        bg_depth,
        truth_temperature,
        bg_temperature,
        delta_temperature,
        truth_salinity,
        bg_salinity,
        heave_salinity,
        delta_salinity,
        valid_level,
    )
    if (
        ml_depth is not None
        and ml_truth_salinity is not None
        and ml_background_salinity is not None
        and ml_salinity is not None
        and ml_delta_salinity is not None
    ):
        _write_ml_profile_csv(
            output_dir / f"{safe_stem}_ml_reduced.csv",
            ml_depth,
            ml_truth_salinity,
            ml_background_salinity,
            ml_salinity,
            ml_delta_salinity,
        )
    title = (
        f"truth ({lon[truth_iy, truth_ix]:.2f}, {lat[truth_iy, truth_ix]:.2f}) "
        f"vs background ({lon[bg_iy, bg_ix]:.2f}, {lat[bg_iy, bg_ix]:.2f}); "
        f"surface dT={surface_temperature_difference:+.2f} C"
    )
    _plot_profiles(
        png_path,
        truth_depth,
        bg_depth,
        truth_temperature,
        bg_temperature,
        truth_salinity,
        bg_salinity,
        heave_salinity,
        valid_level,
        ml_depth,
        ml_truth_salinity,
        ml_salinity,
        title,
    )
    np.savez_compressed(
        npz_path,
        target_lon=target_lon,
        target_lat=target_lat,
        truth_iy=truth_iy,
        truth_ix=truth_ix,
        background_iy=bg_iy,
        background_ix=bg_ix,
        truth_lon=lon[truth_iy, truth_ix],
        truth_lat=lat[truth_iy, truth_ix],
        background_lon=lon[bg_iy, bg_ix],
        background_lat=lat[bg_iy, bg_ix],
        surface_temperature_difference=surface_temperature_difference,
        truth_depth=truth_depth,
        background_depth=bg_depth,
        truth_temperature=truth_temperature,
        background_temperature=bg_temperature,
        delta_temperature=delta_temperature,
        truth_salinity=truth_salinity,
        background_salinity=bg_salinity,
        heave_salinity=heave_salinity,
        delta_salinity=delta_salinity,
        valid_level=valid_level,
        background_rmse=background_rmse,
        heave_rmse=heave_rmse,
        background_mae=background_mae,
        heave_mae=heave_mae,
        ml_depth=ml_depth,
        ml_truth_salinity=ml_truth_salinity,
        ml_background_salinity=ml_background_salinity,
        ml_salinity=ml_salinity,
        ml_delta_salinity=ml_delta_salinity,
        ml_rmse=ml_rmse,
        ml_mae=ml_mae,
        ml_bias=ml_bias,
    )

    print(f"Requested lon/lat     : {target_lon:.3f}, {target_lat:.3f}")
    print(
        f"Truth column          : iy={truth_iy}, ix={truth_ix}, "
        f"lon={lon[truth_iy, truth_ix]:.3f}, lat={lat[truth_iy, truth_ix]:.3f}"
    )
    print(
        f"Background column     : iy={bg_iy}, ix={bg_ix}, "
        f"lon={lon[bg_iy, bg_ix]:.3f}, lat={lat[bg_iy, bg_ix]:.3f}"
    )
    print(f"Surface T difference  : {surface_temperature_difference:+.3f} degC")
    print(f"Valid compared levels : {int(valid_level.sum())} / {valid_level.size}")
    print(f"Background RMSE/MAE   : {background_rmse:.6f} / {background_mae:.6f} PSU")
    print(f"Heave RMSE/MAE        : {heave_rmse:.6f} / {heave_mae:.6f} PSU")
    if ml_rmse is not None and ml_mae is not None and ml_bias is not None:
        print(
            f"ML Jacobian RMSE/MAE/bias: "
            f"{ml_rmse:.6f} / {ml_mae:.6f} / {ml_bias:+.6f} PSU"
        )
    print(f"Wrote                 : {png_path}")
    print(f"Wrote                 : {csv_path}")
    print(f"Wrote                 : {npz_path}")


if __name__ == "__main__":
    main()
