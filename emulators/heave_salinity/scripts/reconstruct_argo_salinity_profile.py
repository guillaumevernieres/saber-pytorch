#!/usr/bin/env python3
"""Reconstruct Argo salinity profiles with the heave balance.

For every retained Argo profile whose near-surface temperature is inside a
configurable range, pick a nearby background profile and apply the localized
heave balance:

    deltaT = T_truth - T_background
    S_heave = S_background + K_ST(background) * deltaT

One three-panel plot (temperature, salinity, salinity error) is saved per
qualifying observation.

Example usage:
    python emulators/heave_salinity/scripts/reconstruct_argo_salinity_profile.py \\
        --config emulators/heave_salinity/config.yaml \\
        --surface-temp-min 20 \\
        --surface-temp-max 30 \\
        --max-profiles 50
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(REPO_ROOT / "emulators/heave_salinity/outputs/mplconfig"),
)

from saber_pytorch.ml.ml_balance import FFNNSalinityProfileEmulator
from saber_pytorch.physics.heave_salinity import WeaverTSBalance


# ---------------------------------------------------------------------------
# Config and I/O helpers
# ---------------------------------------------------------------------------

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


def _load_argo_dataset(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. "
            "Build it with emulators/heave_salinity/scripts/build_real_argo_ts_profiles.py"
        )
    with np.load(path, allow_pickle=True) as data:
        required = [
            "model_depth", "potential_temperature", "salinity",
            "valid_mask", "latitude", "longitude",
        ]
        missing = [k for k in required if k not in data.files]
        if missing:
            raise KeyError(f"{path} is missing arrays: {missing}")
        return {name: data[name] for name in data.files}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _longitude_delta(lon: np.ndarray, target_lon: float) -> np.ndarray:
    return ((lon - target_lon + 180.0) % 360.0) - 180.0


def _centers_to_thickness(depth: np.ndarray) -> np.ndarray:
    """Estimate layer thicknesses from cell-centre depths (strictly increasing)."""
    depth = np.asarray(depth, dtype=np.float64)
    if depth.size == 1:
        return np.asarray([max(2.0 * depth[0], 0.1)], dtype=np.float32)
    interfaces = np.empty(depth.size + 1, dtype=np.float64)
    interfaces[0] = max(0.0, depth[0] - 0.5 * (depth[1] - depth[0]))
    interfaces[1:-1] = 0.5 * (depth[:-1] + depth[1:])
    interfaces[-1] = depth[-1] + 0.5 * (depth[-1] - depth[-2])
    thickness = np.diff(interfaces)
    if np.any(thickness <= 0.0):
        raise ValueError("depth must be strictly increasing")
    return thickness.astype(np.float32)


def _near_surface_temperature(
    temperature: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    max_depth_m: float,
) -> float:
    """Mean temperature of valid levels at or above max_depth_m."""
    near = valid & (depth <= max_depth_m) & np.isfinite(temperature)
    if near.sum() > 0:
        return float(np.mean(temperature[near]))
    # fall back to shallowest valid finite level
    idx = np.flatnonzero(valid & np.isfinite(temperature))
    return float(temperature[idx[0]]) if idx.size > 0 else float("nan")


def _first_common_surface_dtemp(
    truth_temperature: np.ndarray,
    background_temperature: np.ndarray,
    common_valid: np.ndarray,
) -> float:
    idx = np.flatnonzero(common_valid)
    if idx.size == 0:
        return float("nan")
    level = int(idx[0])
    return float(background_temperature[level] - truth_temperature[level])


# ---------------------------------------------------------------------------
# Background profile selection
# ---------------------------------------------------------------------------

def _choose_background_profile(
    truth_index: int,
    longitude: np.ndarray,
    latitude: np.ndarray,
    temperature: np.ndarray,
    valid_mask: np.ndarray,
    valid_profile: np.ndarray,
    *,
    search_radius_deg: float,
    min_distance_deg: float,
    min_surface_temperature_difference: float,
    min_common_valid_levels: int,
) -> int:
    """Return index of a suitable background profile, or -1 if none found."""
    target_lon = float(longitude[truth_index])
    target_lat = float(latitude[truth_index])
    dlon = _longitude_delta(longitude, target_lon) * np.cos(np.deg2rad(target_lat))
    dlat = latitude - target_lat
    distance = np.sqrt(dlon * dlon + dlat * dlat)

    truth_valid = valid_mask[truth_index]
    truth_temp = temperature[truth_index]
    best_index = -1
    best_distance = float("inf")
    fallback_index = -1
    fallback_abs_dtemp = -1.0
    fallback_distance = float("inf")

    for index in range(longitude.size):
        if index == truth_index or not bool(valid_profile[index]):
            continue
        d = float(distance[index])
        if d < min_distance_deg or d > search_radius_deg:
            continue
        common = truth_valid & valid_mask[index]
        if int(common.sum()) < min_common_valid_levels:
            continue
        surface_dtemp = abs(_first_common_surface_dtemp(truth_temp, temperature[index], common))
        if (
            surface_dtemp > fallback_abs_dtemp
            or (surface_dtemp == fallback_abs_dtemp and d < fallback_distance)
        ):
            fallback_index = index
            fallback_abs_dtemp = surface_dtemp
            fallback_distance = d
        if surface_dtemp < min_surface_temperature_difference:
            continue
        if d < best_distance:
            best_index = index
            best_distance = d

    if best_index >= 0:
        return best_index
    return fallback_index  # may be -1 if no candidate found at all


# ---------------------------------------------------------------------------
# Heave balance helpers
# ---------------------------------------------------------------------------

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


def _apply_heave(
    emulator: WeaverTSBalance,
    truth_temperature: np.ndarray,
    background_temperature: np.ndarray,
    background_salinity: np.ndarray,
    background_thickness: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (heave_salinity, delta_temperature) arrays."""
    T_bg = torch.from_numpy(
        background_temperature.reshape(1, -1).astype(np.float32)
    )
    S_bg = torch.from_numpy(
        background_salinity.reshape(1, -1).astype(np.float32)
    )
    h_bg = torch.from_numpy(
        background_thickness.reshape(1, -1).astype(np.float32)
    )
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
    heave_sal = background_salinity + delta_s[0].detach().cpu().numpy()
    delta_temp = dT[0].detach().cpu().numpy()
    return heave_sal, delta_temp


def _load_ml_salinity_emulator(path: Path) -> torch.nn.Module:
    """Load a salinity-profile training checkpoint for fast reduced-grid diagnostics."""
    if path.is_dir():
        model_dir_checkpoint = path / "models_salt_profile" / "best_model.pt"
        if model_dir_checkpoint.exists():
            path = model_dir_checkpoint
        else:
            path = path / "best_model.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model_cfg = config["model"]
    var_cfg = config["variables"]
    if str(model_cfg.get("emulator_type", "")).lower() != "salinity_profile":
        raise ValueError(
            "ml_salinity.checkpoint must be a salinity_profile training checkpoint. "
            f"Got emulator_type={model_cfg.get('emulator_type')!r}"
        )

    norm_path = path.parent / "normalization.pt"
    if not norm_path.exists():
        raise FileNotFoundError(f"normalization.pt not found next to {path}")
    moments = torch.load(norm_path, map_location="cpu", weights_only=False)

    input_names = list(var_cfg["input_variables"])
    output_names = list(var_cfg["output_variables"])
    if len(input_names) != 2 or len(output_names) != 1:
        raise ValueError(
            "salinity_profile checkpoint must have two inputs and one output; "
            f"got inputs={input_names}, outputs={output_names}"
        )

    source_num_levels = int(var_cfg["num_levels"])
    target_num_levels = int(var_cfg.get("target_num_levels", moments["output_mean"].shape[0]))
    reduced_grid_cfg = var_cfg.get(
        "reduced_grid",
        config.get("reduced_grid", config.get("data", {}).get("reduced_grid", {})),
    )
    reduced_grid_method = str(reduced_grid_cfg.get("method", "uniform_depth"))
    default_gradient_weight = (
        2.0
        if reduced_grid_method.lower() in ("temperature_gradient", "temp_gradient")
        else 0.0
    )
    temperature_gradient_weight = float(
        reduced_grid_cfg.get("gradient_weight", default_gradient_weight)
    )

    emulator = FFNNSalinityProfileEmulator(
        temperature_variable_name=str(input_names[0]),
        thickness_variable_name=str(input_names[1]),
        output_variable_name=str(output_names[0]),
        source_num_levels=source_num_levels,
        target_num_levels=target_num_levels,
        hidden_size=int(model_cfg["hidden_size"]),
        hidden_layers=int(model_cfg.get("hidden_layers", 2)),
        activation=str(model_cfg.get("activation", "gelu")),
        use_conv1d=bool(model_cfg.get("use_conv1d", False)),
        conv_channels=int(model_cfg.get("conv_channels", 32)),
        conv_kernel_size=int(model_cfg.get("conv_kernel_size", 3)),
        reduced_grid_method=reduced_grid_method,
        temperature_gradient_weight=temperature_gradient_weight,
    )

    mapped: Dict[str, torch.Tensor] = {}
    for key, value in checkpoint["model_state_dict"].items():
        if key.startswith("network."):
            mapped["ffnn.network." + key[len("network."):]] = value.float()
        elif key.startswith("conv1d."):
            mapped["ffnn.conv1d." + key[len("conv1d."):]] = value.float()

    missing, unexpected = emulator.load_state_dict(mapped, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in ML salinity checkpoint: {unexpected}")
    non_norm_missing = [
        key for key in missing if not any(key.endswith(s) for s in ("_mean", "_std"))
    ]
    if non_norm_missing:
        raise RuntimeError(f"Missing keys in ML salinity checkpoint: {non_norm_missing}")

    emulator.init_norm(
        moments["input_mean"].float(),
        moments["input_std"].float(),
        moments["output_mean"].float(),
        moments["output_std"].float(),
    )
    return emulator.eval()


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


def _uniform_target_depth(
    target_level: int,
    target_num_levels: int,
    first_depth: float,
    last_depth: float,
) -> float:
    if target_num_levels == 1:
        return first_depth
    fraction = float(target_level) / float(target_num_levels - 1)
    return first_depth + fraction * (last_depth - first_depth)


def _temperature_gradient_target_depth(
    target_level: int,
    target_num_levels: int,
    first: int,
    last: int,
    temperature: np.ndarray,
    depths: np.ndarray,
    valid: np.ndarray,
    gradient_weight: float,
) -> float:
    first_depth = float(depths[first])
    last_depth = float(depths[last])
    if target_num_levels == 1:
        return first_depth

    fraction = float(target_level) / float(target_num_levels - 1)
    max_slope = 0.0
    prev = -1
    for level in range(first, temperature.size):
        if valid[level]:
            if prev >= 0:
                dz = float(depths[level] - depths[prev])
                if dz > 1.0e-12:
                    slope = abs(float(temperature[level] - temperature[prev]) / dz)
                    max_slope = max(max_slope, slope)
            prev = level

    if max_slope <= 1.0e-12:
        return _uniform_target_depth(target_level, target_num_levels, first_depth, last_depth)

    metric_total = 0.0
    prev = -1
    for level in range(first, temperature.size):
        if valid[level]:
            if prev >= 0:
                dz = float(depths[level] - depths[prev])
                if dz > 1.0e-12:
                    slope = abs(float(temperature[level] - temperature[prev]) / dz)
                    metric_total += dz * (1.0 + gradient_weight * slope / max_slope)
            prev = level

    if metric_total <= 1.0e-12:
        return _uniform_target_depth(target_level, target_num_levels, first_depth, last_depth)

    target_metric = metric_total * fraction
    metric = 0.0
    prev = -1
    for level in range(first, temperature.size):
        if valid[level]:
            if prev >= 0:
                dz = float(depths[level] - depths[prev])
                if dz > 1.0e-12:
                    slope = abs(float(temperature[level] - temperature[prev]) / dz)
                    step = dz * (1.0 + gradient_weight * slope / max_slope)
                    next_metric = metric + step
                    if next_metric >= target_metric:
                        weight = (target_metric - metric) / step
                        return float(depths[prev] * (1.0 - weight) + depths[level] * weight)
                    metric = next_metric
            prev = level

    return last_depth


def _ml_target_depths(
    emulator: torch.nn.Module,
    temperature: np.ndarray,
    thickness: np.ndarray,
) -> np.ndarray:
    source_num_levels = int(emulator.source_num_levels)
    target_num_levels = int(emulator.target_num_levels)
    min_layer_thickness = float(emulator.min_layer_thickness)
    fill_value_threshold = float(emulator.fill_value_threshold)
    use_temperature_gradient_grid = bool(emulator.use_temperature_gradient_grid)
    temperature_gradient_weight = float(emulator.temperature_gradient_weight)

    temperature = np.asarray(temperature[:source_num_levels], dtype=np.float64)
    thickness = np.asarray(thickness[:source_num_levels], dtype=np.float64)
    depths = _center_depth(thickness)
    valid = _source_valid_levels(
        temperature,
        thickness,
        min_layer_thickness,
        fill_value_threshold,
    )

    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        raise RuntimeError("ML salinity emulator found no valid source levels")
    first = int(valid_indices[0])
    last = int(valid_indices[-1])
    first_depth = float(depths[first])
    last_depth = float(depths[last])

    target_depths = np.zeros(target_num_levels, dtype=np.float32)
    for target_level in range(target_num_levels):
        if use_temperature_gradient_grid and temperature_gradient_weight > 0.0:
            target_depth = _temperature_gradient_target_depth(
                target_level,
                target_num_levels,
                first,
                last,
                temperature,
                depths,
                valid,
                temperature_gradient_weight,
            )
        else:
            target_depth = _uniform_target_depth(
                target_level,
                target_num_levels,
                first_depth,
                last_depth,
            )
        target_depths[target_level] = target_depth
    return target_depths


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


def _reduced_temperature_jacobian(
    emulator: torch.nn.Module,
    reduced_inputs: torch.Tensor,
) -> torch.Tensor:
    target_num_levels = int(emulator.target_num_levels)
    if hasattr(emulator.ffnn, "_jac_physical"):
        return emulator.ffnn._jac_physical(reduced_inputs)[0, :, :target_num_levels]

    x = reduced_inputs.detach().requires_grad_(True)
    y = emulator.ffnn.predict(x)
    rows = []
    for output_level in range(target_num_levels):
        grad_outputs = torch.zeros_like(y)
        grad_outputs[:, output_level] = 1.0
        grad = torch.autograd.grad(
            y,
            x,
            grad_outputs=grad_outputs,
            retain_graph=True,
        )[0]
        rows.append(grad[0, :target_num_levels])
    return torch.stack(rows, dim=0)


def _apply_ml_jacobian(
    emulator: torch.nn.Module,
    truth_temperature: np.ndarray,
    truth_salinity: np.ndarray,
    truth_thickness: np.ndarray,
    background_temperature: np.ndarray,
    background_salinity: np.ndarray,
    background_thickness: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_num_levels = int(emulator.source_num_levels)
    target_num_levels = int(emulator.target_num_levels)
    min_layer_thickness = float(emulator.min_layer_thickness)
    fill_value_threshold = float(emulator.fill_value_threshold)

    T_bg = np.asarray(background_temperature[:source_num_levels], dtype=np.float32)
    h_bg = np.asarray(background_thickness[:source_num_levels], dtype=np.float32)
    inputs = torch.from_numpy(np.concatenate([T_bg, h_bg]).reshape(1, -1))

    target_depths = _ml_target_depths(emulator, T_bg, h_bg)
    background_depths = _center_depth(background_thickness[:source_num_levels])
    background_valid = _source_valid_levels(
        background_temperature[:source_num_levels],
        background_thickness[:source_num_levels],
        min_layer_thickness,
        fill_value_threshold,
    )
    truth_depths = _center_depth(truth_thickness[:source_num_levels])
    truth_valid = _source_valid_levels(
        truth_temperature[:source_num_levels],
        truth_thickness[:source_num_levels],
        min_layer_thickness,
        fill_value_threshold,
    )

    background_t_reduced = (
        emulator.reduced_temperature_inputs(inputs)[0].detach().cpu().numpy()
    )
    background_h_reduced = emulator.reduced_thickness_inputs(inputs)[0].detach()
    truth_t_reduced = _interpolate_profile_to_depths(
        truth_temperature[:source_num_levels],
        truth_depths,
        truth_valid,
        target_depths,
    )
    background_s_reduced = _interpolate_profile_to_depths(
        background_salinity[:source_num_levels],
        background_depths,
        background_valid,
        target_depths,
    )
    truth_s_reduced = _interpolate_profile_to_depths(
        truth_salinity[:source_num_levels],
        truth_depths,
        truth_valid,
        target_depths,
    )

    reduced_inputs = torch.cat(
        [
            torch.from_numpy(background_t_reduced).view(1, -1),
            background_h_reduced.view(1, -1),
        ],
        dim=1,
    )
    reduced_jac = _reduced_temperature_jacobian(emulator, reduced_inputs)
    dT_reduced = torch.from_numpy(
        (truth_t_reduced - background_t_reduced).astype(np.float32)
    )
    delta_s = torch.matmul(reduced_jac, dT_reduced).detach().cpu().numpy()
    if delta_s.shape[0] != target_num_levels:
        raise RuntimeError(
            f"Expected {target_num_levels} ML salinity increments, got {delta_s.shape[0]}"
        )
    return (
        background_s_reduced + delta_s,
        truth_s_reduced,
        background_s_reduced,
        delta_s,
        target_depths,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_profile(
    output_path: Path,
    depth: np.ndarray,
    truth_temperature: np.ndarray,
    background_temperature: np.ndarray,
    truth_salinity: np.ndarray,
    background_salinity: np.ndarray,
    heave_salinity: np.ndarray,
    title: str,
    ml_depth: np.ndarray | None = None,
    ml_truth_salinity: np.ndarray | None = None,
    ml_salinity: np.ndarray | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = (
        np.isfinite(truth_salinity)
        & np.isfinite(background_salinity)
        & np.isfinite(heave_salinity)
    )
    background_error = np.where(valid, background_salinity - truth_salinity, np.nan)
    heave_error = np.where(valid, heave_salinity - truth_salinity, np.nan)
    stats_valid = valid.copy()
    ml_rmse = float("nan")
    ml_bias = float("nan")
    if ml_depth is not None and ml_truth_salinity is not None and ml_salinity is not None:
        ml_valid = np.isfinite(ml_truth_salinity) & np.isfinite(ml_salinity)
        ml_native_error = np.where(ml_valid, ml_salinity - ml_truth_salinity, np.nan)
        ml_error = _interpolate_error_to_model_depth(
            ml_depth,
            ml_native_error,
            depth,
        )
        stats_valid = stats_valid & np.isfinite(ml_error)
    else:
        ml_error = None

    if int(stats_valid.sum()) > 0:
        bg_rmse = float(np.sqrt(np.nanmean(background_error[stats_valid] ** 2)))
        hv_rmse = float(np.sqrt(np.nanmean(heave_error[stats_valid] ** 2)))
        bg_bias = float(np.nanmean(background_error[stats_valid]))
        hv_bias = float(np.nanmean(heave_error[stats_valid]))
        if ml_error is not None:
            ml_rmse = float(np.sqrt(np.nanmean(ml_error[stats_valid] ** 2)))
            ml_bias = float(np.nanmean(ml_error[stats_valid]))
    else:
        bg_rmse = float("nan")
        hv_rmse = float("nan")
        bg_bias = float("nan")
        hv_bias = float("nan")

    fig, axes = plt.subplots(1, 3, figsize=(14, 7), constrained_layout=True)

    axes[0].plot(truth_temperature, depth, color="black", label="truth T")
    axes[0].plot(
        background_temperature, depth,
        color="tab:blue", linestyle="--", label="background T",
    )
    axes[0].invert_yaxis()
    axes[0].set_xlabel("potential temperature [°C]")
    axes[0].set_ylabel("depth [m]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(truth_salinity, depth, color="black", label="truth S")
    axes[1].plot(
        background_salinity, depth,
        color="tab:blue", linestyle="--", label="background S",
    )
    axes[1].plot(heave_salinity, depth, color="tab:red", label="heave S")
    if ml_depth is not None and ml_salinity is not None:
        axes[1].plot(
            ml_salinity,
            ml_depth,
            color="tab:green",
            marker=".",
            markersize=3,
            label="ML Jacobian S",
        )
    axes[1].invert_yaxis()
    axes[1].set_xlabel("salinity [PSU]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    axes[2].axvline(0.0, color="0.4", linewidth=1)
    axes[2].plot(
        background_error, depth, color="tab:blue",
        label=f"bg − truth  RMSE={bg_rmse:.4f}  bias={bg_bias:+.4f}",
    )
    axes[2].plot(
        heave_error, depth, color="tab:red",
        label=f"heave − truth  RMSE={hv_rmse:.4f}  bias={hv_bias:+.4f}",
    )
    if ml_error is not None and ml_depth is not None:
        axes[2].plot(
            ml_error,
            depth,
            color="tab:green",
            marker=".",
            markersize=3,
            label=f"ML − truth  RMSE={ml_rmse:.4f}  bias={ml_bias:+.4f}",
        )
    axes[2].invert_yaxis()
    axes[2].set_xlabel("salinity error [PSU]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)

    fig.suptitle(title, fontsize=9)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _depth_stats_from_sums(
    count: np.ndarray,
    error_sum: np.ndarray,
    error_sumsq: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    bias = np.full(count.shape, np.nan, dtype=np.float64)
    rmse = np.full(count.shape, np.nan, dtype=np.float64)
    valid = count > 0
    bias[valid] = error_sum[valid] / count[valid]
    rmse[valid] = np.sqrt(error_sumsq[valid] / count[valid])
    return rmse, bias


def _write_depth_stats_csv(
    output_path: Path,
    depth: np.ndarray,
    count: np.ndarray,
    background_rmse: np.ndarray,
    background_bias: np.ndarray,
    heave_rmse: np.ndarray,
    heave_bias: np.ndarray,
    ml_count: np.ndarray | None = None,
    ml_rmse: np.ndarray | None = None,
    ml_bias: np.ndarray | None = None,
    level_indices: np.ndarray | None = None,
) -> None:
    import csv

    if level_indices is None:
        level_indices = np.arange(depth.size)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "level",
                "depth_m",
                "n_profiles",
                "background_rmse",
                "background_bias",
                "heave_rmse",
                "heave_bias",
                "ml_n_profiles",
                "ml_rmse",
                "ml_bias",
            ]
        )
        for level in range(depth.size):
            writer.writerow(
                [
                    int(level_indices[level]),
                    float(depth[level]),
                    int(count[level]),
                    float(background_rmse[level]),
                    float(background_bias[level]),
                    float(heave_rmse[level]),
                    float(heave_bias[level]),
                    int(ml_count[level]) if ml_count is not None else 0,
                    float(ml_rmse[level]) if ml_rmse is not None else float("nan"),
                    float(ml_bias[level]) if ml_bias is not None else float("nan"),
                ]
            )


def _plot_depth_stats(
    output_path: Path,
    depth: np.ndarray,
    count: np.ndarray,
    background_rmse: np.ndarray,
    background_bias: np.ndarray,
    heave_rmse: np.ndarray,
    heave_bias: np.ndarray,
    ml_count: np.ndarray | None = None,
    ml_rmse: np.ndarray | None = None,
    ml_bias: np.ndarray | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = count > 0
    fig, axes = plt.subplots(1, 3, figsize=(13, 7), sharey=True, constrained_layout=True)

    axes[0].plot(background_rmse[valid], depth[valid], color="tab:blue", label="background")
    axes[0].plot(heave_rmse[valid], depth[valid], color="tab:red", label="heave")
    if ml_count is not None and ml_rmse is not None:
        ml_valid = ml_count > 0
        axes[0].plot(ml_rmse[ml_valid], depth[ml_valid], color="tab:green", label="ML")
    axes[0].set_xlabel("salinity RMSE [PSU]")
    axes[0].set_ylabel("depth [m]")
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].axvline(0.0, color="0.4", linewidth=1)
    axes[1].plot(background_bias[valid], depth[valid], color="tab:blue", label="background")
    axes[1].plot(heave_bias[valid], depth[valid], color="tab:red", label="heave")
    if ml_count is not None and ml_bias is not None:
        ml_valid = ml_count > 0
        axes[1].plot(ml_bias[ml_valid], depth[ml_valid], color="tab:green", label="ML")
    axes[1].set_xlabel("salinity bias [PSU]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    axes[2].plot(count[valid], depth[valid], color="0.25", label="background/heave")
    if ml_count is not None:
        ml_valid = ml_count > 0
        axes[2].plot(ml_count[ml_valid], depth[ml_valid], color="tab:green", label="ML")
    axes[2].set_xlabel("profiles contributing")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)

    fig.suptitle("Selected Argo profile salinity error statistics by depth")
    fig.savefig(output_path, dpi=110)
    plt.close(fig)


def _interpolate_error_to_model_depth(
    error_depth: np.ndarray,
    error: np.ndarray,
    model_depth: np.ndarray,
) -> np.ndarray:
    valid = np.isfinite(error_depth) & np.isfinite(error)
    output = np.full(model_depth.shape, np.nan, dtype=np.float64)
    if int(valid.sum()) < 2:
        return output
    depth = np.asarray(error_depth[valid], dtype=np.float64)
    values = np.asarray(error[valid], dtype=np.float64)
    order = np.argsort(depth)
    depth = depth[order]
    values = values[order]
    unique_depth, unique_inverse = np.unique(depth, return_inverse=True)
    if unique_depth.size < 2:
        return output
    unique_values = np.zeros(unique_depth.shape, dtype=np.float64)
    for index in range(unique_depth.size):
        unique_values[index] = np.mean(values[unique_inverse == index])
    inside = (model_depth >= unique_depth[0]) & (model_depth <= unique_depth[-1])
    output[inside] = np.interp(model_depth[inside], unique_depth, unique_values)
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="emulators/heave_salinity/config.yaml",
    )
    parser.add_argument("--profiles-file", type=Path, default=None)
    parser.add_argument(
        "--surface-temp-min", type=float, default=None,
        help="Minimum near-surface temperature to qualify (degC)",
    )
    parser.add_argument(
        "--surface-temp-max", type=float, default=None,
        help="Maximum near-surface temperature to qualify (degC)",
    )
    parser.add_argument(
        "--near-surface-depth-m", type=float, default=None,
        help="Depth above which to average temperature for the surface check (m)",
    )
    parser.add_argument(
        "--background-search-radius-deg", type=float, default=None,
    )
    parser.add_argument("--background-min-distance-deg", type=float, default=None)
    parser.add_argument(
        "--background-min-surface-temp-diff", type=float, default=None,
    )
    parser.add_argument("--min-common-valid-levels", type=int, default=None)
    parser.add_argument(
        "--max-profiles", type=int, default=None,
        help="Stop after this many qualifying profiles (useful for testing)",
    )
    parser.add_argument(
        "--output-subdir", default="argo_heave_profiles",
        help="Sub-directory under the config output directory for the plots",
    )
    parser.add_argument(
        "--no-profile-plots", action="store_true",
        help="Only write aggregate depth statistics; skip per-profile PNGs.",
    )
    parser.add_argument(
        "--max-stats-depth-m", type=float, default=None,
        help="Deepest model depth included in aggregate depth statistics.",
    )
    parser.add_argument(
        "--max-background-temperature-rmse",
        type=float,
        default=None,
        help=(
            "Skip profile/background pairs whose background temperature RMSE "
            "against Argo temperature exceeds this value (degC). Defaults to "
            "argo.max_background_temperature_rmse or 1.0."
        ),
    )
    parser.add_argument(
        "--max-background-salinity-rmse",
        type=float,
        default=None,
        help=(
            "Skip profile/background pairs whose background salinity RMSE "
            "against Argo salinity exceeds this value (PSU). Defaults to "
            "argo.max_background_salinity_rmse or 1.0."
        ),
    )
    parser.add_argument(
        "--salinity-min", type=float, default=None,
        help="Minimum physical salinity retained from the Argo profile file.",
    )
    parser.add_argument(
        "--salinity-max", type=float, default=None,
        help="Maximum physical salinity retained from the Argo profile file.",
    )
    parser.add_argument(
        "--no-ml", action="store_true",
        help="Disable the ML salinity emulator even if enabled in the config.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = _load_config(config_path)
    argo_cfg = config.get("argo", {})
    ml_cfg = config.get("ml_salinity", {})

    base_output_dir = _resolve_path(config["output"]["directory"], config_path)
    output_dir = base_output_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    profiles_file = (
        args.profiles_file
        if args.profiles_file is not None
        else _resolve_path(
            str(
                argo_cfg.get(
                    "profiles_file",
                    "emulators/heave_salinity/outputs/argo_ts_profiles.npz",
                )
            ),
            config_path,
        )
    )

    dataset = _load_argo_dataset(profiles_file)
    model_depth = np.asarray(dataset["model_depth"], dtype=np.float64)
    temperature = np.asarray(dataset["potential_temperature"], dtype=np.float32).copy()
    salinity = np.asarray(dataset["salinity"], dtype=np.float32).copy()
    valid_mask = np.asarray(dataset["valid_mask"], dtype=bool)
    longitude = np.asarray(dataset["longitude"], dtype=np.float64)
    latitude = np.asarray(dataset["latitude"], dtype=np.float64)

    salinity_min = float(
        args.salinity_min
        if args.salinity_min is not None
        else argo_cfg.get("salinity_min", 20.0)
    )
    salinity_max = float(
        args.salinity_max
        if args.salinity_max is not None
        else argo_cfg.get("salinity_max", 45.0)
    )
    original_valid_mask = valid_mask.copy()
    physical_salinity = (
        np.isfinite(salinity)
        & (salinity >= salinity_min)
        & (salinity <= salinity_max)
    )
    finite_temperature = np.isfinite(temperature)
    valid_mask = valid_mask & physical_salinity & finite_temperature
    dropped_salinity = original_valid_mask & (~physical_salinity)
    temperature[~valid_mask] = np.nan
    salinity[~valid_mask] = np.nan

    min_common = int(
        args.min_common_valid_levels
        if args.min_common_valid_levels is not None
        else argo_cfg.get("min_common_valid_levels", 10)
    )
    valid_profile = valid_mask.sum(axis=1) >= min_common

    near_surface_depth_m = float(
        args.near_surface_depth_m
        if args.near_surface_depth_m is not None
        else argo_cfg.get("near_surface_depth_m", 30.0)
    )
    surface_temp_min = float(
        args.surface_temp_min
        if args.surface_temp_min is not None
        else argo_cfg.get(
            "surface_temperature_min",
            argo_cfg.get("surface_temperature_threshold", 20.0),
        )
    )
    surface_temp_max = float(
        args.surface_temp_max
        if args.surface_temp_max is not None
        else argo_cfg.get("surface_temperature_max", float("inf"))
    )
    if surface_temp_min > surface_temp_max:
        raise ValueError(
            "surface temperature range is invalid: "
            f"min {surface_temp_min} > max {surface_temp_max}"
        )
    search_radius_deg = float(
        args.background_search_radius_deg
        if args.background_search_radius_deg is not None
        else argo_cfg.get("background_search_radius_deg", 3.0)
    )
    min_distance_deg = float(
        args.background_min_distance_deg
        if args.background_min_distance_deg is not None
        else argo_cfg.get("background_min_distance_deg", 1.0)
    )
    min_surf_dtemp = float(
        args.background_min_surface_temp_diff
        if args.background_min_surface_temp_diff is not None
        else argo_cfg.get("background_min_surface_temperature_difference", 1.0)
    )
    max_stats_depth_m = float(
        args.max_stats_depth_m
        if args.max_stats_depth_m is not None
        else argo_cfg.get("max_stats_depth_m", 2000.0)
    )
    max_background_temperature_rmse = float(
        args.max_background_temperature_rmse
        if args.max_background_temperature_rmse is not None
        else argo_cfg.get("max_background_temperature_rmse", 1.0)
    )
    if max_background_temperature_rmse < 0.0:
        raise ValueError(
            "max background temperature RMSE must be non-negative, got "
            f"{max_background_temperature_rmse}"
        )
    max_background_salinity_rmse = float(
        args.max_background_salinity_rmse
        if args.max_background_salinity_rmse is not None
        else argo_cfg.get("max_background_salinity_rmse", 1.0)
    )
    if max_background_salinity_rmse < 0.0:
        raise ValueError(
            "max background salinity RMSE must be non-negative, got "
            f"{max_background_salinity_rmse}"
    )
    stats_level_indices = np.flatnonzero(model_depth <= max_stats_depth_m)
    if stats_level_indices.size == 0:
        raise ValueError(
            f"No model depths are shallower than max_stats_depth_m={max_stats_depth_m}"
        )

    emulator = _make_emulator(config)
    ml_emulator = None
    if bool(ml_cfg.get("enabled", False)) and not args.no_ml:
        ml_checkpoint = _resolve_path(str(ml_cfg["checkpoint"]), config_path)
        ml_emulator = _load_ml_salinity_emulator(ml_checkpoint)
    full_thickness = _centers_to_thickness(model_depth)

    print(f"Profiles file            : {profiles_file}")
    print(f"Total retained profiles  : {int(valid_profile.sum())}")
    print(f"Near-surface depth       : {near_surface_depth_m:.0f} m")
    if np.isfinite(surface_temp_max):
        print(
            "Surface T range         : "
            f"[{surface_temp_min:.1f}, {surface_temp_max:.1f}] °C"
        )
    else:
        print(f"Surface T range         : >= {surface_temp_min:.1f} °C")
    print(f"Background search radius : {search_radius_deg:.1f} deg")
    print(f"Stats max depth          : {max_stats_depth_m:.0f} m")
    print(
        "Background temperature QC: "
        f"RMSE <= {max_background_temperature_rmse:.2f} degC"
    )
    print(
        "Background salinity QC  : "
        f"RMSE <= {max_background_salinity_rmse:.2f} PSU"
    )
    print(f"Salinity valid range     : [{salinity_min:.1f}, {salinity_max:.1f}] PSU")
    print(
        f"Salinity points dropped  : {int(dropped_salinity.sum())} "
        f"from {int(original_valid_mask.sum())} valid points"
    )
    print(f"Output directory         : {output_dir}")
    print()

    n_qualifying = 0
    n_processed = 0
    n_no_background = 0
    n_temperature_qc_rejected = 0
    n_salinity_qc_rejected = 0
    all_bg_rmse: list[float] = []
    all_hv_rmse: list[float] = []
    depth_count = np.zeros(model_depth.shape, dtype=np.int64)
    depth_bg_error_sum = np.zeros(model_depth.shape, dtype=np.float64)
    depth_bg_error_sumsq = np.zeros(model_depth.shape, dtype=np.float64)
    depth_hv_error_sum = np.zeros(model_depth.shape, dtype=np.float64)
    depth_hv_error_sumsq = np.zeros(model_depth.shape, dtype=np.float64)
    all_ml_rmse: list[float] = []
    depth_ml_count = np.zeros(model_depth.shape, dtype=np.int64)
    depth_ml_error_sum = np.zeros(model_depth.shape, dtype=np.float64)
    depth_ml_error_sumsq = np.zeros(model_depth.shape, dtype=np.float64)

    for truth_index in range(longitude.size):
        if not bool(valid_profile[truth_index]):
            continue

        surf_temp = _near_surface_temperature(
            temperature[truth_index], model_depth,
            valid_mask[truth_index], near_surface_depth_m,
        )
        if (
            not np.isfinite(surf_temp)
            or surf_temp < surface_temp_min
            or surf_temp > surface_temp_max
        ):
            continue

        n_qualifying += 1
        if args.max_profiles is not None and n_qualifying > args.max_profiles:
            break

        background_index = _choose_background_profile(
            truth_index,
            longitude,
            latitude,
            temperature,
            valid_mask,
            valid_profile,
            search_radius_deg=search_radius_deg,
            min_distance_deg=min_distance_deg,
            min_surface_temperature_difference=min_surf_dtemp,
            min_common_valid_levels=min_common,
        )
        if background_index < 0:
            n_no_background += 1
            continue

        common_valid = (
            valid_mask[truth_index]
            & valid_mask[background_index]
            & np.isfinite(temperature[truth_index])
            & np.isfinite(temperature[background_index])
            & np.isfinite(salinity[truth_index])
            & np.isfinite(salinity[background_index])
        )
        if int(common_valid.sum()) < min_common:
            n_no_background += 1
            continue

        selected = np.flatnonzero(common_valid)
        sel_depth = model_depth[selected]
        truth_temp = temperature[truth_index, selected]
        truth_sal = salinity[truth_index, selected]
        bg_temp = temperature[background_index, selected]
        bg_sal = salinity[background_index, selected]
        thickness = _centers_to_thickness(sel_depth)

        temp_err = bg_temp - truth_temp
        temp_rmse = float(np.sqrt(np.nanmean(temp_err ** 2)))
        if (
            not np.isfinite(temp_rmse)
            or temp_rmse > max_background_temperature_rmse
        ):
            n_temperature_qc_rejected += 1
            continue

        bg_err = bg_sal - truth_sal
        bg_rmse = float(np.sqrt(np.nanmean(bg_err ** 2)))
        if (
            not np.isfinite(bg_rmse)
            or bg_rmse > max_background_salinity_rmse
        ):
            n_salinity_qc_rejected += 1
            continue

        heave_sal, _ = _apply_heave(emulator, truth_temp, bg_temp, bg_sal, thickness)

        ml_depth = None
        ml_truth_sal = None
        ml_sal = None
        ml_rmse = None
        ml_error_on_model_depth = None
        if ml_emulator is not None:
            (
                ml_sal,
                ml_truth_sal,
                _ml_background_sal,
                _ml_delta_sal,
                ml_depth,
            ) = _apply_ml_jacobian(
                ml_emulator,
                temperature[truth_index],
                salinity[truth_index],
                full_thickness,
                temperature[background_index],
                salinity[background_index],
                full_thickness,
            )
            ml_error = ml_sal - ml_truth_sal
            ml_valid = np.isfinite(ml_error)
            if int(ml_valid.sum()) > 0:
                ml_error_on_model_depth = _interpolate_error_to_model_depth(
                    ml_depth,
                    ml_error,
                    model_depth,
                )

        hv_err = heave_sal - truth_sal
        finite_profile_error = np.isfinite(bg_err) & np.isfinite(hv_err)
        if ml_error_on_model_depth is not None:
            ml_selected_error = ml_error_on_model_depth[selected]
            finite_profile_error = (
                finite_profile_error & np.isfinite(ml_selected_error)
            )
        else:
            ml_selected_error = None
        if int(finite_profile_error.sum()) == 0:
            continue

        bg_rmse = float(np.sqrt(np.nanmean(bg_err[finite_profile_error] ** 2)))
        hv_rmse = float(np.sqrt(np.nanmean(hv_err[finite_profile_error] ** 2)))
        all_bg_rmse.append(bg_rmse)
        all_hv_rmse.append(hv_rmse)
        if ml_selected_error is not None:
            ml_rmse = float(
                np.sqrt(np.nanmean(ml_selected_error[finite_profile_error] ** 2))
            )
            all_ml_rmse.append(ml_rmse)

        finite_error = finite_profile_error & (sel_depth <= max_stats_depth_m)
        finite_selected = selected[finite_error]
        if finite_selected.size > 0:
            depth_count[finite_selected] += 1
            depth_bg_error_sum[finite_selected] += bg_err[finite_error]
            depth_bg_error_sumsq[finite_selected] += bg_err[finite_error] ** 2
            depth_hv_error_sum[finite_selected] += hv_err[finite_error]
            depth_hv_error_sumsq[finite_selected] += hv_err[finite_error] ** 2
            if ml_selected_error is not None:
                depth_ml_count[finite_selected] += 1
                depth_ml_error_sum[finite_selected] += ml_selected_error[finite_error]
                depth_ml_error_sumsq[finite_selected] += (
                    ml_selected_error[finite_error] ** 2
                )

        lon_t = float(longitude[truth_index])
        lat_t = float(latitude[truth_index])
        lon_b = float(longitude[background_index])
        lat_b = float(latitude[background_index])
        surf_dtemp = _first_common_surface_dtemp(
            temperature[truth_index], temperature[background_index], common_valid,
        )

        title = (
            f"truth #{truth_index}  ({lon_t:.2f}°, {lat_t:.2f}°)  "
            f"surf_T={surf_temp:.1f}°C\n"
            f"background #{background_index}  ({lon_b:.2f}°, {lat_b:.2f}°)  "
            f"dT(bg−truth)={surf_dtemp:+.2f}°C  N_levels={int(common_valid.sum())}"
        )

        stem = (
            f"prof{truth_index:05d}"
            f"_lon{lon_t:+.2f}_lat{lat_t:+.2f}"
        ).replace("-", "m").replace("+", "p").replace(".", "d")
        png_path = output_dir / f"{stem}.png"
        if not args.no_profile_plots:
            _plot_profile(
                png_path, sel_depth,
                truth_temp, bg_temp,
                truth_sal, bg_sal, heave_sal,
                title,
                ml_depth,
                ml_truth_sal,
                ml_sal,
            )
        n_processed += 1
        output_note = png_path.name if not args.no_profile_plots else "stats only"
        ml_note = f"  ml_RMSE={ml_rmse:.4f}" if ml_rmse is not None else ""
        print(
            f"[{n_processed:4d}] prof #{truth_index:4d} "
            f"({lon_t:8.3f}, {lat_t:7.3f})  "
            f"T_surf={surf_temp:.1f}  dT={surf_dtemp:+.2f}  "
            f"bg_RMSE={bg_rmse:.4f}  hv_RMSE={hv_rmse:.4f}  "
            f"{ml_note}  → {output_note}"
        )

    print()
    print(f"Qualifying profiles  : {n_qualifying}")
    print(f"Plots written        : {n_processed}")
    print(f"No background found  : {n_no_background}")
    print(f"Temperature QC rejected : {n_temperature_qc_rejected}")
    print(f"Salinity QC rejected : {n_salinity_qc_rejected}")
    if all_bg_rmse:
        bg_depth_rmse, bg_depth_bias = _depth_stats_from_sums(
            depth_count, depth_bg_error_sum, depth_bg_error_sumsq,
        )
        hv_depth_rmse, hv_depth_bias = _depth_stats_from_sums(
            depth_count, depth_hv_error_sum, depth_hv_error_sumsq,
        )
        if ml_emulator is not None:
            ml_depth_rmse, ml_depth_bias = _depth_stats_from_sums(
                depth_ml_count, depth_ml_error_sum, depth_ml_error_sumsq,
            )
        else:
            ml_depth_rmse = None
            ml_depth_bias = None
        stats_png = output_dir / "selected_profile_error_stats_by_depth.png"
        stats_csv = output_dir / "selected_profile_error_stats_by_depth.csv"
        _plot_depth_stats(
            stats_png,
            model_depth[stats_level_indices],
            depth_count[stats_level_indices],
            bg_depth_rmse[stats_level_indices],
            bg_depth_bias[stats_level_indices],
            hv_depth_rmse[stats_level_indices],
            hv_depth_bias[stats_level_indices],
            depth_ml_count[stats_level_indices] if ml_emulator is not None else None,
            ml_depth_rmse[stats_level_indices] if ml_depth_rmse is not None else None,
            ml_depth_bias[stats_level_indices] if ml_depth_bias is not None else None,
        )
        _write_depth_stats_csv(
            stats_csv,
            model_depth[stats_level_indices],
            depth_count[stats_level_indices],
            bg_depth_rmse[stats_level_indices],
            bg_depth_bias[stats_level_indices],
            hv_depth_rmse[stats_level_indices],
            hv_depth_bias[stats_level_indices],
            depth_ml_count[stats_level_indices] if ml_emulator is not None else None,
            ml_depth_rmse[stats_level_indices] if ml_depth_rmse is not None else None,
            ml_depth_bias[stats_level_indices] if ml_depth_bias is not None else None,
            stats_level_indices,
        )
        summary = (
            f"Mean RMSE  background : {np.mean(all_bg_rmse):.4f} PSU  "
            f"heave : {np.mean(all_hv_rmse):.4f} PSU"
        )
        if all_ml_rmse:
            summary += f"  ML : {np.mean(all_ml_rmse):.4f} PSU"
        print(summary)
        print(f"Depth stats plot    : {stats_png}")
        print(f"Depth stats CSV     : {stats_csv}")


if __name__ == "__main__":
    main()
