#!/usr/bin/env python3
"""Plot salinity-profile inference examples against reduced-grid depth."""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import netCDF4 as nc
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from saber_pytorch.ml.ffnn import FFNN  # noqa: E402


def _build_model(checkpoint: Dict, input_size: int, output_size: int) -> FFNN:
    model_cfg = checkpoint["config"]["model"]
    model = FFNN(
        input_size=input_size,
        output_size=output_size,
        hidden_size=int(model_cfg["hidden_size"]),
        hidden_layers=int(model_cfg.get("hidden_layers", 2)),
        activation=str(model_cfg.get("activation", "gelu")),
        use_conv1d=bool(model_cfg.get("use_conv1d", False)),
        conv_channels=int(model_cfg.get("conv_channels", 32)),
        conv_kernel_size=int(model_cfg.get("conv_kernel_size", 3)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _read_h(mom_file: str) -> np.ndarray:
    with nc.Dataset(mom_file) as ds:
        h = ds.variables["h"][:]
        if np.ma.is_masked(h):
            h = np.ma.filled(h, np.nan)
        if h.ndim == 4 and h.shape[0] == 1:
            h = h[0]
        if h.shape[0] < 200:
            h = np.transpose(h, (1, 2, 0))
        return h.astype(np.float32)


def _target_depths(
    h_profile: np.ndarray,
    target_num_levels: int,
    min_layer_thickness: float,
) -> np.ndarray:
    valid = np.isfinite(h_profile) & (h_profile > min_layer_thickness)
    safe_h = np.where(valid, h_profile, 0.0).astype(np.float32)
    source_depths = np.cumsum(safe_h) - 0.5 * safe_h
    valid_depths = source_depths[valid]
    if len(valid_depths) == 0:
        return np.arange(target_num_levels, dtype=np.float32)
    if len(valid_depths) == 1:
        return np.full(target_num_levels, valid_depths[0], dtype=np.float32)
    return np.linspace(
        valid_depths[0],
        valid_depths[-1],
        target_num_levels,
        dtype=np.float32,
    )


def _depths_from_thickness(thickness_profile: np.ndarray) -> np.ndarray:
    safe_thickness = np.where(
        np.isfinite(thickness_profile) & (thickness_profile > 0.0),
        thickness_profile,
        0.0,
    ).astype(np.float32)
    return np.cumsum(safe_thickness) - 0.5 * safe_thickness


def _parse_indices(raw: str, n_samples: int) -> List[int]:
    indices = [int(x.strip()) for x in raw.split(",") if x.strip()]
    for idx in indices:
        if idx < 0 or idx >= n_samples:
            raise ValueError(f"sample index {idx} out of range [0, {n_samples})")
    return indices


def _select_random_indices(
    surface_temp: np.ndarray,
    count: int,
    min_temp: Optional[float],
    max_temp: Optional[float],
    seed: Optional[int],
) -> List[int]:
    if count < 1:
        raise ValueError("--count must be at least 1")

    candidates = np.isfinite(surface_temp)
    if min_temp is not None:
        candidates &= surface_temp >= min_temp
    if max_temp is not None:
        candidates &= surface_temp <= max_temp

    candidate_indices = np.flatnonzero(candidates)
    if len(candidate_indices) == 0:
        bounds = []
        if min_temp is not None:
            bounds.append(f">= {min_temp:g}")
        if max_temp is not None:
            bounds.append(f"<= {max_temp:g}")
        suffix = f" with surface temperature {' and '.join(bounds)}" if bounds else ""
        raise ValueError(f"No profiles found{suffix}")

    rng = np.random.default_rng(seed)
    sample_count = min(count, len(candidate_indices))
    selected = rng.choice(candidate_indices, size=sample_count, replace=False)
    return sorted(selected.tolist())


def _count_matching_profiles(
    surface_temp: np.ndarray,
    min_temp: Optional[float],
    max_temp: Optional[float],
) -> int:
    candidates = np.isfinite(surface_temp)
    if min_temp is not None:
        candidates &= surface_temp >= min_temp
    if max_temp is not None:
        candidates &= surface_temp <= max_temp
    return int(np.count_nonzero(candidates))


def _select_indices(
    raw: str,
    surface_temp: np.ndarray,
    count: int,
    min_temp: Optional[float],
    max_temp: Optional[float],
    seed: Optional[int],
) -> List[int]:
    if raw:
        return _parse_indices(raw, len(surface_temp))
    return _select_random_indices(surface_temp, count, min_temp, max_temp, seed)


def _metadata_dict(raw: np.lib.npyio.NpzFile) -> Dict:
    if "metadata" not in raw:
        return {}
    metadata = raw["metadata"]
    if hasattr(metadata, "item"):
        return metadata.item()
    return dict(metadata)


def _depth_spacing_summary(depths: np.ndarray, indices: List[int]) -> str:
    ratios: List[float] = []
    for idx in indices:
        spacing = np.diff(depths[idx])
        spacing = spacing[np.isfinite(spacing) & (spacing > 0.0)]
        if len(spacing) == 0:
            continue
        ratios.append(float(np.max(spacing) / np.min(spacing)))

    if not ratios:
        return "no finite positive spacing"
    return (
        f"min/max spacing ratio across plotted profiles: "
        f"{min(ratios):.2f}-{max(ratios):.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot reduced-grid salinity inference examples"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--mom-file",
        default=None,
        help="MOM restart file with h variable; only needed when the training "
             "data does not contain thickness (old format without reduced dz).",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--indices", default="")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument(
        "--surface-temp-min",
        type=float,
        default=None,
        help="Only randomly sample profiles with surface temperature >= this value",
    )
    parser.add_argument(
        "--surface-temp-max",
        type=float,
        default=None,
        help="Only randomly sample profiles with surface temperature <= this value",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible profile selection",
    )
    parser.add_argument("--min-layer-thickness", type=float, default=0.1)
    args = parser.parse_args()

    if (
        args.surface_temp_min is not None
        and args.surface_temp_max is not None
        and args.surface_temp_min > args.surface_temp_max
    ):
        raise ValueError("--surface-temp-min cannot be greater than --surface-temp-max")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = np.load(args.data, allow_pickle=True)
    model_inputs = raw["inputs"].astype(np.float32)
    salt_target = raw["targets"].astype(np.float32)
    target_levels = salt_target.shape[1]
    if model_inputs.shape[1] >= 2 * target_levels:
        temp = model_inputs[:, :target_levels]
        thickness = model_inputs[:, target_levels:2 * target_levels]
    else:
        temp = model_inputs
        thickness = None
    lats = raw["lats"].astype(np.float32)
    lons = raw["lons"].astype(np.float32)
    metadata = _metadata_dict(raw)
    reduced_grid = metadata.get("reduced_grid", {})
    saved_target_depths = (
        raw["target_depths"].astype(np.float32)
        if "target_depths" in raw
        else None
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = _build_model(checkpoint, model_inputs.shape[1], salt_target.shape[1])
    with torch.no_grad():
        salt_pred = model.predict(torch.from_numpy(model_inputs).float()).cpu().numpy()

    surface_temp = temp[:, 0]
    indices = _select_indices(
        args.indices,
        surface_temp,
        args.count,
        args.surface_temp_min,
        args.surface_temp_max,
        args.seed,
    )

    if args.indices:
        print(f"Plotting requested profiles: {indices}")
    else:
        n_matching = _count_matching_profiles(
            surface_temp,
            args.surface_temp_min,
            args.surface_temp_max,
        )
        print(
            f"Randomly selected {len(indices)} profiles from "
            f"{n_matching} profiles matching the surface-temperature bounds"
        )

    if thickness is not None:
        input_depths = np.stack(
            [_depths_from_thickness(profile) for profile in thickness],
            axis=0,
        )
        method = reduced_grid.get("method", "unknown")
        weight = reduced_grid.get("gradient_weight", "unknown")
        print(
            f"Using depth reconstructed from reduced sea_water_cell_thickness "
            f"input (method={method}, gradient_weight={weight}); "
            f"{_depth_spacing_summary(input_depths, indices)}"
        )
    elif saved_target_depths is None:
        if args.mom_file is None:
            raise ValueError(
                "Training data has no thickness input and no saved "
                "target_depths. Supply --mom-file to derive depths from h."
            )
        h = _read_h(args.mom_file)
        input_depths = None
        print(
            "Warning: training data has no reduced thickness input; plotting "
            "uniform depths from the MOM layer thickness."
        )
    else:
        h = None
        input_depths = saved_target_depths
        method = reduced_grid.get("method", "unknown")
        weight = reduced_grid.get("gradient_weight", "unknown")
        print(
            f"Using saved target_depths from training data "
            f"(method={method}, gradient_weight={weight}); "
            f"{_depth_spacing_summary(saved_target_depths, indices)}"
        )

    for idx in indices:
        lat = float(lats[idx])
        lon = float(lons[idx])
        if input_depths is not None:
            depth = input_depths[idx]
        else:
            y = int(round(lat))
            x = int(round(lon))
            depth = _target_depths(
                h[y, x, :],  # type: ignore[index]
                target_levels,
                args.min_layer_thickness,
            )
        err = salt_pred[idx] - salt_target[idx]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        bias = float(np.mean(err))

        fig, axes = plt.subplots(1, 3, figsize=(14, 7), sharey=True)

        axes[0].plot(temp[idx], depth, "b-o", markersize=3, linewidth=1.5)
        axes[0].set_xlabel("Temp")
        axes[0].set_ylabel("Depth (m)")
        axes[0].set_title("Input Temp")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(
            salt_target[idx], depth, "k-o", markersize=3, label="Target"
        )
        axes[1].plot(
            salt_pred[idx], depth, "r--s", markersize=3, label="Predicted"
        )
        axes[1].set_xlabel("Salt")
        axes[1].set_title("Salt Target vs Output")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(err, depth, "g-o", markersize=3)
        axes[2].axvline(
            0.0, color="k", linestyle="--", linewidth=1, alpha=0.5
        )
        axes[2].set_xlabel("Salt error")
        axes[2].set_title(f"Error\nRMSE={rmse:.4f}, bias={bias:.4f}")
        axes[2].grid(True, alpha=0.3)

        axes[0].invert_yaxis()
        fig.suptitle(
            f"Sample {idx} "
            f"(lon={lon:.2f}, lat={lat:.2f}, "
            f"surface temp={surface_temp[idx]:.3f})"
        )
        fig.tight_layout()

        out = out_dir / f"inference_sample_{idx:04d}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
