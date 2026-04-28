#!/usr/bin/env python3
"""Build matched real Argo T/S profile pairs on the model depth grid."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from saber_pytorch.observations.argo_profiles import (
    build_matched_argo_ts_dataset,
    read_model_depth_grid,
)


DEFAULT_TEMP_FILE = (
    "/home/gvernier/Documents/gomo/gomo-mom6/gdas.202604*/00/analysis/"
    "ocean/diags/insitu_temp_profile_argo.nc"
)
DEFAULT_SALT_FILE = (
    "/home/gvernier/Documents/gomo/gomo-mom6/gdas.202604*/00/analysis/"
    "ocean/diags/insitu_salt_profile_argo.nc"
)
DEFAULT_MODEL_DEPTH_FILE = (
    REPO_ROOT.parent / "i-jedi/test-soca/geom100/MOM.res.nc"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "emulators/heave_salinity/outputs/argo_ts_profiles.npz"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--temp-file",
        default=DEFAULT_TEMP_FILE,
        help="Temperature IODA file or glob pattern",
    )
    parser.add_argument(
        "--salt-file",
        default=DEFAULT_SALT_FILE,
        help="Salinity IODA file or glob pattern",
    )
    parser.add_argument("--model-depth-file", type=Path, default=DEFAULT_MODEL_DEPTH_FILE)
    parser.add_argument("--model-depth-variable", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot-dir", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--max-profiles", type=int, default=None)
    parser.add_argument("--min-profile-levels", type=int, default=2)
    parser.add_argument("--min-valid-model-levels", type=int, default=5)
    parser.add_argument("--allow-extrapolation", action="store_true")
    parser.add_argument("--max-position-difference-deg", type=float, default=0.02)
    parser.add_argument("--max-time-difference-seconds", type=int, default=3600)
    parser.add_argument("--min-depth-overlap-m", type=float, default=1.0)
    parser.add_argument("--temperature-variable", default="waterTemperature")
    parser.add_argument("--salinity-variable", default=None)
    parser.add_argument("--salinity-min", type=float, default=20.0)
    parser.add_argument("--salinity-max", type=float, default=45.0)
    return parser.parse_args()


def _expand_input_files(pattern: str | Path) -> list[Path]:
    pattern_text = str(pattern)
    matches = sorted(Path(path).expanduser() for path in glob.glob(os.path.expanduser(pattern_text)))
    if matches:
        return matches
    path = Path(pattern_text).expanduser()
    if path.exists():
        return [path]
    raise FileNotFoundError(f"No files matched {pattern_text}")


def _cycle_key(path: Path) -> str:
    for part in path.parts:
        if part.startswith("gdas."):
            return part
    return str(path.parent)


def _pair_profile_files(temp_pattern: str | Path, salt_pattern: str | Path) -> list[tuple[Path, Path]]:
    temp_files = _expand_input_files(temp_pattern)
    salt_files = _expand_input_files(salt_pattern)
    salt_by_parent = {path.parent: path for path in salt_files}
    pairs = []
    for temp_path in temp_files:
        salt_path = salt_by_parent.get(temp_path.parent)
        if salt_path is not None:
            pairs.append((temp_path, salt_path))
    if not pairs:
        raise RuntimeError(
            "No temperature/salinity file pairs found. Files are paired by "
            "their common diagnostics directory."
        )
    return pairs


def _as_object_array(values: list[Any]) -> np.ndarray:
    return np.asarray(["" if value is None else str(value) for value in values], dtype=object)


def _save_npz(path: Path, profiles: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if profiles:
        potential_temperature = np.stack(
            [profile["model_potential_temperature"] for profile in profiles]
        )
        salinity = np.stack([profile["model_salinity"] for profile in profiles])
        valid_mask = np.stack([profile["model_valid_mask"] for profile in profiles])
        model_depth = np.asarray(profiles[0]["model_depth"], dtype=np.float64)
    else:
        model_depth = np.asarray([], dtype=np.float64)
        potential_temperature = np.empty((0, 0), dtype=np.float64)
        salinity = np.empty((0, 0), dtype=np.float64)
        valid_mask = np.empty((0, 0), dtype=bool)

    np.savez_compressed(
        path,
        model_depth=model_depth,
        potential_temperature=potential_temperature,
        salinity=salinity,
        valid_mask=valid_mask,
        latitude=np.asarray([profile["latitude"] for profile in profiles], dtype=np.float64),
        longitude=np.asarray([profile["longitude"] for profile in profiles], dtype=np.float64),
        stationID=_as_object_array([profile["stationID"] for profile in profiles]),
        dateTime=np.asarray(
            [-1 if profile["dateTime"] is None else profile["dateTime"] for profile in profiles],
            dtype=np.int64,
        ),
        originalDateTime=np.asarray(
            [
                -1 if profile["originalDateTime"] is None else profile["originalDateTime"]
                for profile in profiles
            ],
            dtype=np.int64,
        ),
        oceanBasin=np.asarray(
            [-1 if profile["oceanBasin"] is None else profile["oceanBasin"] for profile in profiles],
            dtype=np.int32,
        ),
        n_temperature_obs=np.asarray(
            [profile["n_temperature_obs"] for profile in profiles],
            dtype=np.int32,
        ),
        n_salinity_obs=np.asarray(
            [profile["n_salinity_obs"] for profile in profiles],
            dtype=np.int32,
        ),
        n_overlapping_obs=np.asarray(
            [profile["n_overlapping_obs"] for profile in profiles],
            dtype=np.int32,
        ),
        n_valid_model_levels=np.asarray(
            [profile["n_valid_model_levels"] for profile in profiles],
            dtype=np.int32,
        ),
        source_pair_index=np.asarray(
            [profile.get("source_pair_index", -1) for profile in profiles],
            dtype=np.int32,
        ),
        source_cycle=_as_object_array([profile.get("source_cycle") for profile in profiles]),
        source_temperature_file=_as_object_array(
            [profile.get("source_temperature_file") for profile in profiles]
        ),
        source_salinity_file=_as_object_array(
            [profile.get("source_salinity_file") for profile in profiles]
        ),
        summary_json=json.dumps(summary, sort_keys=True),
    )


def _make_plots(plot_dir: Path, profiles: list[dict[str, Any]]) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plot_dir / "mplconfig"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not profiles:
        return

    lat = np.asarray([profile["latitude"] for profile in profiles], dtype=np.float64)
    lon = np.asarray([profile["longitude"] for profile in profiles], dtype=np.float64)
    valid_levels = np.asarray(
        [profile["n_valid_model_levels"] for profile in profiles],
        dtype=np.int32,
    )
    max_valid_depth = []
    for profile in profiles:
        depth = np.asarray(profile["model_depth"], dtype=np.float64)
        mask = np.asarray(profile["model_valid_mask"], dtype=bool)
        max_valid_depth.append(float(depth[mask].max()))
    max_valid_depth = np.asarray(max_valid_depth, dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    scatter = axes[0].scatter(lon, lat, c=max_valid_depth, s=12, cmap="viridis")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].set_title("Retained profiles")
    fig.colorbar(scatter, ax=axes[0], label="Max valid depth (m)")

    axes[1].hist(valid_levels, bins=30, color="0.25")
    axes[1].set_xlabel("Valid model levels")
    axes[1].set_ylabel("Profiles")

    axes[2].hist(max_valid_depth, bins=30, color="0.25")
    axes[2].set_xlabel("Max valid depth (m)")
    axes[2].set_ylabel("Profiles")

    fig.savefig(plot_dir / "argo_ts_profile_summary.png", dpi=150)
    plt.close(fig)

    example = max(profiles, key=lambda profile: profile["n_valid_model_levels"])
    depth = np.asarray(example["model_depth"], dtype=np.float64)
    mask = np.asarray(example["model_valid_mask"], dtype=bool)
    fig, axes = plt.subplots(1, 2, figsize=(7, 6), sharey=True, constrained_layout=True)
    axes[0].plot(example["model_potential_temperature"][mask], depth[mask], marker=".")
    axes[0].set_xlabel("Potential temperature (degC)")
    axes[0].set_ylabel("Depth (m)")
    axes[0].invert_yaxis()
    axes[1].plot(example["model_salinity"][mask], depth[mask], marker=".")
    axes[1].set_xlabel("Salinity")
    axes[1].invert_yaxis()
    fig.suptitle(
        f"Example Argo profile lon={example['longitude']:.2f}, "
        f"lat={example['latitude']:.2f}"
    )
    fig.savefig(plot_dir / "argo_ts_example_profile.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    model_depth = read_model_depth_grid(args.model_depth_file, args.model_depth_variable)
    file_pairs = _pair_profile_files(args.temp_file, args.salt_file)
    profiles: list[dict[str, Any]] = []
    pair_summaries = []
    for pair_index, (temp_file, salt_file) in enumerate(file_pairs):
        remaining = None
        if args.max_profiles is not None:
            remaining = args.max_profiles - len(profiles)
            if remaining <= 0:
                break
        pair_profiles, pair_summary = build_matched_argo_ts_dataset(
            temp_file,
            salt_file,
            model_depth,
            temperature_variable=args.temperature_variable,
            salinity_variable=args.salinity_variable,
            min_profile_levels=args.min_profile_levels,
            min_valid_model_levels=args.min_valid_model_levels,
            allow_extrapolation=args.allow_extrapolation,
            max_position_difference_deg=args.max_position_difference_deg,
            max_time_difference_seconds=args.max_time_difference_seconds,
            min_depth_overlap_m=args.min_depth_overlap_m,
            max_profiles=remaining,
            salinity_min=args.salinity_min,
            salinity_max=args.salinity_max,
        )
        cycle = _cycle_key(temp_file)
        for profile in pair_profiles:
            profile["source_pair_index"] = pair_index
            profile["source_cycle"] = cycle
            profile["source_temperature_file"] = str(temp_file)
            profile["source_salinity_file"] = str(salt_file)
        profiles.extend(pair_profiles)
        pair_summary["source_cycle"] = cycle
        pair_summaries.append(pair_summary)

    summary = {
        "temperature_file_pattern": str(args.temp_file),
        "salinity_file_pattern": str(args.salt_file),
        "file_pairs": len(file_pairs),
        "processed_file_pairs": len(pair_summaries),
        "temperature_obs_variable": (
            pair_summaries[0]["temperature_obs_variable"] if pair_summaries else None
        ),
        "salinity_obs_variable": (
            pair_summaries[0]["salinity_obs_variable"] if pair_summaries else None
        ),
        "temperature_locations": int(
            sum(item["temperature_locations"] for item in pair_summaries)
        ),
        "salinity_locations": int(
            sum(item["salinity_locations"] for item in pair_summaries)
        ),
        "temperature_profiles": int(
            sum(item["temperature_profiles"] for item in pair_summaries)
        ),
        "salinity_profiles": int(sum(item["salinity_profiles"] for item in pair_summaries)),
        "matched_profiles": int(sum(item["matched_profiles"] for item in pair_summaries)),
        "retained_profiles": len(profiles),
        "model_depth_levels": int(np.asarray(model_depth).size),
        "allow_extrapolation": bool(args.allow_extrapolation),
        "min_valid_model_levels": int(args.min_valid_model_levels),
        "salinity_min": args.salinity_min,
        "salinity_max": args.salinity_max,
        "pair_summaries": pair_summaries,
    }
    _save_npz(args.output, profiles, summary)

    plot_dir = args.plot_dir
    if plot_dir is None:
        plot_dir = args.output.parent / "argo_ts_profiles"
    if not args.no_plots:
        _make_plots(plot_dir, profiles)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {args.output}")
    if not args.no_plots:
        print(f"Wrote plots under {plot_dir}")


if __name__ == "__main__":
    main()
