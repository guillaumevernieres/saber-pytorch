#!/usr/bin/env python3
"""Plot localized heave salinity Jacobian diagnostics for geom100.

The full vertical Jacobian has shape [nnodes, nlevels, 3*nlevels].  This script
keeps the investigation lightweight by computing:

- diagonal temperature sensitivity maps, d(delta S_k) / d(delta T_k)
- row-sum absolute sensitivity by output level
- one representative column's full [output level, input temperature level] block
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from netCDF4 import Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
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


def _read_geom100_fields(
    ocean_file: Path,
    temperature_variable: str,
    salinity_variable: str,
    thickness_variable: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with Dataset(ocean_file) as ds:
        temperature = np.asarray(ds.variables[temperature_variable][0], dtype=np.float32)
        salinity = np.asarray(ds.variables[salinity_variable][0], dtype=np.float32)
        thickness = np.asarray(ds.variables[thickness_variable][0], dtype=np.float32)
    return temperature, salinity, thickness


def _read_grid(
    grid_file: Path,
    longitude_variable: str,
    latitude_variable: str,
) -> tuple[np.ndarray, np.ndarray]:
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
) -> tuple[int, int]:
    dlon = _longitude_delta(lon, target_lon) * np.cos(np.deg2rad(target_lat))
    dlat = lat - target_lat
    distance2 = dlon * dlon + dlat * dlat
    distance2 = np.where(valid, distance2, np.inf)
    iy, ix = np.unravel_index(np.argmin(distance2), distance2.shape)
    if not np.isfinite(distance2[iy, ix]):
        raise RuntimeError("No valid ocean columns found")
    return int(iy), int(ix)


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


def _compute_diagnostics(
    emulator: WeaverTSBalance,
    temperature: np.ndarray,
    salinity: np.ndarray,
    thickness: np.ndarray,
    valid_columns: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    nlevels, ny, nx = temperature.shape
    nnodes = ny * nx
    flat_valid = valid_columns.reshape(-1)

    temp_flat = temperature.reshape(nlevels, nnodes).T
    salt_flat = salinity.reshape(nlevels, nnodes).T
    thick_flat = thickness.reshape(nlevels, nnodes).T

    diag_flat = np.full((nnodes, nlevels), np.nan, dtype=np.float32)
    row_abs_sum_flat = np.full((nnodes, nlevels), np.nan, dtype=np.float32)

    valid_indices = np.flatnonzero(flat_valid)
    rows = torch.arange(nlevels, dtype=torch.long)
    diag_cols = torch.arange(nlevels, dtype=torch.long)

    for start in range(0, valid_indices.size, batch_size):
        idx = valid_indices[start : start + batch_size]
        inputs = np.concatenate(
            [temp_flat[idx], salt_flat[idx], thick_flat[idx]],
            axis=1,
        )
        inputs_t = torch.from_numpy(inputs)
        mask_t = torch.ones(inputs_t.shape[0], 1, dtype=inputs_t.dtype)

        diag = emulator.jac_physical(inputs_t, mask_t, rows, diag_cols)
        full = emulator.jac_from_state(
            {
                emulator.input_names[0]: inputs_t[:, 0*nlevels:1*nlevels],
                emulator.input_names[1]: inputs_t[:, 1*nlevels:2*nlevels],
                emulator.input_names[2]: inputs_t[:, 2*nlevels:3*nlevels],
            },
            mask_t,
        )
        row_abs_sum = full[:, :, :nlevels].abs().sum(dim=2)

        diag_flat[idx] = diag.detach().cpu().numpy()
        row_abs_sum_flat[idx] = row_abs_sum.detach().cpu().numpy()

    diag_maps = diag_flat.T.reshape(nlevels, ny, nx)
    row_abs_sum_maps = row_abs_sum_flat.T.reshape(nlevels, ny, nx)
    return diag_maps, row_abs_sum_maps


def _representative_matrix(
    emulator: WeaverTSBalance,
    temperature: np.ndarray,
    salinity: np.ndarray,
    thickness: np.ndarray,
    iy: int,
    ix: int,
) -> np.ndarray:
    T = torch.from_numpy(temperature[:, iy, ix].reshape(1, -1))
    S = torch.from_numpy(salinity[:, iy, ix].reshape(1, -1))
    h = torch.from_numpy(thickness[:, iy, ix].reshape(1, -1))
    mask = torch.ones(1, 1, dtype=T.dtype)
    jac = emulator.jac_from_state(
        {
            emulator.input_names[0]: T,
            emulator.input_names[1]: S,
            emulator.input_names[2]: h,
        },
        mask,
    )
    return jac[0, :, : T.shape[1]].detach().cpu().numpy()


def _plot_column_matrix(matrix: np.ndarray, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmax = np.nanpercentile(np.abs(matrix), 99.0)
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="upper")
    ax.set_title("Representative column heave Jacobian")
    ax.set_xlabel("input temperature level j")
    ax.set_ylabel("output salinity level k")
    fig.colorbar(image, ax=ax, label="d(delta S_k) / d(delta T_j)")
    fig.savefig(output_dir / "geom100_heave_column_matrix.png", dpi=180)
    plt.close(fig)


def _plot_maps(
    diag_maps: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    map_levels: List[int],
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nlevels = diag_maps.shape[0]
    levels = [level for level in map_levels if 0 <= level < nlevels]
    if not levels:
        levels = [0, nlevels // 3, 2 * nlevels // 3, nlevels - 1]

    values = np.concatenate([diag_maps[level].reshape(-1) for level in levels])
    vmax = np.nanpercentile(np.abs(values), 99.0)
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 1.0

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for ax, level in zip(axes.ravel(), levels[:4]):
        image = ax.pcolormesh(
            lon,
            lat,
            diag_maps[level],
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            shading="nearest",
            rasterized=True,
        )
        ax.set_title(f"level {level}: d(delta S_k)/d(delta T_k)")
        ax.set_xlabel("longitude [deg]")
        ax.set_ylabel("latitude [deg]")
        ax.set_xlim(float(np.nanmin(lon)), float(np.nanmax(lon)))
        ax.set_ylim(float(np.nanmin(lat)), float(np.nanmax(lat)))
        ax.grid(True, alpha=0.25)
    for ax in axes.ravel()[len(levels[:4]) :]:
        ax.axis("off")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="PSU / degC")
    fig.savefig(output_dir / "geom100_heave_diag_maps.png", dpi=180)
    plt.close(fig)


def _plot_profiles(
    diag_maps: np.ndarray,
    row_abs_sum_maps: np.ndarray,
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = np.arange(diag_maps.shape[0])
    abs_diag = np.abs(diag_maps)
    diag_mean = np.nanmean(abs_diag, axis=(1, 2))
    diag_p95 = np.nanpercentile(abs_diag, 95.0, axis=(1, 2))
    rowsum_mean = np.nanmean(row_abs_sum_maps, axis=(1, 2))
    rowsum_p95 = np.nanpercentile(row_abs_sum_maps, 95.0, axis=(1, 2))

    fig, ax = plt.subplots(figsize=(7, 8), constrained_layout=True)
    ax.plot(diag_mean, levels, label="mean |diagonal|")
    ax.plot(diag_p95, levels, label="p95 |diagonal|")
    ax.plot(rowsum_mean, levels, label="mean row |J_T| sum")
    ax.plot(rowsum_p95, levels, label="p95 row |J_T| sum")
    ax.invert_yaxis()
    ax.set_xlabel("sensitivity magnitude [PSU / degC]")
    ax.set_ylabel("model level")
    ax.set_title("Geom100 heave Jacobian vertical summary")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_dir / "geom100_heave_vertical_profiles.png", dpi=180)
    plt.close(fig)


def _write_summary(
    diag_maps: np.ndarray,
    row_abs_sum_maps: np.ndarray,
    output_dir: Path,
) -> None:
    abs_diag = np.abs(diag_maps)
    with (output_dir / "geom100_heave_summary.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "level",
                "diag_mean",
                "diag_abs_mean",
                "diag_abs_p95",
                "diag_abs_max",
                "row_abs_sum_mean",
                "row_abs_sum_p95",
                "row_abs_sum_max",
            ]
        )
        for level in range(diag_maps.shape[0]):
            writer.writerow(
                [
                    level,
                    float(np.nanmean(diag_maps[level])),
                    float(np.nanmean(abs_diag[level])),
                    float(np.nanpercentile(abs_diag[level], 95.0)),
                    float(np.nanmax(abs_diag[level])),
                    float(np.nanmean(row_abs_sum_maps[level])),
                    float(np.nanpercentile(row_abs_sum_maps[level], 95.0)),
                    float(np.nanmax(row_abs_sum_maps[level])),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="emulators/heave_salinity/config.yaml",
        help="YAML config path",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = _load_config(config_path)
    input_cfg = config["input"]
    diag_cfg = config["diagnostics"]
    output_dir = _resolve_path(config["output"]["directory"], config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    ocean_file = _resolve_path(input_cfg["ocean_file"], config_path)
    temperature, salinity, thickness = _read_geom100_fields(
        ocean_file=ocean_file,
        temperature_variable=input_cfg.get("temperature_variable", "Temp"),
        salinity_variable=input_cfg.get("salinity_variable", "Salt"),
        thickness_variable=input_cfg.get("thickness_variable", "h"),
    )
    valid_columns = _valid_column_mask(
        temperature,
        salinity,
        thickness,
        float(config["model"].get("min_valid_thickness_m", 0.1)),
    )
    grid_file = _resolve_path(input_cfg["grid_file"], config_path)
    lon, lat = _read_grid(
        grid_file,
        input_cfg.get("longitude_variable", "lon"),
        input_cfg.get("latitude_variable", "lat"),
    )

    emulator = _make_emulator(config)
    diag_maps, row_abs_sum_maps = _compute_diagnostics(
        emulator,
        temperature,
        salinity,
        thickness,
        valid_columns,
        int(diag_cfg.get("batch_size", 4096)),
    )

    target_lon = float(
        diag_cfg.get("target_lon", diag_cfg.get("reconstruction_lon", -119.5))
    )
    target_lat = float(
        diag_cfg.get("target_lat", diag_cfg.get("reconstruction_lat", 4.36))
    )
    iy, ix = _nearest_valid_column(lon, lat, valid_columns, target_lon, target_lat)

    matrix = _representative_matrix(emulator, temperature, salinity, thickness, iy, ix)
    _plot_column_matrix(matrix, output_dir)
    _plot_maps(
        diag_maps,
        lon,
        lat,
        list(diag_cfg.get("map_levels", [5, 15, 30, 50])),
        output_dir,
    )
    _plot_profiles(diag_maps, row_abs_sum_maps, output_dir)
    _write_summary(diag_maps, row_abs_sum_maps, output_dir)

    np.savez_compressed(
        output_dir / "geom100_heave_representative_column.npz",
        iy=iy,
        ix=ix,
        jacobian_temperature_block=matrix,
        temperature=temperature[:, iy, ix],
        salinity=salinity[:, iy, ix],
        thickness=thickness[:, iy, ix],
    )

    print(f"Input ocean file: {ocean_file}")
    print(f"Valid columns   : {int(valid_columns.sum())} / {valid_columns.size}")
    print(
        f"Representative : iy={iy}, ix={ix}, "
        f"lon={lon[iy, ix]:.3f}, lat={lat[iy, ix]:.3f}"
    )
    print(f"Wrote outputs   : {output_dir}")


if __name__ == "__main__":
    main()
