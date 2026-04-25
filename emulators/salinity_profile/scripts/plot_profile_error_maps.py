#!/usr/bin/env python3
"""Plot horizontal maps of profile RMSE and bias for salinity inference."""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from saber_pytorch.ml.ffnn import FFNN  # noqa: E402


def _read_wet_mask(mom_file: str, min_layer_thickness: float) -> np.ndarray:
    try:
        import netCDF4 as nc
    except ImportError as exc:
        raise RuntimeError("netCDF4 is required when --mom-file is supplied") from exc

    with nc.Dataset(mom_file) as ds:
        h = ds.variables["h"][:]
        if np.ma.is_masked(h):
            h = np.ma.filled(h, np.nan)
        h = np.asarray(h)
        if h.ndim == 4 and h.shape[0] == 1:
            h = h[0]
        if h.ndim != 3:
            raise ValueError(f"Expected h to be 3D after squeezing time, got {h.shape}")
        if h.shape[0] < 200:
            h = np.transpose(h, (1, 2, 0))
        return np.any(np.isfinite(h) & (h > min_layer_thickness), axis=-1)


def _sample_wet_along_triangle_edges(
    vertices: np.ndarray,
    wet_mask: np.ndarray,
) -> bool:
    samples = [vertices.mean(axis=0)]
    for start, end in ((0, 1), (1, 2), (2, 0)):
        p0 = vertices[start]
        p1 = vertices[end]
        steps = max(2, int(np.ceil(np.max(np.abs(p1 - p0)))) + 1)
        weights = np.linspace(0.0, 1.0, steps)
        samples.extend((1.0 - w) * p0 + w * p1 for w in weights)

    ny, nx = wet_mask.shape
    for point in samples:
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        if x < 0 or x >= nx or y < 0 or y >= ny:
            return False
        if not bool(wet_mask[y, x]):
            return False
    return True


def _build_triangle_mask(
    triangulation,
    x: np.ndarray,
    y: np.ndarray,
    wet_mask: Optional[np.ndarray],
    max_edge: Optional[float],
) -> np.ndarray:
    triangles = triangulation.triangles
    mask = np.zeros(triangles.shape[0], dtype=bool)
    points = np.column_stack([x, y])

    for idx, triangle in enumerate(triangles):
        vertices = points[triangle]
        edges = np.array(
            [
                np.linalg.norm(vertices[1] - vertices[0]),
                np.linalg.norm(vertices[2] - vertices[1]),
                np.linalg.norm(vertices[0] - vertices[2]),
            ]
        )
        if max_edge is not None and np.max(edges) > max_edge:
            mask[idx] = True
            continue
        if wet_mask is not None and not _sample_wet_along_triangle_edges(vertices, wet_mask):
            mask[idx] = True

    return mask


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


def _contour_map(ax, triangulation, values, title, label, cmap, vmin=None, vmax=None):
    if vmin is None:
        vmin = float(np.nanmin(values))
    if vmax is None:
        vmax = float(np.nanmax(values))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0e-6

    levels = np.linspace(vmin, vmax, 31)
    cf = ax.tricontourf(triangulation, values, levels=levels, cmap=cmap, extend="both")
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    return cf, label


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot horizontal maps of salinity profile RMSE and bias"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="profile_error_map")
    parser.add_argument("--rmse-vmax", type=float, default=None)
    parser.add_argument("--bias-abs-max", type=float, default=None)
    parser.add_argument(
        "--mom-file",
        default=None,
        help="Optional MOM restart file with h; used to mask triangles crossing land",
    )
    parser.add_argument("--min-layer-thickness", type=float, default=0.1)
    parser.add_argument(
        "--max-triangle-edge",
        type=float,
        default=25.0,
        help="Mask triangles with an edge longer than this many grid cells",
    )
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = np.load(args.data, allow_pickle=True)
    inputs = raw["inputs"].astype(np.float32)
    targets = raw["targets"].astype(np.float32)
    y_index = raw["lats"].astype(np.float32)
    x_index = raw["lons"].astype(np.float32)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = _build_model(checkpoint, inputs.shape[1], targets.shape[1])
    with torch.no_grad():
        preds = model.predict(torch.from_numpy(inputs).float()).cpu().numpy()

    err = preds - targets
    rmse = np.sqrt(np.mean(err ** 2, axis=1))
    bias = np.mean(err, axis=1)
    bias_abs = (
        float(args.bias_abs_max)
        if args.bias_abs_max is not None
        else float(np.nanmax(np.abs(bias)))
    )

    wet_mask = (
        _read_wet_mask(args.mom_file, args.min_layer_thickness)
        if args.mom_file is not None
        else None
    )
    triangulation = mtri.Triangulation(x_index, y_index)
    triangle_mask = _build_triangle_mask(
        triangulation,
        x_index,
        y_index,
        wet_mask,
        args.max_triangle_edge,
    )
    triangulation.set_mask(triangle_mask)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    sc, label = _contour_map(
        axes[0],
        triangulation,
        rmse,
        "Profile RMSE",
        "Salt RMSE",
        "viridis",
        vmin=0.0,
        vmax=args.rmse_vmax,
    )
    fig.colorbar(sc, ax=axes[0], label=label)

    sc, label = _contour_map(
        axes[1],
        triangulation,
        bias,
        "Profile Bias",
        "Salt bias",
        "RdBu_r",
        vmin=-bias_abs,
        vmax=bias_abs,
    )
    fig.colorbar(sc, ax=axes[1], label=label)

    out_png = out_dir / f"{args.prefix}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    out_csv = out_dir / f"{args.prefix}.csv"
    table = np.column_stack([x_index, y_index, rmse, bias])
    np.savetxt(
        out_csv,
        table,
        delimiter=",",
        header="x_index,y_index,profile_rmse,profile_bias",
        comments="",
    )

    print(f"Saved {out_png}")
    print(f"Saved {out_csv}")
    print(
        f"Triangulation: masked {int(np.sum(triangle_mask))} of "
        f"{int(triangle_mask.size)} triangles"
    )
    print(
        "RMSE range: "
        f"{float(np.min(rmse)):.6f} to {float(np.max(rmse)):.6f}; "
        f"bias range: {float(np.min(bias)):.6f} to {float(np.max(bias)):.6f}"
    )


if __name__ == "__main__":
    main()
