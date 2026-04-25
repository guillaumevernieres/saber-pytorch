#!/usr/bin/env python3
"""Plot salinity-profile inference examples against reduced-grid depth."""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

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


def _parse_indices(raw: str, n_samples: int, count: int) -> List[int]:
    if raw:
        indices = [int(x.strip()) for x in raw.split(",") if x.strip()]
    else:
        indices = list(range(min(count, n_samples)))
    for idx in indices:
        if idx < 0 or idx >= n_samples:
            raise ValueError(f"sample index {idx} out of range [0, {n_samples})")
    return indices


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot reduced-grid salinity inference examples"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--mom-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--indices", default="")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--min-layer-thickness", type=float, default=0.1)
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = np.load(args.data, allow_pickle=True)
    temp = raw["inputs"].astype(np.float32)
    salt_target = raw["targets"].astype(np.float32)
    y_index = raw["lats"].astype(np.int64)
    x_index = raw["lons"].astype(np.int64)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = _build_model(checkpoint, temp.shape[1], salt_target.shape[1])
    with torch.no_grad():
        salt_pred = model.predict(torch.from_numpy(temp).float()).cpu().numpy()

    h = _read_h(args.mom_file)
    indices = _parse_indices(args.indices, temp.shape[0], args.count)

    for idx in indices:
        y = int(y_index[idx])
        x = int(x_index[idx])
        depth = _target_depths(
            h[y, x, :],
            temp.shape[1],
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

        axes[1].plot(salt_target[idx], depth, "k-o", markersize=3, label="Target")
        axes[1].plot(salt_pred[idx], depth, "r--s", markersize=3, label="Predicted")
        axes[1].set_xlabel("Salt")
        axes[1].set_title("Salt Target vs Output")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(err, depth, "g-o", markersize=3)
        axes[2].axvline(0.0, color="k", linestyle="--", linewidth=1, alpha=0.5)
        axes[2].set_xlabel("Salt error")
        axes[2].set_title(f"Error\nRMSE={rmse:.4f}, bias={bias:.4f}")
        axes[2].grid(True, alpha=0.3)

        axes[0].invert_yaxis()
        fig.suptitle(f"Sample {idx} (y={y}, x={x})")
        fig.tight_layout()

        out = out_dir / f"inference_sample_{idx:04d}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
