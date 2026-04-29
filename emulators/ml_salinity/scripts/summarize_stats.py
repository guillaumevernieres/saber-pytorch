#!/usr/bin/env python3
"""Summarize salinity-profile emulator skill and reduced-grid Jacobians."""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

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


def _skill_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[Dict, np.ndarray]:
    err = y_pred - y_true
    sse = float(np.sum(err ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    err_var = float(np.var(err))
    target_var = float(np.var(y_true))

    summary = {
        "n_samples": int(y_true.shape[0]),
        "n_levels": int(y_true.shape[1]),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "target_std": float(np.std(y_true)),
        "prediction_std": float(np.std(y_pred)),
        "r2": float(1.0 - sse / sst) if sst > 0.0 else float("nan"),
        "explained_variance": (
            float(1.0 - err_var / target_var) if target_var > 0.0 else float("nan")
        ),
    }

    rows = []
    for level in range(y_true.shape[1]):
        yt = y_true[:, level]
        yp = y_pred[:, level]
        er = yp - yt
        level_sse = float(np.sum(er ** 2))
        level_sst = float(np.sum((yt - np.mean(yt)) ** 2))
        level_var = float(np.var(yt))
        rows.append(
            (
                level,
                float(np.sqrt(np.mean(er ** 2))),
                float(np.mean(np.abs(er))),
                float(np.mean(er)),
                float(1.0 - level_sse / level_sst) if level_sst > 0.0 else float("nan"),
                float(1.0 - np.var(er) / level_var) if level_var > 0.0 else float("nan"),
                float(np.std(yt)),
            )
        )
    return summary, np.array(rows, dtype=np.float64)


def _jacobian_stats(model: FFNN, inputs: np.ndarray, max_samples: int) -> Dict:
    n = min(max_samples, inputs.shape[0])
    x = torch.from_numpy(inputs[:n]).float()
    jac = model._jac_physical(x).detach().cpu().numpy()
    frob = np.linalg.norm(jac.reshape(n, -1), axis=1)
    jac_mean = np.mean(jac, axis=0)
    jac_abs_mean = np.mean(np.abs(jac), axis=0)
    return {
        "n_samples": int(n),
        "frobenius_mean": float(np.mean(frob)),
        "frobenius_std": float(np.std(frob)),
        "mean_abs_element": float(np.mean(np.abs(jac))),
        "max_abs_element": float(np.max(np.abs(jac))),
        "diag_abs_mean": float(np.mean(np.abs(np.diagonal(jac, axis1=1, axis2=2)))),
        "jacobian_mean": jac_mean,
        "jacobian_abs_mean": jac_abs_mean,
    }


def _write_per_level(path: Path, rows: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["level", "rmse", "mae", "bias", "r2", "explained_variance", "target_std"]
        )
        writer.writerows(rows.tolist())


def _write_plots(out_dir: Path, y_true: np.ndarray, y_pred: np.ndarray, per_level: np.ndarray, jac: Dict) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true.ravel(), y_pred.ravel(), s=3, alpha=0.35)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    ax.set_xlabel("Target Salt")
    ax.set_ylabel("Predicted Salt")
    ax.set_title("Salinity Emulator Prediction Skill")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "prediction_scatter.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(per_level[:, 0], per_level[:, 5], marker="o")
    ax.set_xlabel("Reduced target level")
    ax.set_ylabel("Explained variance")
    ax.set_title("Explained Variance by Reduced Level")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "explained_variance_by_level.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(jac["jacobian_abs_mean"], origin="upper", aspect="auto")
    fig.colorbar(im, ax=ax, label="mean |dSalt/dTemp|")
    ax.set_xlabel("Reduced Temp input level")
    ax.set_ylabel("Reduced Salt output level")
    ax.set_title("Mean Absolute Reduced-Grid Jacobian")
    fig.tight_layout()
    fig.savefig(out_dir / "jacobian_abs_mean.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize salinity-profile emulator skill and Jacobians"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-jacobian-samples", type=int, default=32)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.data, allow_pickle=True)
    inputs = data["inputs"].astype(np.float32)
    targets = data["targets"].astype(np.float32)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = _build_model(checkpoint, inputs.shape[1], targets.shape[1])
    with torch.no_grad():
        preds = model.predict(torch.from_numpy(inputs).float()).cpu().numpy()

    skill, per_level = _skill_stats(targets, preds)
    jac = _jacobian_stats(model, inputs, args.max_jacobian_samples)

    np.savetxt(out_dir / "jacobian_mean.csv", jac["jacobian_mean"], delimiter=",")
    np.savetxt(out_dir / "jacobian_abs_mean.csv", jac["jacobian_abs_mean"], delimiter=",")
    _write_per_level(out_dir / "per_level_skill.csv", per_level)

    summary = {
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "skill": skill,
        "jacobian": {k: v for k, v in jac.items() if not isinstance(v, np.ndarray)},
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    _write_plots(out_dir, targets, preds, per_level, jac)

    print(json.dumps(summary, indent=2))
    print(f"Saved stats to: {out_dir}")


if __name__ == "__main__":
    main()
