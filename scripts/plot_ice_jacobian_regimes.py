#!/usr/bin/env python3
"""Visualize ice concentration Jacobian across different background regimes.

This script generates plots showing how Jacobian sensitivities vary with:
  - Sea surface temperature (SST)
  - Sea surface salinity (SSS)
  - Sea ice thickness (HI)
  - Snow depth (HS)
  - Sea ice area fraction (aice)

Usage
-----
python scripts/plot_ice_jacobian_regimes.py [--output-dir ./figs]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from saber_pytorch.physics.ice_concentration import SurfaceIceConcentrationEmulator  # noqa: E402


def make_emulator() -> SurfaceIceConcentrationEmulator:
    """Create a default ice concentration emulator."""
    input_names = [
        "sea_water_potential_temperature",
        "sea_water_salinity",
        "sea_ice_volume",
        "sea_ice_snow_volume",
        "sea_ice_area_fraction",
    ]
    output_names = ["sea_ice_area_fraction"]
    return SurfaceIceConcentrationEmulator(
        input_names=input_names,
        output_names=output_names,
    )


def plot_jacobian_vs_sst(
    emulator: SurfaceIceConcentrationEmulator, output_dir: Path
) -> None:
    """Plot d(aice)/dX as a function of SST for different salinities."""
    sst_vals = np.linspace(-3, 2, 50)
    sss_vals = [32.0, 34.0, 36.0]
    aice_val = 0.5
    hi = 0.5
    hs = 0.1

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Ice Concentration Jacobian vs. Sea Surface Temperature", fontsize=14)

    deriv_names = ["d(aice)/dSST", "d(aice)/dSSS", "d(aice)/dHI", "d(aice)/dHS"]
    deriv_indices = [0, 1, 2, 3]

    for ax, deriv_name, deriv_idx in zip(axes.flat, deriv_names, deriv_indices):
        for sss in sss_vals:
            derivs = []
            for sst in sst_vals:
                inputs = torch.tensor(
                    [[sst, sss, hi, hs, aice_val]], dtype=torch.float32
                )
                mask = torch.ones(1, 1, dtype=torch.float32)
                jac = emulator.jac_physical(inputs, mask)
                derivs.append(jac[0, 0, deriv_idx].item())
            ax.plot(sst_vals, derivs, label=f"SSS = {sss:.1f}", marker="o", markersize=3)

        ax.axhline(0, color="k", linestyle="--", linewidth=0.5, alpha=0.3)
        ax.set_xlabel("SST (°C)")
        ax.set_ylabel(deriv_name)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "jacobian_vs_sst.png", dpi=150)
    print(f"Saved: {output_dir / 'jacobian_vs_sst.png'}")
    plt.close()


def plot_jacobian_vs_ice_thickness(
    emulator: SurfaceIceConcentrationEmulator, output_dir: Path
) -> None:
    """Plot d(aice)/dX as a function of ice thickness for different SSTs."""
    hi_vals = np.linspace(0.0, 3.0, 50)
    sst_vals = [-2.0, -1.0, 0.0]
    sss = 34.0
    aice_val = 0.5
    hs = 0.1

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Ice Concentration Jacobian vs. Ice Thickness", fontsize=14)

    deriv_names = ["d(aice)/dSST", "d(aice)/dSSS", "d(aice)/dHI", "d(aice)/dHS"]
    deriv_indices = [0, 1, 2, 3]

    for ax, deriv_name, deriv_idx in zip(axes.flat, deriv_names, deriv_indices):
        for sst in sst_vals:
            derivs = []
            for hi in hi_vals:
                inputs = torch.tensor(
                    [[sst, sss, hi, hs, aice_val]], dtype=torch.float32
                )
                mask = torch.ones(1, 1, dtype=torch.float32)
                jac = emulator.jac_physical(inputs, mask)
                derivs.append(jac[0, 0, deriv_idx].item())
            ax.plot(hi_vals, derivs, label=f"SST = {sst:.1f}°C", marker="o", markersize=3)

        ax.axhline(0, color="k", linestyle="--", linewidth=0.5, alpha=0.3)
        ax.set_xlabel("Ice Thickness (m)")
        ax.set_ylabel(deriv_name)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "jacobian_vs_hi.png", dpi=150)
    print(f"Saved: {output_dir / 'jacobian_vs_hi.png'}")
    plt.close()


def plot_jacobian_vs_aice_weight(
    emulator: SurfaceIceConcentrationEmulator, output_dir: Path
) -> None:
    """Plot d(aice)/dX as a function of prior ice area fraction."""
    aice_vals = np.linspace(0.0, 1.0, 50)
    conditions = [
        {"sst": -2.0, "sss": 34.0, "label": "Cold (SST=-2°C, SSS=34)"},
        {"sst": -1.0, "sss": 34.0, "label": "Moderate (SST=-1°C, SSS=34)"},
        {"sst": -0.5, "sss": 32.0, "label": "Warm fresher (SST=-0.5°C, SSS=32)"},
    ]
    hi = 0.5
    hs = 0.1

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Ice Concentration Jacobian vs. Prior aice (Weight Function)", fontsize=14)

    deriv_names = ["d(aice)/dSST", "d(aice)/dSSS", "d(aice)/dHI", "d(aice)/dHS"]
    deriv_indices = [0, 1, 2, 3]

    for ax, deriv_name, deriv_idx in zip(axes.flat, deriv_names, deriv_indices):
        for cond in conditions:
            derivs = []
            for aice in aice_vals:
                inputs = torch.tensor(
                    [[cond["sst"], cond["sss"], hi, hs, aice]], dtype=torch.float32
                )
                mask = torch.ones(1, 1, dtype=torch.float32)
                jac = emulator.jac_physical(inputs, mask)
                derivs.append(jac[0, 0, deriv_idx].item())
            ax.plot(aice_vals, derivs, label=cond["label"], marker="o", markersize=3)

        ax.axhline(0, color="k", linestyle="--", linewidth=0.5, alpha=0.3)
        ax.set_xlabel("Prior Sea Ice Area Fraction")
        ax.set_ylabel(deriv_name)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "jacobian_vs_aice.png", dpi=150)
    print(f"Saved: {output_dir / 'jacobian_vs_aice.png'}")
    plt.close()


def plot_jacobian_magnitude_2d_heatmap(
    emulator: SurfaceIceConcentrationEmulator, output_dir: Path
) -> None:
    """Plot |Jacobian| as a 2D heatmap over SST and SSS space."""
    sst_vals = np.linspace(-3, 2, 40)
    sss_vals = np.linspace(30, 37, 40)
    aice_val = 0.5
    hi = 0.5
    hs = 0.1

    # Compute |J| = sqrt(sum of squared derivatives)
    jac_mag = np.zeros((len(sss_vals), len(sst_vals)))

    for i, sss in enumerate(sss_vals):
        for j, sst in enumerate(sst_vals):
            inputs = torch.tensor(
                [[sst, sss, hi, hs, aice_val]], dtype=torch.float32
            )
            mask = torch.ones(1, 1, dtype=torch.float32)
            jac = emulator.jac_physical(inputs, mask)
            # Frobenius norm of the 4-element Jacobian row
            jac_mag[i, j] = torch.norm(jac[0, 0, :]).item()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.contourf(sst_vals, sss_vals, jac_mag, levels=20, cmap="viridis")
    ax.contour(sst_vals, sss_vals, jac_mag, levels=10, colors="white", alpha=0.3, linewidths=0.5)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("|Jacobian| (Frobenius norm)", fontsize=11)

    ax.set_xlabel("SST (°C)", fontsize=11)
    ax.set_ylabel("SSS (psu)", fontsize=11)
    ax.set_title(
        f"Jacobian Magnitude in SST-SSS Space\n(aice={aice_val}, hi={hi}m, hs={hs}m)",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "jacobian_magnitude_2d.png", dpi=150)
    print(f"Saved: {output_dir / 'jacobian_magnitude_2d.png'}")
    plt.close()


def plot_individual_derivatives_2d(
    emulator: SurfaceIceConcentrationEmulator, output_dir: Path
) -> None:
    """Plot each derivative as a 2D heatmap over SST/SSS space."""
    sst_vals = np.linspace(-3, 2, 40)
    sss_vals = np.linspace(30, 37, 40)
    aice_val = 0.5
    hi = 0.5
    hs = 0.1

    deriv_names = ["d(aice)/dSST", "d(aice)/dSSS", "d(aice)/dHI", "d(aice)/dHS"]
    deriv_indices = [0, 1, 2, 3]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Individual Jacobian Components (aice={aice_val}, hi={hi}m, hs={hs}m)", fontsize=14)

    for ax, deriv_name, deriv_idx in zip(axes.flat, deriv_names, deriv_indices):
        deriv_map = np.zeros((len(sss_vals), len(sst_vals)))

        for i, sss in enumerate(sss_vals):
            for j, sst in enumerate(sst_vals):
                inputs = torch.tensor(
                    [[sst, sss, hi, hs, aice_val]], dtype=torch.float32
                )
                mask = torch.ones(1, 1, dtype=torch.float32)
                jac = emulator.jac_physical(inputs, mask)
                deriv_map[i, j] = jac[0, 0, deriv_idx].item()

        im = ax.contourf(sst_vals, sss_vals, deriv_map, levels=20, cmap="RdBu_r")
        ax.contour(sst_vals, sss_vals, deriv_map, levels=10, colors="gray", alpha=0.2, linewidths=0.5)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(deriv_name, fontsize=10)

        ax.set_xlabel("SST (°C)", fontsize=10)
        ax.set_ylabel("SSS (psu)", fontsize=10)
        ax.set_title(deriv_name, fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "jacobian_components_2d.png", dpi=150)
    print(f"Saved: {output_dir / 'jacobian_components_2d.png'}")
    plt.close()


def plot_freezing_tolerance_effect(
    emulator: SurfaceIceConcentrationEmulator, output_dir: Path
) -> None:
    """Illustrate the freezing_tolerance parameter cutoff."""
    sst_vals = np.linspace(-3, 2, 50)
    sss = 34.0
    aice_val = 0.5
    hi = 0.5
    hs = 0.1

    # Compute the freezing temperature
    sss_eff = max(sss, 0.0)
    tf = emulator.tf0 + emulator.tf_s_linear * sss_eff + emulator.tf_s_pow * sss_eff * np.sqrt(sss_eff)

    fig, ax = plt.subplots(figsize=(11, 7))

    # Plot d(aice)/dSST across SST range
    derivs_sst = []
    derivs_sss = []
    for sst in sst_vals:
        inputs = torch.tensor([[sst, sss, hi, hs, aice_val]], dtype=torch.float32)
        mask = torch.ones(1, 1, dtype=torch.float32)
        jac = emulator.jac_physical(inputs, mask)
        derivs_sst.append(jac[0, 0, 0].item())
        derivs_sss.append(jac[0, 0, 1].item())

    ax.plot(sst_vals, derivs_sst, "o-", linewidth=2, markersize=4, label="d(aice)/dSST")
    ax.plot(sst_vals, derivs_sss, "s-", linewidth=2, markersize=4, label="d(aice)/dSSS")

    # Mark the freezing point and tolerance band
    ax.axvline(tf, color="red", linestyle="--", linewidth=2, label=f"Freezing pt: Tf={tf:.2f}°C")
    ax.axvspan(
        tf - emulator.freezing_tolerance,
        tf + emulator.freezing_tolerance,
        alpha=0.2,
        color="green",
        label=f"Active zone (±{emulator.freezing_tolerance:.1f}°C)",
    )
    ax.axhline(0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)

    ax.set_xlabel("SST (°C)", fontsize=12)
    ax.set_ylabel("Jacobian Derivative", fontsize=12)
    ax.set_title(
        f"Freezing-Point Cutoff Effect (SSS={sss}, aice={aice_val})",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc="best")

    plt.tight_layout()
    plt.savefig(output_dir / "freezing_tolerance_effect.png", dpi=150)
    print(f"Saved: {output_dir / 'freezing_tolerance_effect.png'}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate figures showing ice concentration Jacobian regimes"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".") / "figs",
        help="Output directory for figures",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    emulator = make_emulator()

    print(f"Generating plots in {output_dir}")
    print(f"  Freezing tolerance: {emulator.freezing_tolerance}°C")
    print(f"  tf0={emulator.tf0}, tf_s_linear={emulator.tf_s_linear}, tf_s_pow={emulator.tf_s_pow}")

    plot_jacobian_vs_sst(emulator, output_dir)
    plot_jacobian_vs_ice_thickness(emulator, output_dir)
    plot_jacobian_vs_aice_weight(emulator, output_dir)
    plot_jacobian_magnitude_2d_heatmap(emulator, output_dir)
    plot_individual_derivatives_2d(emulator, output_dir)
    plot_freezing_tolerance_effect(emulator, output_dir)

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
