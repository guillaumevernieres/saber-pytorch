#!/usr/bin/env python3
"""Plot inference fields and Jacobian sensitivities for trained surface models.

This script is adapted for the saber-pytorch repository.
It loads a training checkpoint + normalization, reads atmosphere/ocean-ice
NetCDF inputs through UFSEmulatorDataBuilder, runs physical-space inference,
and writes domain maps for predictions and d(output)/d(input) sensitivities.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from saber_pytorch.ml.data import UFSEmulatorDataBuilder  # noqa: E402
from saber_pytorch.ml.ffnn import FFNN  # noqa: E402
from saber_pytorch.ml.training import load_config  # noqa: E402


class SaberInferencePlotter:
    """Load model/checkpoint, run inference, and produce maps."""

    def __init__(self, model_path: str, config_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.input_variables = []
        self.output_variables = []
        self.input_units: Dict[str, str] = {}
        self.output_units: Dict[str, str] = {}

        self.model, self.config = self._load_model_and_config(model_path, config_path)

    def _setup_variables_from_config(self, config: Dict) -> None:
        variables_config = config.get("variables", {})
        self.input_variables = list(variables_config.get("input_variables", []))
        self.output_variables = list(variables_config.get("output_variables", ["aice"]))

        self.input_units = {
            "sst": "degC",
            "sss": "psu",
            "tair": "K",
            "tsfc": "K",
            "hi": "m",
            "hs": "m",
            "sice": "psu",
            "uocn": "m/s",
            "vocn": "m/s",
            "uatm": "m/s",
            "vatm": "m/s",
            "qref": "kg/kg",
            "flwdn": "W/m^2",
            "fswdn": "W/m^2",
        }
        self.output_units = {
            "aice": "fraction",
            "hi": "m",
            "hs": "m",
            "sice": "psu",
            "tair": "K",
        }

        print(f"Input variables ({len(self.input_variables)}): {self.input_variables}")
        print(f"Output variables ({len(self.output_variables)}): {self.output_variables}")

    def _load_model_and_config(
        self, model_path: str, config_path: Optional[str] = None
    ) -> Tuple[FFNN, Dict]:
        checkpoint_path = Path(model_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

        print(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if config_path:
            config = load_config(config_path)
            print(f"Using external config: {config_path}")
        else:
            config = checkpoint.get("config")
            if config is None:
                raise RuntimeError(
                    "Checkpoint does not contain config and --config was not provided"
                )

        self._setup_variables_from_config(config)

        model_cfg = config.get("model", {})
        input_size = int(model_cfg.get("input_size", len(self.input_variables)))
        output_size = int(model_cfg.get("output_size", len(self.output_variables)))

        model = FFNN(
            input_size=input_size,
            output_size=output_size,
            hidden_size=int(model_cfg.get("hidden_size", 64)),
            hidden_layers=int(model_cfg.get("hidden_layers", 3)),
            activation=str(model_cfg.get("activation", "gelu")),
            use_conv1d=bool(model_cfg.get("use_conv1d", False)),
            conv_channels=int(model_cfg.get("conv_channels", 32)),
            conv_kernel_size=int(model_cfg.get("conv_kernel_size", 3)),
        )

        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state)

        norm_path = checkpoint_path.parent / "normalization.pt"
        if norm_path.exists():
            moments = torch.load(norm_path, map_location=self.device, weights_only=False)
            model.init_norm(
                moments["input_mean"].float(),
                moments["input_std"].float(),
                moments["output_mean"].float(),
                moments["output_std"].float(),
            )
            print(f"Loaded normalization from: {norm_path}")
        else:
            print("Warning: normalization.pt not found, using identity normalization")

        model.to(self.device)
        model.eval()
        print("Model loaded successfully")
        return model, config

    def read_data(self, atm_file: Optional[str], ocn_file: str) -> Dict[str, np.ndarray]:
        builder = UFSEmulatorDataBuilder(self.config)
        return builder.read_netcdf_data_pair(atm_file, ocn_file)

    def filter_domain(
        self,
        data: Dict[str, np.ndarray],
        domain: str = "arctic",
        min_ice: Optional[float] = None,
        mask_mode: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if min_ice is None:
            min_ice = float(self.config.get("domain", {}).get("min_ice_concentration", 0.0))
        if mask_mode is None:
            mask_mode = str(self.config.get("domain", {}).get("mask_mode", "both"))

        lats = data["lat"].astype(np.float32)
        lons = data["lon"].astype(np.float32)
        mask = data["mask"] == 1

        if "aice" not in data:
            raise ValueError("aice is required for mask-mode filtering")
        aice = data["aice"].astype(np.float32)

        valid = mask.copy()
        if mask_mode == "sea_ice":
            valid &= aice > min_ice
        elif mask_mode == "ocean":
            valid &= aice < min_ice

        if domain.lower() == "arctic":
            valid &= lats >= 50.0
        elif domain.lower() == "antarctic":
            valid &= lats <= -50.0

        idx = np.where(valid)[0]
        if len(idx) == 0:
            raise ValueError(f"No valid points for domain={domain}, mask_mode={mask_mode}")

        features = []
        for var_name in self.input_variables:
            if var_name not in data:
                raise ValueError(f"Missing required input variable: {var_name}")
            var = data[var_name]
            if var.ndim == 1:
                features.append(var[idx])
            elif var.ndim == 2 and var.shape[1] == 1:
                features.append(var[idx, 0])
            else:
                raise ValueError(
                    f"This plotting script expects surface (1D) inputs. "
                    f"Got {var_name} shape={var.shape}"
                )

        targets = []
        for var_name in self.output_variables:
            if var_name not in data:
                raise ValueError(f"Missing required output variable: {var_name}")
            var = data[var_name]
            if var.ndim == 1:
                targets.append(var[idx])
            elif var.ndim == 2 and var.shape[1] == 1:
                targets.append(var[idx, 0])
            else:
                raise ValueError(
                    f"This plotting script expects surface (1D) outputs. "
                    f"Got {var_name} shape={var.shape}"
                )

        x = np.column_stack(features).astype(np.float32)
        y = np.column_stack(targets).astype(np.float32)
        return x, y, lons[idx], lats[idx]

    def run_inference(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = torch.from_numpy(features).float().to(self.device)
        with torch.no_grad():
            y = self.model.predict(x).cpu().numpy()

        jac_rows = []
        for i in range(x.shape[0]):
            sample = x[i : i + 1]
            jac = self.model._jac_physical(sample).detach().cpu().numpy()[0]
            jac_rows.append(jac)

        jac_all = np.stack(jac_rows, axis=0)
        return y, jac_all

    def _get_projection(self, domain: str, global_plot: bool):
        import cartopy.crs as ccrs

        if domain == "global" or global_plot:
            return ccrs.PlateCarree(), [-180, 180, -90, 90]
        if domain == "arctic":
            return ccrs.NorthPolarStereo(), [-180, 180, 50, 90]
        return ccrs.SouthPolarStereo(), [-180, 180, -90, -50]

    def _plot_field(
        self,
        ax,
        lons: np.ndarray,
        lats: np.ndarray,
        values: np.ndarray,
        title: str,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = "viridis",
        cbar_label: str = "",
    ):
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.pyplot as plt

        ax.add_feature(cfeature.COASTLINE, alpha=0.5)
        ax.add_feature(cfeature.LAND, alpha=0.3, color="lightgray")
        ax.gridlines(draw_labels=False, alpha=0.3)

        sc = ax.scatter(
            lons,
            lats,
            c=values,
            s=0.8,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
        )
        ax.set_title(title, fontsize=10)
        plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.03, label=cbar_label)

    def plot_domain_fields(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        predictions: np.ndarray,
        jacobians: np.ndarray,
        lons: np.ndarray,
        lats: np.ndarray,
        domain: str,
        output_dir: str,
        global_plot: bool,
        jacobian_vmin: Optional[float],
        jacobian_vmax: Optional[float],
    ) -> None:
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        projection, extent = self._get_projection(domain.lower(), global_plot)
        out_name = self.output_variables[0]

        target_main = targets[:, 0]
        pred_main = predictions[:, 0]

        if out_name == "aice":
            vmin, vmax = 0.0, 1.0
            cmap = "Blues"
        else:
            vmin = float(np.nanmin(target_main))
            vmax = float(np.nanmax(target_main))
            cmap = "RdBu_r"

        fig = plt.figure(figsize=(16, 7))
        ax1 = plt.subplot(1, 2, 1, projection=projection)
        ax2 = plt.subplot(1, 2, 2, projection=projection)

        ax1.set_extent(extent, crs=ccrs.PlateCarree())
        ax2.set_extent(extent, crs=ccrs.PlateCarree())

        self._plot_field(
            ax1,
            lons,
            lats,
            target_main,
            f"Target {out_name}",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            cbar_label=self.output_units.get(out_name, ""),
        )
        self._plot_field(
            ax2,
            lons,
            lats,
            pred_main,
            f"Predicted {out_name}",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            cbar_label=self.output_units.get(out_name, ""),
        )

        rmse = float(np.sqrt(np.mean((pred_main - target_main) ** 2)))
        corr = float(np.corrcoef(pred_main, target_main)[0, 1])
        fig.suptitle(f"{domain.title()} {out_name} | RMSE={rmse:.4f}, r={corr:.4f}")
        fig.tight_layout()

        field_path = out / f"inference_{out_name}_{domain.lower()}.png"
        fig.savefig(field_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {field_path}")

        # Jacobian maps for first output component.
        jac = jacobians[:, 0, :]
        for i, var_name in enumerate(self.input_variables):
            sens = jac[:, i]
            if jacobian_vmin is None or jacobian_vmax is None:
                std = float(np.nanstd(sens))
                if std > 0:
                    jvmin, jvmax = -3.0 * std, 3.0 * std
                else:
                    jvmin, jvmax = float(np.nanmin(sens)), float(np.nanmax(sens))
            else:
                jvmin, jvmax = jacobian_vmin, jacobian_vmax

            fig_j = plt.figure(figsize=(10, 8))
            ax = plt.subplot(1, 1, 1, projection=projection)
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            self._plot_field(
                ax,
                lons,
                lats,
                sens,
                f"d{out_name}/d{var_name}",
                vmin=jvmin,
                vmax=jvmax,
                cmap="RdBu_r",
                cbar_label=f"d{out_name}/d{var_name}",
            )
            fig_j.tight_layout()
            jac_path = out / f"jacobian_d{out_name}_d{var_name}_{domain.lower()}.png"
            fig_j.savefig(jac_path, dpi=150, bbox_inches="tight")
            plt.close(fig_j)
            print(f"Saved: {jac_path}")

    def create_summary_statistics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        jacobians: np.ndarray,
        features: np.ndarray,
        domain: str,
    ) -> None:
        y = targets[:, 0]
        p = predictions[:, 0]

        rmse = float(np.sqrt(np.mean((p - y) ** 2)))
        mae = float(np.mean(np.abs(p - y)))
        corr = float(np.corrcoef(p, y)[0, 1])

        print(f"\n{domain.title()} statistics")
        print("-" * 40)
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE:  {mae:.6f}")
        print(f"Corr: {corr:.6f}")

        jac = jacobians[:, 0, :]
        for i, name in enumerate(self.input_variables):
            vals = jac[:, i]
            print(
                f"d{self.output_variables[0]}/d{name}: "
                f"min={np.min(vals):.3e}, max={np.max(vals):.3e}, "
                f"mean={np.mean(vals):.3e}, std={np.std(vals):.3e}"
            )
            fvals = features[:, i]
            unit = self.input_units.get(name, "")
            print(
                f"  {name}: min={np.min(fvals):.3f}, max={np.max(fvals):.3f}, "
                f"mean={np.mean(fvals):.3f}, std={np.std(fvals):.3f} {unit}"
            )


def thin_data(
    inputs: np.ndarray,
    targets: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    fraction: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if fraction >= 1.0:
        return inputs, targets, lons, lats
    n = len(targets)
    m = max(1, int(n * fraction))
    idx = np.random.choice(n, m, replace=False)
    return inputs[idx], targets[idx], lons[idx], lats[idx]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inference and Jacobian visualization for saber-pytorch surface models"
    )
    parser.add_argument("--ocn-file", required=True, help="Ocean/ice NetCDF file")
    parser.add_argument("--atm-file", default=None, help="Atmosphere NetCDF file (optional)")
    parser.add_argument("--model", required=True, help="Trained checkpoint (.pt)")
    parser.add_argument("--config", default=None, help="Optional config YAML override")
    parser.add_argument("--output-dir", default="plots", help="Output directory")
    parser.add_argument("--arctic-only", action="store_true")
    parser.add_argument("--antarctic-only", action="store_true")
    parser.add_argument("--global-only", action="store_true")
    parser.add_argument("--global-plot", action="store_true")
    parser.add_argument("--thin-fraction", type=float, default=1.0)
    parser.add_argument("--jacobian-vmin", type=float, default=None)
    parser.add_argument("--jacobian-vmax", type=float, default=None)
    args = parser.parse_args()

    try:
        import cartopy  # noqa: F401
        import matplotlib.pyplot  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("cartopy and matplotlib are required for plotting") from exc

    plotter = SaberInferencePlotter(args.model, args.config)
    data = plotter.read_data(args.atm_file, args.ocn_file)

    if args.arctic_only:
        domains = ["arctic"]
    elif args.antarctic_only:
        domains = ["antarctic"]
    elif args.global_only or args.global_plot:
        domains = ["global"]
    else:
        domains = ["arctic", "antarctic"]

    for domain in domains:
        print(f"\nProcessing domain: {domain}")
        try:
            features, targets, lons, lats = plotter.filter_domain(data, domain)
            if args.thin_fraction < 1.0:
                features, targets, lons, lats = thin_data(
                    features, targets, lons, lats, args.thin_fraction
                )
            preds, jacs = plotter.run_inference(features)
            plotter.plot_domain_fields(
                features,
                targets,
                preds,
                jacs,
                lons,
                lats,
                domain,
                args.output_dir,
                args.global_plot,
                args.jacobian_vmin,
                args.jacobian_vmax,
            )
            plotter.create_summary_statistics(preds, targets, jacs, features, domain)
        except Exception as exc:
            print(f"Failed for domain={domain}: {exc}")

    print(f"\nDone. Plots written under: {args.output_dir}")


if __name__ == "__main__":
    main()
