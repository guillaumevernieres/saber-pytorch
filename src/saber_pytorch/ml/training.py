"""Training loop for the ML balance FFNN emulator.

Supports single-node and multi-node distributed training (DDP via SLURM
or torchrun), Exponential Moving Average (EMA) weight tracking, Jacobian
convergence stopping, and an 8-panel training-history plot.

Adapted from aibalance/ufsemulator/training.py with the following changes:
- Uses saber_pytorch.ffnn.FFNN instead of UfsEmulatorFFNN.
- No conv1d support (use the plain FFNN).
- Normalization is applied to both inputs and targets before training so that
  FFNN.forward() (which operates in normalized space) is used correctly.

Quick-start:
    python scripts/train_ml_balance.py --config emulators/ml_aice/config.yaml

Distributed (SLURM):
    sbatch hpc/train_distributed.sh
"""

import argparse
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data.distributed import DistributedSampler

from .data import create_training_data_from_netcdf
from .ffnn import FFNN
from .losses import build_loss_terms


# ---------------------------------------------------------------------------
# Exponential Moving Average
# ---------------------------------------------------------------------------

class ExponentialMovingAverage:
    """EMA shadow weights for better generalization.

    Enable via config:
        training:
            use_ema: true
            ema_decay: 0.999
    """

    def __init__(self, model: nn.Module, decay: float = 0.999,
                 device: Optional[torch.device] = None) -> None:
        self.decay = decay
        self.device = device if device is not None else next(model.parameters()).device
        self.shadow_params: Dict[str, torch.Tensor] = {}
        self.backup_params: Dict[str, torch.Tensor] = {}
        actual = model.module if isinstance(model, DDP) else model
        for name, param in actual.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone().to(self.device)

    def update(self, model: nn.Module) -> None:
        actual = model.module if isinstance(model, DDP) else model
        with torch.no_grad():
            for name, param in actual.named_parameters():
                if param.requires_grad and name in self.shadow_params:
                    self.shadow_params[name].mul_(self.decay).add_(
                        param.data.to(self.device), alpha=1.0 - self.decay
                    )

    def apply_shadow(self, model: nn.Module) -> None:
        actual = model.module if isinstance(model, DDP) else model
        for name, param in actual.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                self.backup_params[name] = param.data.clone()
                param.data.copy_(self.shadow_params[name].to(param.device))

    def restore(self, model: nn.Module) -> None:
        actual = model.module if isinstance(model, DDP) else model
        for name, param in actual.named_parameters():
            if param.requires_grad and name in self.backup_params:
                param.data.copy_(self.backup_params[name])
        self.backup_params.clear()

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "shadow_params": self.shadow_params}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.decay = state.get("decay", self.decay)
        self.shadow_params = state["shadow_params"]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class MLBalanceTrainer:
    """Training wrapper for FFNN-based ML balance emulators.

    Supports single-node (rank=0, world_size=1) and distributed DDP runs.
    """

    def __init__(self, config: Dict, rank: int = 0,
                 world_size: int = 1) -> None:
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = world_size > 1

        # CPU threading — only matters for non-distributed CPU runs
        if not self.is_distributed and not torch.cuda.is_available():
            num_threads = config.get("num_threads") or int(
                os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1)
            )
            torch.set_num_threads(num_threads)
            if rank == 0:
                print(f"PyTorch CPU threads: {torch.get_num_threads()}")

        # Device
        if self.is_distributed:
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
            if torch.cuda.is_available():
                local_rank = local_rank % torch.cuda.device_count()
                self.device = torch.device(f"cuda:{local_rank}")
                torch.cuda.set_device(local_rank)
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available() and config.get("use_cuda", True)
                else "cpu"
            )

        if rank == 0:
            print(f"Device: {self.device}")
            if self.is_distributed:
                print(f"Distributed: {world_size} processes")

        input_size, output_size = self._get_model_dimensions()
        model_cfg = config["model"]

        base_model = FFNN(
            input_size=input_size,
            output_size=output_size,
            hidden_size=model_cfg["hidden_size"],
            hidden_layers=model_cfg.get("hidden_layers", 2),
            activation=model_cfg.get("activation", "gelu"),
            use_conv1d=model_cfg.get("use_conv1d", False),
            conv_channels=model_cfg.get("conv_channels", 32),
            conv_kernel_size=model_cfg.get("conv_kernel_size", 3),
        ).to(self.device)
        base_model.init_weights()

        if self.is_distributed:
            device_ids = [rank] if torch.cuda.is_available() else None
            self.model: nn.Module = DDP(base_model, device_ids=device_ids)
        else:
            self.model = base_model

        self.optimizer = self._create_optimizer()
        self.criterion = self._create_loss_function()
        self.loss_terms = build_loss_terms(config)
        self.scheduler = self._create_scheduler()

        self.ema: Optional[ExponentialMovingAverage] = None
        if config["training"].get("use_ema", False):
            ema_decay = config["training"].get("ema_decay", 0.999)
            self.ema = ExponentialMovingAverage(
                self.model, decay=ema_decay, device=self.device
            )
            if rank == 0:
                print(f"EMA enabled (decay={ema_decay})")

        self.history: Dict[str, List] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "jacobian_frobenius_norm": [],
            "jacobian_spectral_norm": [],
            "jacobian_stability": [],
            "jacobian_metrics_history": [],
        }

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _get_model_dimensions(self) -> Tuple[int, int]:
        """Determine input/output sizes.

        Priority: (1) explicit model.input_size/output_size in config,
                  (2) .npz metadata, (3) variable count.
        """
        model_cfg = self.config.get("model", {})
        cfg_in = model_cfg.get("input_size")
        cfg_out = model_cfg.get("output_size")
        if cfg_in is not None and cfg_out is not None:
            if self.rank == 0:
                print(f"Model dimensions from config: {cfg_in}→{cfg_out}")
            return int(cfg_in), int(cfg_out)

        data_path = self.config.get("data", {}).get("data_path")
        if data_path:
            npz_path = Path(data_path).with_suffix(".npz")
            if npz_path.exists():
                try:
                    raw = np.load(str(npz_path), allow_pickle=True)
                    meta = raw.get("metadata")
                    if meta is not None:
                        md = meta.item() if hasattr(meta, "item") else meta
                        in_sz = md.get("input_size")
                        out_sz = md.get("output_size")
                        if in_sz is not None and out_sz is not None:
                            if self.rank == 0:
                                print(f"Model dimensions from .npz: {in_sz}→{out_sz}")
                                print(f"  Inputs : {md.get('input_features', [])}")
                                print(f"  Outputs: {md.get('output_features', [])}")
                            self.config["model"]["input_size"] = in_sz
                            self.config["model"]["output_size"] = out_sz
                            return int(in_sz), int(out_sz)
                except Exception as exc:
                    if self.rank == 0:
                        print(f"Could not read .npz metadata: {exc}")

        vcfg = self.config.get("variables", {})
        input_vars: List[str] = vcfg.get(
            "input_variables",
            ["sst", "sss", "tair", "tsfc", "hi", "hs", "sice",
             "uocn", "vocn", "uatm", "vatm", "qref", "flwdn", "fswdn"],
        )
        max_feats = vcfg.get("max_input_features")
        if max_feats and max_feats > 0:
            input_vars = input_vars[:max_feats]
        output_vars: List[str] = vcfg.get("output_variables", ["aice"])

        if model_cfg.get("emulator_type") == "salinity_profile":
            target_levels = vcfg.get("target_num_levels")
            if target_levels is None:
                raise ValueError("salinity_profile requires variables.target_num_levels")
            input_size = len(input_vars) * int(target_levels)
            output_size = int(target_levels)
        else:
            input_size = len(input_vars)
            output_size = len(output_vars)
        if self.rank == 0:
            print(f"Model: {input_size} inputs → {output_size} outputs")
            print(f"  Inputs : {input_vars}")
            print(f"  Outputs: {output_vars}")
        self.config["model"]["input_size"] = input_size
        self.config["model"]["output_size"] = output_size
        return input_size, output_size

    def _create_optimizer(self) -> optim.Optimizer:
        opt = self.config["training"]["optimizer"]
        lr = opt["learning_rate"]
        wd = opt.get("weight_decay", 0.0)
        if opt["type"] == "adam":
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        if opt["type"] == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=opt.get("momentum", 0.9),
                weight_decay=wd,
            )
        raise ValueError(f"Unknown optimizer: {opt['type']}")

    def _create_loss_function(self) -> nn.Module:
        loss_type = self.config["training"].get("loss_function", "mse")
        if loss_type == "mse":
            return nn.MSELoss()
        if loss_type == "mae":
            return nn.L1Loss()
        if loss_type == "huber":
            return nn.SmoothL1Loss()
        raise ValueError(f"Unknown loss function: {loss_type}")

    def _create_scheduler(self) -> Any:
        scfg = self.config["training"].get("scheduler")
        if scfg is None:
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10000, gamma=1.0
            )
        if scfg["type"] == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=scfg["step_size"], gamma=scfg["gamma"]
            )
        if scfg["type"] == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config["training"]["epochs"]
            )
        if scfg["type"] == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=scfg.get("factor", 0.5),
                patience=scfg.get("patience", 10),
            )
        raise ValueError(f"Unknown scheduler: {scfg['type']}")

    def _bare_model(self) -> FFNN:
        """Return the underlying FFNN, unwrapping DDP if needed."""
        if isinstance(self.model, DDP):
            return self.model.module  # type: ignore[return-value]
        return self.model  # type: ignore[return-value]

    def _compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.criterion(predictions, targets)
        bare_model = self._bare_model()
        for term in self.loss_terms:
            loss = loss + term(predictions, targets, bare_model)
        return loss

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self, data_path: str) -> Tuple[DataLoader, DataLoader]:
        """Load training data, set model normalization, return data loaders.

        Rank-0 performs NetCDF conversion; other ranks wait at a barrier.
        Both input and output normalization stats are stored in the model.
        Training is performed in normalized space (FFNN.forward contract).
        """
        if data_path.endswith(".nc"):
            if self.rank == 0:
                print("Converting NetCDF to training format...")
                processed = str(Path(data_path).with_suffix(".npz"))
                create_training_data_from_netcdf(data_path, self.config, processed)
            if self.is_distributed:
                dist.barrier()
            data_path = str(Path(data_path).with_suffix(".npz"))

        if data_path.endswith(".npz"):
            raw = np.load(data_path, allow_pickle=True)
            inputs = torch.FloatTensor(raw["inputs"])
            targets = torch.FloatTensor(raw["targets"])
            input_mean = torch.tensor(raw["input_mean"], dtype=torch.float32)
            input_std = torch.tensor(raw["input_std"], dtype=torch.float32)
            output_mean = torch.tensor(raw["output_mean"], dtype=torch.float32)
            output_std = torch.tensor(raw["output_std"], dtype=torch.float32)
        elif data_path.endswith(".pt"):
            raw = torch.load(data_path, weights_only=False)
            inputs = raw["inputs"].float()
            targets = raw["targets"].float()
            input_mean = raw["input_mean"].float()
            input_std = raw["input_std"].float()
            output_mean = raw.get("output_mean",
                                  torch.zeros(targets.shape[1])).float()
            output_std = raw.get("output_std",
                                 torch.ones(targets.shape[1])).float()
        else:
            raise ValueError(f"Unsupported data format: {data_path}")

        input_std = torch.where(input_std > 1e-6, input_std,
                                torch.ones_like(input_std))
        output_std = torch.where(output_std > 1e-6, output_std,
                                 torch.ones_like(output_std))

        self._bare_model().init_norm(input_mean, input_std, output_mean, output_std)

        if self.rank == 0:
            output_dir = Path(self.config["output"]["model_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            self._bare_model().save_norm(str(output_dir / "normalization.pt"))

        inputs_norm = (inputs - input_mean) / input_std
        targets_norm = (targets - output_mean) / output_std

        dataset = TensorDataset(inputs_norm, targets_norm)
        val_frac = self.config["data"]["validation_split"]
        val_size = int(len(dataset) * val_frac)
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        batch = self.config["training"]["batch_size"]
        nw = self.config["data"].get("num_workers", 0)

        if self.is_distributed:
            train_sampler: Optional[DistributedSampler] = DistributedSampler(
                train_ds, num_replicas=self.world_size, rank=self.rank
            )
            val_sampler: Optional[DistributedSampler] = DistributedSampler(
                val_ds, num_replicas=self.world_size, rank=self.rank
            )
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        train_loader = DataLoader(
            train_ds, batch_size=batch, shuffle=shuffle,
            sampler=train_sampler, num_workers=nw
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch, shuffle=False,
            sampler=val_sampler, num_workers=nw
        )

        if self.rank == 0:
            print(f"Training samples  : {train_size}")
            print(f"Validation samples: {val_size}")
        return train_loader, val_loader

    # ------------------------------------------------------------------
    # Training / validation steps
    # ------------------------------------------------------------------

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(x)
            if out.dim() > y.dim():
                out = out.squeeze()
            loss = self._compute_loss(out, y)
            loss.backward()
            self.optimizer.step()
            if self.ema is not None:
                self.ema.update(self.model)
            total += loss.item()
            n += 1
        return total / n

    def validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        if self.ema is not None:
            self.ema.apply_shadow(self.model)
        total = 0.0
        n = 0
        try:
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out = self.model(x)
                    if out.dim() > y.dim():
                        out = out.squeeze()
                    total += self._compute_loss(out, y).item()
                    n += 1
        finally:
            if self.ema is not None:
                self.ema.restore(self.model)
        return total / n

    def compute_jacobian_metrics(
        self, val_loader: DataLoader
    ) -> Dict[str, Any]:
        """Jacobian metrics on up to 100 validation samples."""
        self.model.eval()
        if self.ema is not None:
            self.ema.apply_shadow(self.model)
        sample_size = min(100, len(val_loader.dataset))
        rows: List[np.ndarray] = []
        try:
            with torch.enable_grad():
                for i, (x, _) in enumerate(val_loader):
                    if i * val_loader.batch_size >= sample_size:
                        break
                    x = x.to(self.device)
                    take = min(x.shape[0],
                               sample_size - i * val_loader.batch_size)
                    for j in range(take):
                        xi = x[j: j + 1].requires_grad_(True)
                        out = self.model(xi)
                        g = torch.ones_like(out)
                        row = torch.autograd.grad(
                            outputs=out, inputs=xi, grad_outputs=g,
                            create_graph=False, retain_graph=False,
                        )[0]
                        rows.append(row.detach().cpu().numpy().flatten())
        finally:
            if self.ema is not None:
                self.ema.restore(self.model)

        if not rows:
            return {"frobenius_norm": 0.0, "spectral_norm": 0.0,
                    "stability": 0.0, "max_gradient": 0.0,
                    "gradient_std": 0.0, "max_feature_sensitivity": 0.0,
                    "most_sensitive_feature": 0}

        J = np.vstack(rows)
        frob = float(np.linalg.norm(J, "fro"))
        mean_abs = float(np.mean(np.abs(J)))
        max_abs = float(np.max(np.abs(J)))
        grad_std = float(np.std(J))
        feat_sens = np.mean(np.abs(J), axis=0)
        top_feat = int(np.argmax(feat_sens))
        max_feat_sens = float(np.max(feat_sens))
        try:
            s = np.linalg.svd(J, compute_uv=False)
            spectral = float(s[0]) if len(s) > 0 else 0.0
            stability = float(s[0] / s[-1]) if len(s) > 0 and s[-1] > 1e-12 else 1e6
        except np.linalg.LinAlgError:
            spectral = 0.0
            stability = 1e6

        return {
            "frobenius_norm": frob,
            "spectral_norm": mean_abs,   # alias kept for history compatibility
            "stability": stability,
            "max_gradient": max_abs,
            "gradient_std": grad_std,
            "feature_sensitivity": feat_sens,
            "most_sensitive_feature": top_feat,
            "max_feature_sensitivity": max_feat_sens,
        }

    def _get_prediction_sample(
        self, val_loader: DataLoader, max_samples: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        if self.ema is not None:
            self.ema.apply_shadow(self.model)
        preds: List[float] = []
        tgts: List[float] = []
        try:
            with torch.no_grad():
                for x, y in val_loader:
                    if len(preds) >= max_samples:
                        break
                    x, y = x.to(self.device), y.to(self.device)
                    out = self.model(x)
                    if out.dim() > y.dim():
                        out = out.squeeze()
                    preds.extend(out.cpu().numpy().flatten())
                    tgts.extend(y.cpu().numpy().flatten())
        finally:
            if self.ema is not None:
                self.ema.restore(self.model)
        return np.array(preds[:max_samples]), np.array(tgts[:max_samples])

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        if self.rank == 0:
            print("Starting training...")
        t0 = time.time()

        best_val = float("inf")
        patience_counter = 0
        patience = self.config["training"].get("early_stopping_patience", 50)
        track_jac = self.config["training"].get("track_jacobian", True)
        jac_freq = self.config["training"].get("jacobian_freq", 5)
        save_interval = self.config["training"].get("save_interval", 20)
        n_epochs = self.config["training"]["epochs"]

        conv_tol = self.config["training"].get("convergence_tolerance", 1e-6)
        conv_win = self.config["training"].get("convergence_window", 5)
        min_epochs = self.config["training"].get("min_epochs", 10)

        jac_conv_tol = self.config["training"].get(
            "jacobian_convergence_tolerance", 1e-4)
        jac_conv_win = self.config["training"].get(
            "jacobian_convergence_window", 5)
        min_epochs_jac = self.config["training"].get("min_epochs_jacobian", 20)
        use_jac_stop = self.config["training"].get("use_jacobian_stopping", True)

        recent_losses: List[float] = []
        recent_jac_frob: List[float] = []
        recent_jac_mag: List[float] = []

        for epoch in range(n_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            self.scheduler.step()

            jac_metrics: Optional[Dict[str, Any]] = None
            if track_jac and (epoch + 1) % jac_freq == 0:
                jac_metrics = self.compute_jacobian_metrics(val_loader)
                self.history["jacobian_frobenius_norm"].append(
                    jac_metrics["frobenius_norm"])
                self.history["jacobian_spectral_norm"].append(
                    jac_metrics["spectral_norm"])
                self.history["jacobian_stability"].append(
                    jac_metrics["stability"])
                self.history["jacobian_metrics_history"].append(jac_metrics)

                recent_jac_frob.append(jac_metrics["frobenius_norm"])
                recent_jac_mag.append(jac_metrics["spectral_norm"])
                if len(recent_jac_frob) > jac_conv_win:
                    recent_jac_frob.pop(0)
                if len(recent_jac_mag) > jac_conv_win:
                    recent_jac_mag.pop(0)
            elif track_jac and self.history["jacobian_frobenius_norm"]:
                # Repeat last value so arrays stay epoch-aligned
                self.history["jacobian_frobenius_norm"].append(
                    self.history["jacobian_frobenius_norm"][-1])
                self.history["jacobian_spectral_norm"].append(
                    self.history["jacobian_spectral_norm"][-1])
                self.history["jacobian_stability"].append(
                    self.history["jacobian_stability"][-1])

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["learning_rate"].append(
                self.optimizer.param_groups[0]["lr"])

            recent_losses.append(val_loss)
            if len(recent_losses) > conv_win:
                recent_losses.pop(0)

            # Best-model checkpoint and early stopping counter
            if val_loss < best_val:
                best_val = val_loss
                patience_counter = 0
                if self.rank == 0:
                    self.save_checkpoint("best_model.pt")
            else:
                patience_counter += 1

            # Convergence checks
            loss_converged = False
            if epoch + 1 >= min_epochs and len(recent_losses) == conv_win:
                hi, lo = max(recent_losses), min(recent_losses)
                if hi > 0 and (hi - lo) / hi < conv_tol:
                    loss_converged = True
                    if self.rank == 0:
                        print(f"Loss plateau at epoch {epoch+1}")

            jac_converged = False
            if (use_jac_stop and track_jac and epoch + 1 >= min_epochs_jac):
                if len(recent_jac_mag) >= jac_conv_win:
                    hi, lo = max(recent_jac_mag), min(recent_jac_mag)
                    if hi > 0 and (hi - lo) / hi < jac_conv_tol:
                        jac_converged = True
                        if self.rank == 0:
                            print(f"Jacobian MAG converged at epoch {epoch+1}")
                if not jac_converged and len(recent_jac_frob) >= jac_conv_win:
                    hi, lo = max(recent_jac_frob), min(recent_jac_frob)
                    if hi > 0 and (hi - lo) / hi < jac_conv_tol:
                        jac_converged = True
                        if self.rank == 0:
                            print(f"Jacobian Frobenius converged at epoch {epoch+1}")

            # Progress log (rank 0 only)
            if self.rank == 0 and ((epoch + 1) % 10 == 0 or epoch == 0):
                jac_info = ""
                if jac_metrics:
                    jac_info = f", Jac: {jac_metrics['spectral_norm']:.4f}"
                elif self.history["jacobian_metrics_history"]:
                    jac_info = (
                        f", Jac: "
                        f"{self.history['jacobian_metrics_history'][-1]['spectral_norm']:.4f}"
                    )
                print(
                    f"Epoch {epoch+1}/{n_epochs} — "
                    f"train: {train_loss:.6f}, val: {val_loss:.6f}{jac_info}"
                )

            if patience_counter >= patience:
                if self.rank == 0:
                    print(f"Early stopping after {epoch+1} epochs")
                break
            if loss_converged or jac_converged:
                if self.rank == 0:
                    print(f"Convergence stop after {epoch+1} epochs")
                break

            if self.rank == 0 and (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
                preds, tgts = self._get_prediction_sample(val_loader)
                self.plot_training_history(sample_predictions=preds,
                                           sample_targets=tgts)

        if self.rank == 0:
            print(f"Training completed in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(self, filename: str) -> None:
        output_dir = Path(self.config["output"]["model_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt: Dict[str, Any] = {
            "model_state_dict": self._bare_model().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": self.history,
            "config": self.config,
        }
        if self.ema is not None:
            ckpt["ema_state_dict"] = self.ema.state_dict()
        torch.save(ckpt, output_dir / filename)

    def load_checkpoint(self, path: str) -> None:
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._bare_model().load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if self.ema is not None and "ema_state_dict" in ckpt:
            self.ema.load_state_dict(ckpt["ema_state_dict"])
            if self.rank == 0:
                print("EMA state loaded from checkpoint")
        if "history" in ckpt:
            self.history = ckpt["history"]
        if self.rank == 0:
            print(f"Resumed from epoch {len(self.history['train_loss'])}")

    # ------------------------------------------------------------------
    # 8-panel training history plot
    # ------------------------------------------------------------------

    def plot_training_history(
        self,
        sample_predictions: Optional[np.ndarray] = None,
        sample_targets: Optional[np.ndarray] = None,
    ) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available — skipping plot")
            return

        jh = self.history["jacobian_metrics_history"]
        frob_vals = [m.get("frobenius_norm", 0) for m in jh]
        mag_vals = [m.get("spectral_norm", 0) for m in jh]
        stab_vals = [m.get("stability", 0) for m in jh]
        max_grads = [m.get("max_gradient", 0) for m in jh]
        max_sens = [m.get("max_feature_sensitivity", 0) for m in jh]

        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        use_log = self.config.get("training", {}).get("use_log_scale_loss", True)

        # 1 — Loss
        axes[0, 0].plot(self.history["train_loss"], label="Train", color="blue")
        axes[0, 0].plot(self.history["val_loss"], label="Val", color="red")
        if use_log:
            axes[0, 0].set_yscale("log")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training and Validation Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2 — Learning rate
        axes[0, 1].plot(self.history["learning_rate"], color="green")
        axes[0, 1].set_yscale("log")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Learning Rate")
        axes[0, 1].set_title("Learning Rate Schedule")
        axes[0, 1].grid(True, alpha=0.3)

        # 3 — Jacobian Frobenius norm
        if frob_vals:
            axes[0, 2].plot(frob_vals, color="purple")
        else:
            axes[0, 2].text(0.5, 0.5, "No Jacobian data",
                            ha="center", va="center", transform=axes[0, 2].transAxes)
        axes[0, 2].set_xlabel("Jac checkpoint")
        axes[0, 2].set_ylabel("‖J‖_F")
        axes[0, 2].set_title("Jacobian Frobenius Norm")
        axes[0, 2].grid(True, alpha=0.3)

        # 4 — Mean abs gradient
        if mag_vals:
            axes[0, 3].plot(mag_vals, color="orange")
        else:
            axes[0, 3].text(0.5, 0.5, "No Jacobian data",
                            ha="center", va="center", transform=axes[0, 3].transAxes)
        axes[0, 3].set_xlabel("Jac checkpoint")
        axes[0, 3].set_ylabel("Mean |∂y/∂x|")
        axes[0, 3].set_title("Jacobian Mean Abs Gradient")
        axes[0, 3].grid(True, alpha=0.3)

        # 5 — Max gradient
        if max_grads:
            axes[1, 0].plot(max_grads, color="red")
        else:
            axes[1, 0].text(0.5, 0.5, "No Jacobian data",
                            ha="center", va="center", transform=axes[1, 0].transAxes)
        axes[1, 0].set_xlabel("Jac checkpoint")
        axes[1, 0].set_ylabel("Max |J element|")
        axes[1, 0].set_title("Maximum Jacobian Element")
        axes[1, 0].grid(True, alpha=0.3)

        # 6 — Pred vs target scatter
        if sample_predictions is not None and sample_targets is not None:
            mask = ~(np.isnan(sample_predictions) | np.isnan(sample_targets))
            if mask.any():
                p, t = sample_predictions[mask], sample_targets[mask]
                axes[1, 1].scatter(t, p, alpha=0.5, s=1)
                lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
                axes[1, 1].plot([lo, hi], [lo, hi], "r--", alpha=0.8)
                rmse = float(np.sqrt(np.mean((p - t) ** 2)))
                bias = float(np.mean(p - t))
                slope = float(np.polyfit(t, p, 1)[0]) if t.size > 1 else float("nan")
                axes[1, 1].text(
                    0.05, 0.95,
                    f"RMSE: {rmse:.4f}\nBias: {bias:.4f}\nSlope: {slope:.4f}",
                    transform=axes[1, 1].transAxes,
                    va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )
                out_vars = self.config.get("variables", {}).get(
                    "output_variables", ["target"])
                lbl = out_vars[0] if out_vars else "target"
                axes[1, 1].set_xlabel(lbl)
                axes[1, 1].set_ylabel(f"Predicted {lbl}")
            else:
                axes[1, 1].text(0.5, 0.5, "No valid data",
                                ha="center", va="center",
                                transform=axes[1, 1].transAxes)
        else:
            axes[1, 1].text(0.5, 0.5, "No scatter data",
                            ha="center", va="center",
                            transform=axes[1, 1].transAxes)
        axes[1, 1].set_title("Target vs Prediction")
        axes[1, 1].grid(True, alpha=0.3)

        # 7 — Max feature sensitivity
        if max_sens:
            axes[1, 2].plot(max_sens, color="green")
        else:
            axes[1, 2].text(0.5, 0.5, "No Jacobian data",
                            ha="center", va="center", transform=axes[1, 2].transAxes)
        axes[1, 2].set_xlabel("Jac checkpoint")
        axes[1, 2].set_ylabel("Max feature sensitivity")
        axes[1, 2].set_title("Most Significant Jacobian")
        axes[1, 2].grid(True, alpha=0.3)

        # 8 — Stability (condition number)
        if stab_vals:
            axes[1, 3].semilogy(stab_vals, color="brown")
        else:
            axes[1, 3].text(0.5, 0.5, "No Jacobian data",
                            ha="center", va="center", transform=axes[1, 3].transAxes)
        axes[1, 3].set_xlabel("Jac checkpoint")
        axes[1, 3].set_ylabel("Condition number")
        axes[1, 3].set_title("Jacobian Stability")
        axes[1, 3].grid(True, alpha=0.3)

        plt.suptitle("ML Balance Emulator — Training History",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        out = Path(self.config["output"]["model_dir"]) / "training_history.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return cfg


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed(
    rank: Optional[int] = None, world_size: Optional[int] = None
) -> Tuple[int, int]:
    """Initialise the distributed process group (SLURM or torchrun).

    Returns (rank, world_size).
    """
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NPROCS"])
        node_list = os.environ.get("SLURM_NODELIST", "localhost")
        if "[" in node_list:
            base = node_list.split("[")[0]
            ranges = node_list.split("[")[1].split("]")[0]
            first = ranges.split(",")[0]
            num = first.split("-")[0] if "-" in first else first
            master = base + num
        else:
            master = node_list.split(",")[0]
        os.environ["MASTER_ADDR"] = master
        os.environ.setdefault("MASTER_PORT", "29500")
        if rank == 0:
            print(f"SLURM: rank={rank}/{world_size}, master={master}")
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
    else:
        rank = rank if rank is not None else 0
        world_size = world_size if world_size is not None else 1
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")

    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=30),
        )
        if rank == 0:
            print(f"Distributed ({backend}): {world_size} processes, "
                  f"master={os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
    else:
        if rank == 0:
            print("Single-process training")

    return rank, world_size


def train_distributed(
    rank: Optional[int],
    world_size: Optional[int],
    config: Dict,
    data_path: str,
    restart_checkpoint: Optional[str] = None,
    restart_from_best: bool = False,
) -> None:
    rank, world_size = setup_distributed(rank, world_size)
    try:
        trainer = MLBalanceTrainer(config, rank=rank, world_size=world_size)
        if restart_checkpoint:
            trainer.load_checkpoint(restart_checkpoint)
        elif restart_from_best:
            best = Path(config["output"]["model_dir"]) / "best_model.pt"
            trainer.load_checkpoint(str(best))
        train_loader, val_loader = trainer.load_data(data_path)
        trainer.train(train_loader, val_loader)
        if rank == 0:
            preds, tgts = trainer._get_prediction_sample(val_loader)
            trainer.plot_training_history(sample_predictions=preds,
                                          sample_targets=tgts)
            print("Training completed!")
    finally:
        if world_size > 1:
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train ML balance FFNN emulator"
    )
    parser.add_argument("--config", default=None,
                        help="Path to YAML config file")
    parser.add_argument("--data-path", default=None,
                        help="Override data path (.npz, .pt, or .nc)")
    parser.add_argument("--no-distributed", action="store_true",
                        help="Force single-process mode")
    parser.add_argument("--restart-from-best", action="store_true",
                        help="Resume from best_model.pt")
    parser.add_argument("--restart-checkpoint", default=None,
                        help="Resume from specific checkpoint file")
    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
    else:
        raise SystemExit("--config is required")

    if args.data_path:
        config["data"]["data_path"] = args.data_path

    data_file = config["data"]["data_path"]
    if not Path(data_file).exists() and not data_file.endswith(".nc"):
        raise SystemExit(f"Data file not found: {data_file}")

    # Detect distributed environment
    rank: Optional[int] = None
    world_size: Optional[int] = None
    distributed = False

    if not args.no_distributed:
        if "SLURM_PROCID" in os.environ and "SLURM_NPROCS" in os.environ:
            distributed = int(os.environ["SLURM_NPROCS"]) > 1
        elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            distributed = world_size > 1

    if distributed:
        train_distributed(
            rank, world_size, config, data_file,
            restart_checkpoint=args.restart_checkpoint,
            restart_from_best=args.restart_from_best,
        )
    else:
        rank_val, ws_val = setup_distributed(0, 1)
        trainer = MLBalanceTrainer(config, rank=rank_val, world_size=ws_val)
        if args.restart_checkpoint:
            trainer.load_checkpoint(args.restart_checkpoint)
        elif args.restart_from_best:
            best = Path(config["output"]["model_dir"]) / "best_model.pt"
            trainer.load_checkpoint(str(best))
        train_loader, val_loader = trainer.load_data(data_file)
        trainer.train(train_loader, val_loader)
        preds, tgts = trainer._get_prediction_sample(val_loader)
        trainer.plot_training_history(sample_predictions=preds,
                                      sample_targets=tgts)
        out_dir = config["output"]["model_dir"]
        if config.get("variables", {}).get("num_levels"):
            builder = "build_vertical_ml_balance_emulator.py"
            output = "vertical_ml_balance.ts"
        else:
            builder = "build_surface_ml_balance_emulator.py"
            output = "surface_ml_balance.ts"
        print(
            f"\nNext step: build the TorchScript emulator:\n"
            f"  python scripts/{builder} \\\n"
            f"      --checkpoint {out_dir}/best_model.pt \\\n"
            f"      --output {output}"
        )
