#!/usr/bin/env python3
"""Train an ML balance FFNN emulator.

Delegates to saber_pytorch.training.main(), which supports both single-node
and multi-node distributed training (DDP via SLURM or torchrun).

Usage
-----
    # Single node:
    python scripts/train_ml_balance.py --config emulators/ml_aice/config.yaml

    # Override data path:
    python scripts/train_ml_balance.py --config emulators/ml_aice/config.yaml \\
        --data-path /path/to/data.npz

    # Resume from a checkpoint:
    python scripts/train_ml_balance.py --config emulators/ml_aice/config.yaml \\
        --restart-checkpoint models_aice/checkpoint_epoch_200.pt

    # Distributed (usually invoked by hpc/train_distributed.sh):
    srun python scripts/train_ml_balance.py --config emulators/ml_aice/config.yaml

Workflow summary
----------------
1. Prepare training data (if not done yet):
       python scripts/prepare_training_data.py --config emulators/ml_aice/config.yaml \\
           --atm path/to/atm.nc --ocn path/to/ocn.nc --output data/aice.npz

2. Train:
       python scripts/train_ml_balance.py --config emulators/ml_aice/config.yaml

3. Export to TorchScript for SABER:
       python scripts/build_surface_ml_balance_emulator.py \\
           --checkpoint models_aice/best_model.pt \\
           --output surface_ml_balance_aice.ts

   For profile-to-profile configurations, export with:
       python scripts/build_vertical_ml_balance_emulator.py \\
           --checkpoint models_salt_profile/best_model.pt \\
           --output vertical_ml_balance_salt_profile.ts
"""

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from saber_pytorch.ml.training import main  # noqa: E402

if __name__ == "__main__":
    main()
