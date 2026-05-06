# saber-pytorch

Python package implementing machine learning and physics-based balance operators
for the [SABER](https://github.com/JCSDA/saber) data assimilation framework.
Emulators are exported as TorchScript modules loadable by the SABER C++
`TorchBalance` interface via `torch::jit::load()`.

## Repository layout

```
src/saber_pytorch/       # installable package
  physics/               # analytic balance operators (heave salinity, steric height, ice)
  ml/                    # neural network emulators, training loop, data builder
  observations/          # Argo profile reader

emulators/               # one subdirectory per emulator
  heave_salinity/        # physics: T → S balance (Weaver/Ricci)
  steric_height/         # physics: T, S, h → SSH
  surface_ice_concentration/  # physics: prior state → aice Jacobian
  ml_aice/               # ML: 14 surface vars → sea ice concentration
  ml_salinity/           # ML: T(z), h(z) → S(z) on a reduced vertical grid

scripts/                 # shared CLIs (used by all ML emulators)
  prepare_training_data.py
  train_ml_balance.py
```

## Installation

```bash
pip install -e .
```

## How to: ML emulator (sea ice concentration)

The ML emulators follow a three-step prepare → train → export workflow.

**1. Prepare training data** from model background NetCDF files:

```bash
python scripts/prepare_training_data.py \
    --config emulators/ml_aice/config.yaml \
    --atm /path/to/atm.nc \
    --ocn /path/to/ice_ocean.nc \
    --output data/aice_training.npz
```

**2. Train** the feedforward neural network:

```bash
python scripts/train_ml_balance.py \
    --config emulators/ml_aice/config.yaml \
    --data-path data/aice_training.npz
```

Training history and checkpoints are written to `models_aice/` (configurable via
`output.model_dir` in the config).

**3. Export** to TorchScript for SABER:

```bash
python emulators/ml_aice/scripts/build_surface_ml_balance_emulator.py \
    --checkpoint models_aice/best_model.pt \
    --output surface_ml_balance_aice.ts
```

See `emulators/ml_aice/config.yaml` for the full list of variables and
hyperparameters. The vertical salinity emulator (`emulators/ml_salinity/`)
follows the same three steps; see its `config.yaml` and `scripts/train_mlsalt.sh`
for a self-contained example.

## How to: physics emulator (heave salinity balance)

Physics emulators require no training. Build and export directly:

```bash
python emulators/heave_salinity/scripts/write_heave_salinity_ts.py \
    --config emulators/heave_salinity/config.yaml \
    --output heave_salinity.ts
```

To visualise the Jacobian on a model background:

```bash
python emulators/heave_salinity/scripts/plot_geom100_jacobian.py \
    --config emulators/heave_salinity/config.yaml
```

See `emulators/heave_salinity/README.md` for single-profile and Argo
reconstruction examples.

## Running the tests

```bash
pytest tests/
```
