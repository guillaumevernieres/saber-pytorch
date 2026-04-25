#!/usr/bin/env bash
set -euo pipefail

CONFIG="emulators/salinity_profile/geom100.yaml"
OCN_FILE="../i-jedi/test-soca/geom100/MOM.res.nc"
DATA_FILE="data/salt_profile_geom100_training.npz"
CHECKPOINT="models_salt_profile_geom100/best_model.pt"
OUTPUT_TS="vertical_ml_balance_salt_profile_geom100.ts"

python scripts/prepare_training_data.py \
  --config "${CONFIG}" \
  --ocn "${OCN_FILE}" \
  --output "${DATA_FILE}"

python scripts/train_ml_balance.py \
  --config "${CONFIG}" \
  --data-path "${DATA_FILE}"

python scripts/build_vertical_ml_balance_emulator.py \
  --checkpoint "${CHECKPOINT}" \
  --output "${OUTPUT_TS}"
