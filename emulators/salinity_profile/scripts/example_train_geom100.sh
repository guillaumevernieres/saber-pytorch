#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)"
RUN_DIR="$(pwd -P)"

BASE_CONFIG="${REPO_ROOT}/emulators/salinity_profile/config.yaml"
OCN_FILE="${OCN_FILE:-${REPO_ROOT}/../i-jedi/test-soca/geom100/MOM.res.nc}"
DATA_FILE="${DATA_FILE:-data/salt_profile_training.npz}"
MODEL_DIR="${MODEL_DIR:-models_salt_profile}"
OUTPUT_TS="${OUTPUT_TS:-vertical_ml_balance_salt_profile.ts}"
RUN_CONFIG="${RUN_CONFIG:-config.local.yaml}"

resolve_run_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "${RUN_DIR}" "$1" ;;
  esac
}

DATA_FILE="$(resolve_run_path "${DATA_FILE}")"
MODEL_DIR="$(resolve_run_path "${MODEL_DIR}")"
OUTPUT_TS="$(resolve_run_path "${OUTPUT_TS}")"
RUN_CONFIG="$(resolve_run_path "${RUN_CONFIG}")"
CHECKPOINT="${MODEL_DIR}/best_model.pt"

mkdir -p \
  "$(dirname -- "${DATA_FILE}")" \
  "${MODEL_DIR}" \
  "$(dirname -- "${OUTPUT_TS}")" \
  "$(dirname -- "${RUN_CONFIG}")"

python - "${BASE_CONFIG}" "${RUN_CONFIG}" "${DATA_FILE}" "${MODEL_DIR}" <<'PY'
import sys
import yaml

base_config, run_config, data_file, model_dir = sys.argv[1:5]
with open(base_config, "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

config.setdefault("data", {})["data_path"] = data_file
config.setdefault("output", {})["model_dir"] = model_dir

with open(run_config, "w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY

echo "Repository : ${REPO_ROOT}"
echo "Run dir    : ${RUN_DIR}"
echo "Config     : ${RUN_CONFIG}"
echo "Data file  : ${DATA_FILE}"
echo "Model dir  : ${MODEL_DIR}"
echo "TorchScript: ${OUTPUT_TS}"

python "${REPO_ROOT}/scripts/prepare_training_data.py" \
  --config "${RUN_CONFIG}" \
  --ocn "${OCN_FILE}" \
  --output "${DATA_FILE}"

python "${REPO_ROOT}/scripts/train_ml_balance.py" \
  --config "${RUN_CONFIG}" \
  --data-path "${DATA_FILE}"

python "${REPO_ROOT}/scripts/build_vertical_ml_balance_emulator.py" \
  --checkpoint "${CHECKPOINT}" \
  --output "${OUTPUT_TS}"
