#!/usr/bin/env bash
#SBATCH --account=da-cpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --time=01:00:00

# Usage:
#   sbatch hpc/train_distributed.sh [--config configs/aice.yaml]
#
# The config can be overridden by setting SABER_CONFIG before submitting:
#   SABER_CONFIG=configs/both_domain.yaml sbatch hpc/train_distributed.sh

export OMP_NUM_THREADS=80
export MASTER_PORT=29500

# Activate the saber-pytorch virtual environment
source ~/venvs/torch-env/bin/activate

CONFIG=${SABER_CONFIG:-configs/aice.yaml}

stdbuf -oL -eL srun python \
    "$(dirname "$0")/../scripts/train_ml_balance.py" \
    --config "${CONFIG}" \
    "$@"
