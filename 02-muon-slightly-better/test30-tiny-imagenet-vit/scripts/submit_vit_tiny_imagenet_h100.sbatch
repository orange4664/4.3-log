#!/usr/bin/env bash
#SBATCH -J test30_tiny
#SBATCH -p gpu_h100
#SBATCH --gpus=1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

ROOT=${ROOT:?Set ROOT=/data/run01/scwb923/4.3-log/test30-tiny-imagenet-vit}
HAMGNN_ENV=${HAMGNN_ENV:-/data/home/scwb923/run/envs/hamgnn}
HAMGNN_REPO=${HAMGNN_REPO:-/data/home/scwb923/HamGNN}
ADAPTER_REPO=${ADAPTER_REPO:-/data/run01/scwb923/4.3-log/test1-hamgnn-si/code}
PYTHON_BIN=${PYTHON_BIN:-$HAMGNN_ENV/bin/python}
DATA_DIR=${DATA_DIR:-${ROOT}/data}

source /usr/share/modules/init/bash >/dev/null 2>&1 || true
module load cuda/12.4 || true

export PYTHONPATH="${ADAPTER_REPO}:${HAMGNN_REPO}:${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export WANDB_MODE=offline

mkdir -p "${ROOT}/logs" "${ROOT}/runs"
cd "${ROOT}"

bash "${ROOT}/scripts/prepare_tiny_imagenet.sh" "${ROOT}"

"${PYTHON_BIN}" -u "${ROOT}/train_vit_tiny_imagenet.py" \
  --data-dir "${DATA_DIR}" \
  --outdir "${ROOT}/runs" \
  --patch-size 4 \
  --embed-dim 256 \
  --depth 8 \
  --num-heads 8 \
  > "${ROOT}/logs/test30_tiny_${SLURM_JOB_ID}.train.log" 2>&1
