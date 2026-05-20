#!/usr/bin/env bash
#SBATCH -J test18_gmlp_s
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

ROOT=${ROOT:?Set ROOT=/data/run01/scwb923/4.3-log/test18-gmlp-cifar10}
HAMGNN_ENV=${HAMGNN_ENV:-/data/home/scwb923/run/envs/hamgnn}
HAMGNN_REPO=${HAMGNN_REPO:-/data/home/scwb923/HamGNN}
ADAPTER_REPO=${ADAPTER_REPO:-/data/run01/scwb923/4.3-log/test1-hamgnn-si/code}
PYTHON_BIN=${PYTHON_BIN:-$HAMGNN_ENV/bin/python}
DATA_DIR=${DATA_DIR:-/data/run01/scwb923/hyz/cifar10_optimizer_benchmark/data}

source /usr/share/modules/init/bash >/dev/null 2>&1 || true
module load cuda/12.4 || true

export PYTHONPATH="${ADAPTER_REPO}:${HAMGNN_REPO}:${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export WANDB_MODE=offline

mkdir -p "${ROOT}/logs" "${ROOT}/runs_smoke"
cd "${ROOT}"

"${PYTHON_BIN}" -u "${ROOT}/train_mlpmixer_cifar10.py" \
  --data-dir "${DATA_DIR}" \
  --outdir "${ROOT}/runs_smoke" \
  --methods "adamw,muon_ns" \
  --lrs "0.0005" \
  --seeds "0" \
  --epochs 2 \
  --max-steps 780 \
  --eval-batches 20 \
  --ff-dim 512 \
  > "${ROOT}/logs/test18_gmlp_smoke_${SLURM_JOB_ID}.train.log" 2>&1
