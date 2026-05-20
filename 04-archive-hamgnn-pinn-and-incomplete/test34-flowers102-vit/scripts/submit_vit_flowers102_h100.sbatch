#!/usr/bin/env bash
#SBATCH -J test34_flowers102
#SBATCH -p gpu_h100
#SBATCH --gpus=1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

ROOT=${ROOT:?Set ROOT=/data/run01/scwb923/4.3-log/test34-flowers102-vit}
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

mkdir -p "${ROOT}/logs" "${ROOT}/runs_formal"
cd "${ROOT}"

"${PYTHON_BIN}" -u "${ROOT}/train_vit_flowers102.py" \
  --data-dir "${DATA_DIR}" \
  --outdir "${ROOT}/runs_formal" \
  --methods "adamw,muon_ns,rt_v43_stream,rt_v43_ns,rt_v6_fdt_metric" \
  --lrs "0.0003,0.0005" \
  --seeds "0,1,2" \
  --epochs 5 \
  --eval-batches 30 \
  --log-every 25 \
  --batch-size 32 \
  --eval-batch-size 64 \
  > "${ROOT}/logs/test34_flowers102_${SLURM_JOB_ID}.train.log" 2>&1
