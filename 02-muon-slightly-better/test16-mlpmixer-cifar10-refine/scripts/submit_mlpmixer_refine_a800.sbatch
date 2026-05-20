#!/usr/bin/env bash
#SBATCH -J test16_mixr
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

METHOD=${METHOD:?Set METHOD=adamw|muon_ns}
LEARNING_RATE=${LEARNING_RATE:?Set LEARNING_RATE}
ROOT=${ROOT:?Set ROOT=/data/run01/scwb923/4.3-log/test16-mlpmixer-cifar10-refine}
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

mkdir -p "${ROOT}/logs" "${ROOT}/runs"
cd "${ROOT}"

LR_TAG=$(printf "%s" "${LEARNING_RATE}" | tr '+.-' 'p__')

"${PYTHON_BIN}" -u "${ROOT}/train_mlpmixer_cifar10.py" \
  --data-dir "${DATA_DIR}" \
  --outdir "${ROOT}/runs/${METHOD}/lr_${LR_TAG}" \
  --methods "${METHOD}" \
  --lrs "${LEARNING_RATE}" \
  --seeds "0,1,2" \
  --epochs 10 \
  --eval-batches 40 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --log-every 50 \
  --embed-dim 256 \
  --depth 6 \
  --token-dim 128 \
  --channel-dim 512 \
  > "${ROOT}/logs/test16_mlpmixer_${METHOD}_lr${LR_TAG}_${SLURM_JOB_ID}.train.log" 2>&1
