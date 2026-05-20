#!/usr/bin/env bash
#SBATCH -J test14_gpts
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

METHOD=${METHOD:?Set METHOD=adamw|muon_ns}
ROOT=${ROOT:?Set ROOT=/data/run01/scwb923/4.3-log/test14-gpt2-tinystories}
HAMGNN_ENV=${HAMGNN_ENV:-/data/home/scwb923/run/envs/hamgnn}
HAMGNN_REPO=${HAMGNN_REPO:-/data/home/scwb923/HamGNN}
ADAPTER_REPO=${ADAPTER_REPO:-/data/run01/scwb923/4.3-log/test1-hamgnn-si/code}
PYTHON_BIN=${PYTHON_BIN:-$HAMGNN_ENV/bin/python}
TRAIN_TEXT_FILE=${TRAIN_TEXT_FILE:-/data/run01/scwb923/hyz/pythia_optimizer_benchmark/data/train.txt}
VAL_TEXT_FILE=${VAL_TEXT_FILE:-/data/run01/scwb923/hyz/pythia_optimizer_benchmark/data/val.txt}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
TOKENIZER_PATH=${TOKENIZER_PATH:-/data/run01/scwb923/4.3-log/test10-pythia-tinystories/offline_assets/EleutherAI_pythia-160m}

source /usr/share/modules/init/bash >/dev/null 2>&1 || true
module load miniforge3/24.11 || true
module load cuda/12.4 || true

export PYTHONPATH="${ADAPTER_REPO}:${HAMGNN_REPO}:${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export WANDB_MODE=offline

mkdir -p "${ROOT}/logs" "${ROOT}/runs_smoke" "${ROOT}/diag"
cd "${ROOT}"

RUN_NAME="test14_gpt2_${METHOD}_${SLURM_JOB_ID}"
OUT_DIR="${ROOT}/runs_smoke/${METHOD}"
DIAG_DIR="${ROOT}/diag/${RUN_NAME}"
mkdir -p "${OUT_DIR}" "${DIAG_DIR}"

"${PYTHON_BIN}" -u "${ROOT}/train_gpt2_tinystories.py" \
  --optimizer "${METHOD}" \
  --out-dir "${OUT_DIR}" \
  --diag-dir "${DIAG_DIR}" \
  --train-text-file "${TRAIN_TEXT_FILE}" \
  --val-text-file "${VAL_TEXT_FILE}" \
  --tokenizer-name-or-path "${TOKENIZER_PATH}" \
  --learning-rate "${LEARNING_RATE}" \
  --max-steps 300 \
  --eval-interval 100 \
  --eval-batches 16 \
  --train-batch-size 4 \
  --eval-batch-size 4 \
  --grad-accum-steps 4 \
  --n-embd 512 \
  --n-layer 8 \
  --n-head 8 \
  > "${DIAG_DIR}/train.log" 2>&1
