#!/usr/bin/env bash
#SBATCH -J pythia160_opt
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

METHOD=${METHOD:?Set METHOD=adamw|muon_ns|muon_stream|muon_stream_qqq|rt_v43_stream|rt_v43_ns|rt_v6_fdt_metric}
ROOT=${ROOT:?Set ROOT=/path/to/pythia benchmark root}
HAMGNN_ENV=${HAMGNN_ENV:-/data/home/scwb923/run/envs/hamgnn}
HAMGNN_REPO=${HAMGNN_REPO:-/data/home/scwb923/HamGNN}
ADAPTER_REPO=${ADAPTER_REPO:-/data/home/scwb923/hamgnn_rt_v6_adapter}
PYTHON_BIN=${PYTHON_BIN:-$HAMGNN_ENV/bin/python}
SEQ_LEN=${SEQ_LEN:-256}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-4}
MAX_STEPS=${MAX_STEPS:-600}
EVAL_INTERVAL=${EVAL_INTERVAL:-100}
EVAL_BATCHES=${EVAL_BATCHES:-32}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
WARMUP_STEPS=${WARMUP_STEPS:-50}
MODEL_NAME=${MODEL_NAME:-${ROOT}/offline_assets/EleutherAI_pythia-160m}
DATASET_NAME=${DATASET_NAME:-roneneldan/TinyStories}
TRAIN_TEXT_FILE=${TRAIN_TEXT_FILE:-${ROOT}/data/train.txt}
VAL_TEXT_FILE=${VAL_TEXT_FILE:-${ROOT}/data/val.txt}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-1}
AUTOCAST_DTYPE=${AUTOCAST_DTYPE:-none}
INIT_MODE=${INIT_MODE:-pretrained}
RUN_SUFFIX=${RUN_SUFFIX:-}
SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS:-0}

source /usr/share/modules/init/bash >/dev/null 2>&1 || true
module load miniforge3/24.11 || true
module load cuda/12.4 || true

export PYTHONPATH="${ADAPTER_REPO}:${HAMGNN_REPO}:${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export WANDB_MODE=offline
export PIP_NO_CACHE_DIR=1
export HF_HOME="${ROOT}/hf_cache"
export HF_DATASETS_CACHE="${ROOT}/hf_cache/datasets"
export TRANSFORMERS_CACHE="${ROOT}/hf_cache/transformers"
export TRANSFORMERS_OFFLINE="${LOCAL_FILES_ONLY}"

mkdir -p "${ROOT}/logs" "${ROOT}/runs" "${ROOT}/diag" "${ROOT}/hf_cache"
cd "${ROOT}"

LR_TAG=$(printf "%s" "${LEARNING_RATE}" | tr '+.-' 'p__')
RUN_NAME="pythia160_${METHOD}_lr${LR_TAG}_${SLURM_JOB_ID}"
OUT_DIR="${ROOT}/runs/${METHOD}_lr${LR_TAG}${RUN_SUFFIX}"
DIAG_DIR="${ROOT}/diag/${RUN_NAME}"
mkdir -p "${OUT_DIR}" "${DIAG_DIR}"

"${PYTHON_BIN}" "${ADAPTER_REPO}/scripts/gpu_monitor.py" \
  --out "${DIAG_DIR}/gpu_monitor.csv" --interval 5 &
MON_PID=$!
cleanup() {
  kill "${MON_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "START method=${METHOD} job=${SLURM_JOB_ID} root=${ROOT}"
extra_args=()
if [ -n "${TRAIN_TEXT_FILE}" ] && [ -n "${VAL_TEXT_FILE}" ]; then
  extra_args+=(--train-text-file "${TRAIN_TEXT_FILE}" --val-text-file "${VAL_TEXT_FILE}")
fi
if [ "${LOCAL_FILES_ONLY}" = "1" ] || [ "${LOCAL_FILES_ONLY}" = "true" ]; then
  extra_args+=(--local-files-only)
fi
if [ "${SAVE_CHECKPOINTS}" = "1" ] || [ "${SAVE_CHECKPOINTS}" = "true" ]; then
  extra_args+=(--save-checkpoints)
fi
"${PYTHON_BIN}" -u "${ROOT}/train_pythia_tinystories.py" \
  --model-name "${MODEL_NAME}" \
  --dataset-name "${DATASET_NAME}" \
  --optimizer "${METHOD}" \
  --out-dir "${OUT_DIR}" \
  --diag-dir "${DIAG_DIR}" \
  --seq-len "${SEQ_LEN}" \
  --train-batch-size "${TRAIN_BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --grad-accum-steps "${GRAD_ACCUM}" \
  --max-steps "${MAX_STEPS}" \
  --eval-interval "${EVAL_INTERVAL}" \
  --eval-batches "${EVAL_BATCHES}" \
  --learning-rate "${LEARNING_RATE}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --autocast-dtype "${AUTOCAST_DTYPE}" \
  --init-mode "${INIT_MODE}" \
  "${extra_args[@]}" \
  > "${DIAG_DIR}/train.log" 2>&1

cleanup
trap - EXIT
echo "DONE method=${METHOD} job=${SLURM_JOB_ID} out=${OUT_DIR} diag=${DIAG_DIR}"
