#!/usr/bin/env bash
#SBATCH -J test35_5opt
#SBATCH -p gpu_h100
#SBATCH --gpus=1
#SBATCH --exclude=d1n41e21g01
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

ROOT=${ROOT:-/data/run01/scwb923/4.3-log/01-muon-strongly-better/test35-tiny-imagenet-vit-deeper}
HAMGNN_ENV=${HAMGNN_ENV:-/data/home/scwb923/run/envs/hamgnn}
HAMGNN_REPO=${HAMGNN_REPO:-/data/home/scwb923/HamGNN}
ADAPTER_REPO=${ADAPTER_REPO:-/data/run01/scwb923/4.3-log/test1-hamgnn-si/code}
PYTHON_BIN=${PYTHON_BIN:-$HAMGNN_ENV/bin/python}
DATA_DIR=${DATA_DIR:-${ROOT}/data}
OUTDIR=${OUTDIR:-${ROOT}/runs_five_optimizers}

source /usr/share/modules/init/bash >/dev/null 2>&1 || true
module load cuda/12.4 || true

export PYTHONPATH="${ADAPTER_REPO}:${HAMGNN_REPO}:${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export WANDB_MODE=offline

mkdir -p "${ROOT}/logs" "${OUTDIR}"
cd "${ROOT}"

bash "${ROOT}/scripts/prepare_tiny_imagenet.sh" "${ROOT}"

"${PYTHON_BIN}" -u "${ROOT}/train_vit_tiny_imagenet.py" \
  --data-dir "${DATA_DIR}" \
  --outdir "${OUTDIR}" \
  --methods "adamw,muon_ns,muon_adamw,rt_v43_stream,rt_v43_adamw" \
  --lrs "${LRS:-0.0005,0.001}" \
  --seeds "${SEEDS:-0,1,2}" \
  --epochs "${EPOCHS:-6}" \
  --max-steps "${MAX_STEPS:-0}" \
  --batch-size "${BATCH_SIZE:-96}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-192}" \
  --eval-batches "${EVAL_BATCHES:-40}" \
  --log-every "${LOG_EVERY:-100}" \
  --patch-size 4 \
  --embed-dim 320 \
  --depth 10 \
  --num-heads 10 \
  --muon-lr "${MUON_LR:-0.02}" \
  --rt-hybrid-lr "${RT_HYBRID_LR:-0.0032}" \
  --rt-hybrid-adamw-lr "${RT_HYBRID_ADAMW_LR:-0.0032}" \
  > "${ROOT}/logs/test35_5opt_${SLURM_JOB_ID}.train.log" 2>&1

"${PYTHON_BIN}" "${ROOT}/summarize_results.py" \
  --runs-dir "${OUTDIR}" \
  --out-md "${ROOT}/FIVE_OPTIMIZER_BENCHMARK_ANALYSIS.md" \
  --out-csv "${OUTDIR}/five_optimizer_summary.csv"
