#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH -p gpu_a800
#SBATCH -J test8_hgnn_toy
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

source /usr/share/modules/init/bash >/dev/null 2>&1 || true
module load miniforge3/25.11.0-1 || true
source /data/apps/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate /data/home/scwb923/run/envs/hamgnn

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export WANDB_MODE=offline

ROOT=/data/run01/scwb923/4.3-log/test8-hamgnn-toy-muonns
DATA_DIR=/data/run01/scwb923/hamgnn_workspace/hamgnn_toy_ablation_muonns/data
export TEST8_ROOT="$ROOT"
export PYTHONPATH=/data/run01/scwb923/4.3-log/test1-hamgnn-si/code:/data/home/scwb923/run/HamGNN:${PYTHONPATH:-}

cd "$ROOT"
mkdir -p logs configs diag train

python scripts/make_configs.py \
  --outdir "$ROOT/configs" \
  --data-dir "$DATA_DIR" \
  --train-root "$ROOT/train"

RUN_ID=${SLURM_JOB_ID:-manual}
tail -n +2 configs/manifest.tsv | while IFS=$'\t' read -r METHOD CONFIG; do
  RUN_NAME="toy_${METHOD}_${RUN_ID}"
  DIAG_DIR="$ROOT/diag/$RUN_NAME"
  mkdir -p "$DIAG_DIR"
  echo "LAUNCH method=$METHOD config=$CONFIG diag=$DIAG_DIR"
  python /data/home/scwb923/hamgnn_rt_v6_adapter/scripts/gpu_monitor.py \
    --out "$DIAG_DIR/gpu_monitor.csv" \
    --interval 5 &
  MON_PID=$!
  trap 'kill $MON_PID 2>/dev/null || true' EXIT
  python -u -m hamgnn_rt.run_hamgnn_rt \
    --config "$CONFIG" \
    --diag-dir "$DIAG_DIR" \
    --run-name "$RUN_NAME" \
    > "$DIAG_DIR/train.log" 2>&1
  kill "$MON_PID" 2>/dev/null || true
  echo "DONE method=$METHOD config=$CONFIG diag=$DIAG_DIR"
done

python scripts/summarize_results.py
