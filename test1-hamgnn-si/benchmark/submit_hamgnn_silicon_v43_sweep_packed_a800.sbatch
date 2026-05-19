#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH -p gpu_a800
#SBATCH -J hgnn_v43_sw
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -t 03:00:00
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
ROOT=/data/run01/scwb923/hamgnn_workspace/hamgnn_silicon_v43_sweep
export PYTHONPATH=/data/home/scwb923/run/test/chat_archive_full_20260519_113500/hamgnn_rt_v6_adapter:/data/home/scwb923/run/HamGNN:${PYTHONPATH:-}
cd "$ROOT"
RUN_ID=${SLURM_JOB_ID:-manual}
tail -n +2 configs/manifest.tsv | while IFS=$'\t' read -r METHOD CONFIG; do
  RUN_NAME="v43_${METHOD}_${RUN_ID}"
  DIAG_DIR="$ROOT/diag/${RUN_NAME}"
  mkdir -p "$DIAG_DIR"
  echo "LAUNCH method=$METHOD config=$CONFIG diag=$DIAG_DIR"
  python /data/home/scwb923/run/test/chat_archive_full_20260519_113500/hamgnn_rt_v6_adapter/scripts/gpu_monitor.py --out "$DIAG_DIR/gpu_monitor.csv" --interval 5 &
  MON_PID=$!
  trap 'kill $MON_PID 2>/dev/null || true' EXIT
  python -u -m hamgnn_rt.run_hamgnn_rt --config "$CONFIG" --diag-dir "$DIAG_DIR" --run-name "$RUN_NAME" > "$DIAG_DIR/train.log" 2>&1
  kill $MON_PID 2>/dev/null || true
  echo "DONE method=$METHOD config=$CONFIG diag=$DIAG_DIR"
done