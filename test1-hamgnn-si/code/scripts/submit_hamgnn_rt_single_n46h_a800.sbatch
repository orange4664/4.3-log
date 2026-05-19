#!/bin/bash
#SBATCH -J hamgnn_rt
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 24:00:00

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export OPENBLAS_NUM_THREADS=1

# Adjust to your environment if needed.
module load miniforge3/24.11 || true
module load cuda/12.4 || true
source activate "${HAMGNN_ENV:-hamgnn}"

CONFIG=${CONFIG:?Set CONFIG=/path/to/config.yaml}
RUN_NAME=${RUN_NAME:-$(basename "$CONFIG" .yaml)}
DIAG_DIR=${DIAG_DIR:-$(pwd)/rt_diag/${RUN_NAME}_${SLURM_JOB_ID}}
mkdir -p "$DIAG_DIR"

python scripts/gpu_monitor.py --out "$DIAG_DIR/gpu_monitor.csv" --interval "${GPU_MONITOR_INTERVAL:-5}" &
MON_PID=$!
trap 'kill $MON_PID 2>/dev/null || true' EXIT
python -u -m hamgnn_rt.run_hamgnn_rt --config "$CONFIG" --diag-dir "$DIAG_DIR" --run-name "$RUN_NAME" > "$DIAG_DIR/train.log" 2>&1
kill $MON_PID 2>/dev/null || true
