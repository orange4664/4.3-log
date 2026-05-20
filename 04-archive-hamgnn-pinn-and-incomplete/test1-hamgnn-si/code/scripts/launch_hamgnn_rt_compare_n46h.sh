#!/usr/bin/env bash
set -euo pipefail
GPU_TYPE=${1:-h200}
BASE_CONFIG=${BASE_CONFIG:?Set BASE_CONFIG=/path/to/HamGNN config.yaml}
TAG=${RUN_TAG:-hamgnn_rt_compare}
METHODS=${METHODS:-adamw,muon_stream,rt_v43_stream,muon_stream_fdt_metric,rt_v6_fdt_metric,rt_v6_fdt_metric_period4}
LRS=${LRS:-0.003,0.01,0.03}
MAX_PARALLEL=${MAX_PARALLEL:-2}
OUTROOT=${OUTROOT:-$(pwd)/runs_${TAG}_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUTROOT/configs"
MANIFEST=$(python tools/make_hamgnn_optimizer_configs.py --base-config "$BASE_CONFIG" --outdir "$OUTROOT/configs" --methods "$METHODS" --lrs "$LRS" --tag "$TAG")
SCRIPT=scripts/submit_hamgnn_rt_single_n46h_h200.sbatch
if [[ "$GPU_TYPE" == "h100" ]]; then
  SCRIPT=scripts/submit_hamgnn_rt_single_n46h_h100.sbatch
elif [[ "$GPU_TYPE" == "a800" ]]; then
  SCRIPT=scripts/submit_hamgnn_rt_single_n46h_a800.sbatch
fi
# Generate array script with one config per task.
ARRAY_SCRIPT="$OUTROOT/array_${GPU_TYPE}.sbatch"
N=$(($(wc -l < "$MANIFEST") - 1))
cat > "$ARRAY_SCRIPT" <<EOF
#!/bin/bash
#SBATCH -J ${TAG}
#SBATCH -p gpu_${GPU_TYPE}
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c $([[ "$GPU_TYPE" == "a800" ]] && echo 8 || echo 12)
#SBATCH -t 24:00:00
#SBATCH --array=1-${N}%${MAX_PARALLEL}
set -euo pipefail
module load miniforge3/24.11 || true
module load cuda/12.4 || true
source activate "\${HAMGNN_ENV:-hamgnn}"
LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID+1))p" "$MANIFEST")
METHOD=\$(echo "\$LINE" | cut -f1)
LR=\$(echo "\$LINE" | cut -f2)
CONFIG=\$(echo "\$LINE" | cut -f3)
RUN_NAME="${TAG}_\${METHOD}_lr\${LR}_task\${SLURM_ARRAY_TASK_ID}"
DIAG_DIR="$OUTROOT/diag/\${RUN_NAME}"
mkdir -p "\$DIAG_DIR"
python scripts/gpu_monitor.py --out "\$DIAG_DIR/gpu_monitor.csv" --interval "\${GPU_MONITOR_INTERVAL:-5}" &
MON_PID=\$!
trap 'kill \$MON_PID 2>/dev/null || true' EXIT
python -u -m hamgnn_rt.run_hamgnn_rt --config "\$CONFIG" --diag-dir "\$DIAG_DIR" --run-name "\$RUN_NAME" > "\$DIAG_DIR/train.log" 2>&1
kill \$MON_PID 2>/dev/null || true
EOF
chmod +x "$ARRAY_SCRIPT"
echo "$OUTROOT" > latest_hamgnn_rt_root.txt
echo "$MANIFEST" > latest_hamgnn_rt_manifest.txt
sbatch "$ARRAY_SCRIPT"
