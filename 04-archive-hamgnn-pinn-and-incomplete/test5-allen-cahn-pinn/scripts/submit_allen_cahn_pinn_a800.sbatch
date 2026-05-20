#!/bin/bash
#SBATCH -J test5_allencahn
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

cd /data/home/scwb923/run/4.3-log/test5-allen-cahn-pinn
mkdir -p logs results/allen_cahn_pinn_20260519

source /usr/share/modules/init/bash >/dev/null 2>&1 || true
module load miniforge3/25.11.0-1 || true
module load cuda/12.8 || true

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

PYTHON_BIN=/data/home/scwb923/run/envs/hamgnn/bin/python
echo "Using python: $PYTHON_BIN"
"$PYTHON_BIN" --version

"$PYTHON_BIN" -u code/pinn_allen_cahn/run_allen_cahn_pinn_benchmark.py \
  --outdir results/allen_cahn_pinn_20260519 \
  --device cuda \
  --steps 800 \
  --width 96 \
  --depth 6 \
  --n-residual 4096 \
  --n-initial 256 \
  --n-boundary 256 \
  --eps 0.01 \
  --alpha 5.0 \
  --lrs 0.001,0.002 \
  --seeds 0,1,2 \
  --lr-switch 0.0015
