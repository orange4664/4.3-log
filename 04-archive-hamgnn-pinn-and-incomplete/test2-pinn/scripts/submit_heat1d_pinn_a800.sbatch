#!/bin/bash
#SBATCH -J test2_pinn_v43
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

cd /data/home/scwb923/run/4.3-log/test2-pinn
mkdir -p logs results/heat1d_pinn_20260519

source /usr/share/modules/init/bash >/dev/null 2>&1 || true
module load miniforge3/25.11.0-1 || true
module load cuda/12.8 || true

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

if [ -x /data/home/scwb923/run/envs/test2-pinn/bin/python ]; then
  PYTHON_BIN=/data/home/scwb923/run/envs/test2-pinn/bin/python
elif [ -x /data/home/scwb923/run/envs/hamgnn/bin/python ]; then
  PYTHON_BIN=/data/home/scwb923/run/envs/hamgnn/bin/python
elif [ -x /data/home/scwb923/run/rtmuon-env/bin/python ]; then
  PYTHON_BIN=/data/home/scwb923/run/rtmuon-env/bin/python
else
  PYTHON_BIN=/data/apps/miniforge3/25.11.0-1/envs/py310-torch270-vllm090/bin/python
fi

echo "Using python: $PYTHON_BIN"
"$PYTHON_BIN" --version

"$PYTHON_BIN" -u code/pinn_heat/run_heat_pinn_benchmark.py \
  --outdir results/heat1d_pinn_20260519 \
  --device cuda \
  --steps 400 \
  --lrs 0.001,0.002,0.003 \
  --seeds 0,1,2 \
  --lr-switch 0.0015
