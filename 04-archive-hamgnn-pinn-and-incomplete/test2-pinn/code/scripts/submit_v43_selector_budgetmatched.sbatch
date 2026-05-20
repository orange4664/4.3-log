#!/bin/bash
#SBATCH -J sqrt_v43_selector
#SBATCH -p amd_m9_768
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -o logs/v43_selector_%j.out
#SBATCH -e logs/v43_selector_%j.err

set -euo pipefail
mkdir -p logs
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

OUTDIR="results/sq_rt_v43_selector_${SLURM_JOB_ID:-manual}"
python experiments/run_sq_rt_v43_selector_bench.py \
  --mode budget_matched \
  --outdir "${OUTDIR}" \
  --workers "${SLURM_CPUS_PER_TASK:-32}" \
  --seeds 16 \
  --steps 200
