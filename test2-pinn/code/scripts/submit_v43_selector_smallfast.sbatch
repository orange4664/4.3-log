#!/bin/bash
#SBATCH -J sqv43_small
#SBATCH -p amd_m9_768
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -o logs/v43_small_%j.out
#SBATCH -e logs/v43_small_%j.err

set -euo pipefail
mkdir -p logs
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

OUTDIR="results/sq_rt_v43_selector_small_${SLURM_JOB_ID:-manual}"
python experiments/run_sq_rt_v43_selector_bench.py \
  --mode smallfast \
  --outdir "${OUTDIR}" \
  --workers "${SLURM_CPUS_PER_TASK:-32}"
