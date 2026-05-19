#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
python experiments/run_sq_rt_v43_selector_bench.py --mode smallfast --outdir results/smoke_v43 --workers 2
cat results/smoke_v43/search_budget.csv
cat results/smoke_v43/oracle_comparison.csv
