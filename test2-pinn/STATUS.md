# test2-pinn status

This folder is the PDE/PINN-side follow-up bundle.

It is intentionally different from `test1-hamgnn-si`:

- `test1-hamgnn-si` contains a completed HamGNN silicon benchmark with final ranked results.
- `test2-pinn` is the prepared source-and-run bundle for the `v4.3` selector bench line that is meant to support the next PDE/PINN-oriented benchmark stage.

## What is included

- `code/README.md`
  - upstream selector-bench README
- `code/requirements.txt`
  - package requirements
- `code/experiments/`
  - `run_sq_rt_v41_bench.py`
  - `run_sq_rt_v42_bench.py`
  - `run_sq_rt_v43_selector_bench.py`
- `code/rtmuon/`
  - optimizer and toy benchmark implementation
- `code/scripts/`
  - `smoke_test_v43.sh`
  - `submit_v43_selector_budgetmatched.sbatch`
  - `submit_v43_selector_smallfast.sbatch`
  - `submit_v43_selector_with_ablations.sbatch`
- `results/smoke_v43/`
  - included smoke outputs from the selector bench package

## What is not claimed here

This folder does **not** claim that a full PINN/PDE benchmark has already been run and verified.

At the moment, this is a prepared benchmark package and traceable source bundle for the next stage:

- PDE/PINN-style validation for the `v4.3` line
- selector-family verification outside the completed HamGNN silicon benchmark

## Why this folder exists

The work so far established that:

1. the corrected HamGNN `v4.3` path can win on the silicon Hamiltonian benchmark;
2. that benchmark is not a PINN benchmark;
3. the next requested direction is a separate PDE/PINN-oriented benchmark track.

So this folder is the repository location reserved for that follow-up track.
