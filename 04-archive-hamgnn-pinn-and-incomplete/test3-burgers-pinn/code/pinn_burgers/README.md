# pinn_burgers

This folder contains a harder Burgers PINN benchmark than the earlier heat-equation setup.

## Goal

Screen whether this benchmark is a "good benchmark" under the user's rule:

- accept the benchmark only if `muon_ns_adamw` beats `adamw`

## What is harder here

- nonlinear Burgers residual
- smaller viscosity
- deeper and wider network
- more residual points
- extra center-weighting on the residual to emphasize the shock-forming region

## Outputs

- `records.csv`
- `summary_by_optimizer_lr.csv`
- `ranking_by_residual.csv`
- `best_by_optimizer.csv`
- `curves.csv`
