# pinn_poisson

This folder contains a high-frequency Poisson PINN benchmark.

## Goal

Screen whether this benchmark is a "good benchmark" under the user's rule:

- accept the benchmark only if `muon_ns_adamw` beats `adamw`

## What is harder here

- higher-frequency target function
- deeper network
- more interior collocation points
- second-derivative PDE residual

## Outputs

- `records.csv`
- `summary_by_optimizer_lr.csv`
- `ranking_by_rel_l2.csv`
- `best_by_optimizer.csv`
- `curves.csv`
