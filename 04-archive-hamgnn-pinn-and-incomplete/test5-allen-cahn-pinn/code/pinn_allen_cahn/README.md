# pinn_allen_cahn

This folder contains a stiffer nonlinear Allen-Cahn PINN benchmark with a
manufactured exact solution.

## Goal

Screen whether this is a "good benchmark" under the user's rule:

- accept only if `muon_ns_adamw` beats `adamw`

## Why this benchmark is harder than heat1d_pinn

- nonlinear reaction term
- small diffusion coefficient
- deeper and wider network
- more residual points
- direct grid error against an exact solution

## PDE

We solve a manufactured Allen-Cahn type equation on `x in [0, 1]`, `t in [0,1]`:

```text
u_t - eps * u_xx + alpha * (u^3 - u) = f(x,t)
```

with exact solution:

```text
u(x,t) = exp(-t) * sin(pi x)
```

The forcing `f(x,t)` is chosen analytically so the exact solution is known.

## Outputs

- `records.csv`
- `summary_by_optimizer_lr.csv`
- `ranking_by_rel_l2.csv`
- `best_by_optimizer.csv`
- `curves.csv`
