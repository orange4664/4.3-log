# pinn_reaction_diffusion

This folder contains a manufactured reaction-diffusion PINN benchmark with a
multi-frequency exact solution.

## Goal

Screen whether this is a "good benchmark" under the user's rule:

- accept only if `muon_ns_adamw` beats `adamw`

## Why this benchmark is harder

- diffusion plus reaction term
- multi-frequency exact solution
- deeper network
- more residual points
- direct grid error against an exact solution

## PDE

We solve:

```text
u_t - nu * u_xx + gamma * u = f(x,t)
```

on `x in [0,1]`, `t in [0,1]`, with manufactured exact solution

```text
u(x,t) = exp(-t) * (sin(2*pi*x) + 0.5*sin(5*pi*x))
```

## Outputs

- `records.csv`
- `summary_by_optimizer_lr.csv`
- `ranking_by_rel_l2.csv`
- `best_by_optimizer.csv`
- `curves.csv`
