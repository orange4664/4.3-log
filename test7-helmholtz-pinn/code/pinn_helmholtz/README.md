# pinn_helmholtz

This folder contains a high-frequency Helmholtz PINN benchmark with a known
exact solution.

## Goal

Screen whether this benchmark is a "good benchmark" under the user's rule:

- accept only if `muon_ns_adamw` beats `adamw`

## Why this benchmark is hard

- high-frequency oscillatory target
- elliptic residual with large wave number
- deeper network
- many interior collocation points

## PDE

We solve

```text
-u_xx - c * u = f(x)
```

with boundary conditions `u(0)=u(1)=0` and manufactured exact solution

```text
u(x) = sin(k*pi*x)
```

for integer `k`. The coefficient `c` is chosen separately from the target mode so
the PDE does not collapse into a resonant homogeneous problem.

## Outputs

- `records.csv`
- `summary_by_optimizer_lr.csv`
- `ranking_by_rel_l2.csv`
- `best_by_optimizer.csv`
- `curves.csv`
