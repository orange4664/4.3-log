# test5-allen-cahn-pinn status

This folder adds a stiffer nonlinear PDE benchmark to continue screening for
"good benchmarks" under the user's rule.

Acceptance rule:

- only accept this benchmark if `muon_ns_adamw` beats `adamw`

Current contents:

- `code/pinn_allen_cahn/`
- `scripts/submit_allen_cahn_pinn_a800.sbatch`

This folder is intentionally separate from:

- `test2-pinn` (too easy / smooth heat PINN)
- `test3-burgers-pinn` (hard nonlinear transport PINN)
- `test4-poisson-pinn` (high-frequency elliptic PINN)
