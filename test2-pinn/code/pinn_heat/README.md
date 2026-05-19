# pinn_heat

This subfolder is the real PDE/PINN benchmark added after the earlier selector/toy bundle.

## What it runs

- PDE: 1D heat equation
- Method: physics-informed neural network
- Exact solution available, so the benchmark can report direct grid error

Equation:

```text
u_t = nu * u_xx
u(x,0) = sin(pi x)
u(0,t) = u(1,t) = 0
```

Exact solution:

```text
u(x,t) = exp(-nu*pi^2*t) * sin(pi*x)
```

## Optimizers compared

- `adamw`
- `muon_ns_adamw`
- `rt_v43_stream_adamw`
- `rt_v43_ns_adamw`

For the three Muon-family entries, the matrix parameters go through the repaired
`HamGNNMuonFDTOptimizer` implementation from `test1-hamgnn-si/code/hamgnn_rt/optimizers.py`.
Non-matrix parameters use the optimizer's AdamW fallback path. That is the
same hybrid split we accepted on the HamGNN side.

## Run example

```bash
python code/pinn_heat/run_heat_pinn_benchmark.py --outdir results/heat1d_pinn_20260519 --device cpu
```

## Output files

- `records.csv`
- `curves.csv`
- `summary_by_optimizer_lr.csv`
- `ranking_by_rel_l2.csv`
- `best_by_optimizer.csv`
- `run_args.json`
