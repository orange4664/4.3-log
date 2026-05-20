# heat1d_pinn benchmark

This is a real PDE/PINN benchmark, separate from the earlier selector toy benchmark.

## PDE

- Equation: `u_t = nu * u_xx` on `x in [0,1]`, `t in [0,1]`
- Initial condition: `u(x,0) = sin(pi x)`
- Boundary condition: `u(0,t) = u(1,t) = 0`
- Exact solution: `exp(-nu*pi^2*t) * sin(pi*x)`

## Optimizers

- `adamw`: pure AdamW baseline
- `muon_ns_adamw`: repaired Muon-NS for matrix params, AdamW fallback for non-matrix params
- `rt_v43_stream_adamw`: repaired v4.3 stream path for matrix params, AdamW fallback elsewhere
- `rt_v43_ns_adamw`: repaired v4.3 Muon-source path for matrix params, AdamW fallback elsewhere

## Key files

- `records.csv`
- `summary_by_optimizer_lr.csv`
- `ranking_by_rel_l2.csv`
- `best_by_optimizer.csv`
- `curves.csv`

## Best runs by optimizer

- `adamw` at `lr=0.003` seed `2`: `grid_rel_l2=0.00258126`, `grid_mse=1.44083e-06`, `final_train_loss=0.000146751`
- `muon_ns_adamw` at `lr=0.002` seed `0`: `grid_rel_l2=0.0252014`, `grid_mse=0.00013734`, `final_train_loss=0.00117515`
- `rt_v43_ns_adamw` at `lr=0.002` seed `0`: `grid_rel_l2=0.0353985`, `grid_mse=0.000270969`, `final_train_loss=0.00332523`
- `rt_v43_stream_adamw` at `lr=0.002` seed `0`: `grid_rel_l2=0.0229972`, `grid_mse=0.000114367`, `final_train_loss=0.00225744`

This benchmark uses the repaired torch implementation from `test1-hamgnn-si/code/hamgnn_rt/optimizers.py`.
