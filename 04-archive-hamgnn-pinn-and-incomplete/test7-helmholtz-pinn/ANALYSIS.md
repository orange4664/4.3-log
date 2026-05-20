# test7-helmholtz-pinn analysis

## Run record

- Server: `nmcc-n46h1`
- Work tree: `~/run/4.3-log/test7-helmholtz-pinn`
- Queue: `gpu_a800`
- Job id: `73353`
- Python: `~/run/envs/hamgnn/bin/python`

## Acceptance rule

This benchmark is only accepted if:

- `muon_ns_adamw` beats `adamw`

## Result

This benchmark is **not accepted** under the user's rule.

From `best_by_optimizer.csv` on the server:

- `adamw` best run:
  - `lr=0.001`
  - `seed=2`
  - `grid_rel_l2=1.0324506759643555`
  - `grid_mse=0.5309033989906311`
- `muon_ns_adamw` best run:
  - `lr=0.001`
  - `seed=2`
  - `grid_rel_l2=1.1538639068603516`
  - `grid_mse=0.6631106734275818`

The original Muon baseline is again worse than AdamW on the selected error
metric, even though all methods are struggling on this hard oscillatory task.

## Conclusion

- `test7-helmholtz-pinn` is a harder non-toy benchmark
- but original `muon_ns_adamw` still does **not** beat `adamw`
- therefore this benchmark is rejected under the user's rule
