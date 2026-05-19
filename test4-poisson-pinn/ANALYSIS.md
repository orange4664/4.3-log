# test4-poisson-pinn analysis

## Run record

- Server: `nmcc-n46h1`
- Work tree: `~/run/4.3-log/test4-poisson-pinn`
- Queue: `gpu_a800`
- Job id: `73339`
- Python: `~/run/envs/hamgnn/bin/python`

## Acceptance rule

This benchmark is only accepted if:

- `muon_ns_adamw` beats `adamw`

## Result

This benchmark is **not accepted** under the user's rule.

From `best_by_optimizer.csv` on the server:

- `adamw` best run:
  - `lr=0.002`
  - `seed=2`
  - `grid_rel_l2=0.9909694790840149`
  - `grid_mse=0.5086636543273926`
- `muon_ns_adamw` best run:
  - `lr=0.002`
  - `seed=2`
  - `grid_rel_l2=5.124600887298584`
  - `grid_mse=13.602860450744629`

The benchmark is hard in the sense that all methods struggle, but the original
Muon baseline is still clearly worse than AdamW on the selection metric.

## Conclusion

- `test4-poisson-pinn` is a valid harder screening benchmark
- but it does **not** satisfy the acceptance gate
- therefore it remains a rejected screening benchmark, not a "good benchmark"
