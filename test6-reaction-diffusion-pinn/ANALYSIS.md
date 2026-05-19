# test6-reaction-diffusion-pinn analysis

## Run record

- Server: `nmcc-n46h1`
- Work tree: `~/run/4.3-log/test6-reaction-diffusion-pinn`
- Queue: `gpu_a800`
- Job id: `73342`
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
  - `grid_rel_l2=0.009876834228634834`
  - `grid_mse=2.6180028726230375e-05`
- `muon_ns_adamw` best run:
  - `lr=0.002`
  - `seed=2`
  - `grid_rel_l2=0.1059764102101326`
  - `grid_mse=0.003014067653566599`

The original Muon baseline is again substantially worse than AdamW on the main
grid error metric.

## Conclusion

- `test6-reaction-diffusion-pinn` is a valid harder PDE screening task
- but it still fails the benchmark-acceptance gate
- therefore it is rejected under the user's rule
