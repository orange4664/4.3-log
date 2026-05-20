# test3-burgers-pinn analysis

## Run record

- Server: `nmcc-n46h1`
- Work tree: `~/run/4.3-log/test3-burgers-pinn`
- Queue: `gpu_a800`
- Job id: `73340`
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
  - `grid_pde_residual=0.024271160364151`
  - `final_train_loss=0.14088129997253418`
- `muon_ns_adamw` best run:
  - `lr=0.002`
  - `seed=0`
  - `grid_pde_residual=0.03735242038965225`
  - `final_train_loss=0.13149426877498627`

Although `muon_ns_adamw` reaches a slightly lower final train loss on its best
run, the ranking metric for this benchmark is PDE residual, and on that metric
`adamw` is better than `muon_ns_adamw`.

## Conclusion

- `test3-burgers-pinn` is harder than `test2-pinn`
- but it still does **not** satisfy the acceptance gate
- therefore it remains a rejected screening benchmark, not a "good benchmark"
