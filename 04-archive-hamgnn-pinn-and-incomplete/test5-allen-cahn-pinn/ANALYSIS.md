# test5-allen-cahn-pinn analysis

## Run record

- Server: `nmcc-n46h1`
- Work tree: `~/run/4.3-log/test5-allen-cahn-pinn`
- Queue: `gpu_a800`
- Job id: `73341`
- Python: `~/run/envs/hamgnn/bin/python`

## Acceptance rule

This benchmark is only accepted if:

- `muon_ns_adamw` beats `adamw`

## Result

This benchmark is **not accepted** under the user's rule.

From `best_by_optimizer.csv` on the server:

- `adamw` best run:
  - `lr=0.002`
  - `seed=0`
  - `grid_rel_l2=0.009831770323216915`
  - `grid_mse=2.075333941320423e-05`
- `muon_ns_adamw` best run:
  - `lr=0.002`
  - `seed=2`
  - `grid_rel_l2=0.022382961586117744`
  - `grid_mse=0.00010756220581242815`

So the original Muon baseline is still worse than AdamW on the benchmark's main
error metric.

## Important side observation

Although the benchmark is rejected under the user's gate, this run does show an
interesting split inside the optimizer family:

- `rt_v43_stream_adamw` is the best overall method on this benchmark
  - `grid_rel_l2=0.005986535456031561`
  - `grid_mse=7.694417035963852e-06`

That is useful for the repaired `v4.3` line, but it still does **not** satisfy
the benchmark-acceptance rule because the gate is tied to original
`muon_ns_adamw`.

## Conclusion

- `test5-allen-cahn-pinn` is a meaningful harder nonlinear PINN benchmark
- original `muon_ns_adamw` still does **not** beat `adamw`
- therefore this benchmark remains rejected under the user's acceptance rule
