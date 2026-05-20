# test27-tabular-mlp-suite analysis

## Run record

- server: `nmcc-n46h1`
- work tree: `~/run/4.3-log/test27-tabular-mlp-suite`
- queue: `gpu_a800`
- smoke job id: `73422`

Smoke config from `runs_smoke/run_args.json`:

- datasets:
  - `breast_cancer`
  - `wine`
  - `digits`
- model:
  - width `512`
  - depth `5`
  - dropout `0.1`
- learning rates:
  - `0.0005`
  - `0.001`
  - `0.002`
- smoke seeds:
  - `0`

## Acceptance rule

This benchmark is only accepted if:

- `muon_ns` beats `adamw`

## Result

This benchmark is **not accepted** under the user's rule.

From `runs_smoke/overall_optimizer_summary.csv`:

- `adamw`
  - `mean_test_acc_at_best_val = 0.9688596491228071`
- `muon_ns`
  - `mean_test_acc_at_best_val = 0.9585282651072125`
  - `win_count_vs_adamw = 1 / 3`

So even on the aggregate smoke criterion, original `muon_ns` is below `adamw`.

Per-dataset best rows from `runs_smoke/best_by_dataset_optimizer.csv`:

- `breast_cancer`
  - `adamw`: `test_acc_at_best_val_mean = 0.9649122807017544`
  - `muon_ns`: `0.956140350877193`
- `digits`
  - `adamw`: `0.9694444444444444`
  - `muon_ns`: `0.9666666666666667`
- `wine`
  - `adamw`: `1.0`
  - `muon_ns`: `1.0`

On this smoke screen:

- `muon_ns` loses on `breast_cancer`
- `muon_ns` loses on `digits`
- `muon_ns` ties `wine`

That is not enough to justify promotion to formal.

## Notes

This suite follows the public Muon-benchmark direction of evaluating
matrix-heavy MLPs on real tabular datasets instead of shallow toy models.
However, for this specific suite and smoke setting, it does not satisfy the
acceptance gate that original `muon_ns` must beat `adamw`.
