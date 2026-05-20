# test12-mlpmixer-cifar10 analysis

## Run record

- Server: `nmcc-n46h1`
- Work tree: `/data/run01/scwb923/4.3-log/test12-mlpmixer-cifar10`
- Queue: `gpu_a800`
- Job:
  - `73377`

## Benchmark definition

- model: compact MLP-Mixer
- data: real CIFAR-10 server cache
- benchmark type: non-HamGNN, non-toy, matrix-heavy visual classification
- acceptance gate: benchmark is accepted only if original `muon_ns` beats `adamw`

## Current server status

The formal run is still in progress while this note is being updated.

At the current checkpoint, the `adamw` summaries for `lr=0.0005` and all three
seeds have been produced on the server. The `muon_ns` summaries are not yet
complete, so the benchmark cannot yet be accepted or rejected.

## Smoke result

The smoke run is already sufficient to reject the benchmark under the user's
gate for the current configuration.

From the server `runs_smoke` summaries:

### `adamw`

- `steps=780`
- `final_test_acc=0.60625`
- `best_test_acc=0.6130859375`

### `muon_ns`

- `steps=780`
- `final_test_acc=0.533203125`
- `best_test_acc=0.533203125`

So original `muon_ns` is clearly worse than `adamw` on the smoke version of
this benchmark.

## Current conclusion

- benchmark scaffold is valid
- formal server run is active
- but under the current model and hyperparameter setting, the smoke evidence is
  already **rejected** because original `muon_ns` loses to `adamw`

## Formal run update

The formal run now has completed summaries for both screened learning rates.

Best completed `adamw` result:

- `lr=0.0005`, `seed=0`
- `best_test_acc=0.7766`

Best completed `muon_ns` result:

- `lr=0.001`, `seed=1`
- `best_test_acc=0.7741`

This is much closer than the smoke result suggested, but it still does **not**
pass the user's gate because the best original `muon_ns` run remains slightly
worse than the best `adamw` run.

### Updated conclusion

- `test12-mlpmixer-cifar10` is a strong near-miss benchmark
- but under the current evidence it is still **rejected**
