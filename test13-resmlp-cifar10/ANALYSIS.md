# test13-resmlp-cifar10 analysis

## Run record

- Server: `nmcc-n46h1`
- Work tree: `/data/run01/scwb923/4.3-log/test13-resmlp-cifar10`
- Queue: `gpu_a800`
- Job:
  - `73378`

## Benchmark definition

- model: compact ResMLP-style classifier
- data: real CIFAR-10 server cache
- benchmark type: non-HamGNN, non-toy, matrix-heavy visual classification
- acceptance gate: benchmark is accepted only if original `muon_ns` beats `adamw`

## Current server status

The formal run has been submitted and is currently in progress.

No completed `summary.json` files are available yet, so no benchmark decision is
possible at this point.

## Smoke result

The smoke run is already enough to reject the benchmark under the user's gate
for the current configuration.

From the server `runs_smoke` summaries:

### `adamw`

- `steps=780`
- `final_test_acc=0.548828125`
- `best_test_acc=0.548828125`

### `muon_ns`

- `steps=780`
- `final_test_acc=0.4853515625`
- `best_test_acc=0.487109375`

So original `muon_ns` is again clearly worse than `adamw`.

## Current conclusion

- benchmark scaffold is valid
- formal server run is active
- but under the current model and hyperparameter setting, the smoke evidence is
  already **rejected** because original `muon_ns` loses to `adamw`

## Formal run update

The formal run currently continues to support rejection.

Best completed `adamw` result so far:

- `lr=0.001`, `seed=0`
- `best_test_acc=0.7579`

Best completed `muon_ns` result so far:

- `lr=0.0005`, `seed=1`
- `best_test_acc=0.6332`

So the formal gap remains large, not marginal.

### Updated conclusion

- `test13-resmlp-cifar10` remains clearly **rejected**
