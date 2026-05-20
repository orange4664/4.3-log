# test19-vit-cifar10-large analysis

## Run record

- Server: `nmcc-n46h1`
- Work tree: `/data/run01/scwb923/4.3-log/test19-vit-cifar10-large`
- Queue: `gpu_a800`

## Benchmark definition

- model: scaled-up compact ViT
- data: real CIFAR-10 server cache
- benchmark type: non-HamGNN, non-toy, matrix-heavy visual classification
- acceptance gate: benchmark is accepted only if original `muon_ns` beats `adamw`

## Current status

Smoke run completed.

## Smoke result

### `adamw`

- `steps=780`
- `final_test_acc=0.494140625`
- `best_test_acc=0.518359375`

### `muon_ns`

- `steps=780`
- `final_test_acc=0.5427734375`
- `best_test_acc=0.5478515625`

## Conclusion

Under the scaled-up ViT + CIFAR-10 smoke configuration, original `muon_ns`
beats `adamw`, so this benchmark is currently **accepted at the smoke stage**
and should be promoted to a formal run.

## Formal progress

The formal run has already produced enough matched results to accept this
benchmark under the user's gate.

Completed `adamw` summaries so far at `lr=0.0005`:

- `seed=0`: `final_test_acc=0.6535`, `best_test_acc=0.6577`
- `seed=1`: `final_test_acc=0.6610`, `best_test_acc=0.6642`
- `seed=2`: `final_test_acc=0.6544`, `best_test_acc=0.6544`

Completed higher-learning-rate `adamw` runs at `lr=0.001`:

- `lr=0.001`, `seed=0`: `final_test_acc=0.5817`, `best_test_acc=0.5894`
- `lr=0.001`, `seed=1`: `final_test_acc=0.5784`, `best_test_acc=0.5784`
- `lr=0.001`, `seed=2`: `final_test_acc=0.5852`, `best_test_acc=0.5858`

Completed `muon_ns` summaries so far at `lr=0.0005`:

- `seed=0`: `final_test_acc=0.7284`, `best_test_acc=0.7316`
- `seed=1`: `final_test_acc=0.7305`, `best_test_acc=0.7317`
- `seed=2`: `final_test_acc=0.7322`, `best_test_acc=0.7322`

## Formal conclusion

Under the matched formal `lr=0.0005` setting, all landed `muon_ns` runs are
clearly above all landed `adamw` runs:

- best landed `adamw`: `0.6642`
- landed `muon_ns`: `0.7316`, `0.7317`, `0.7322`

That is strong enough to accept this benchmark as a good benchmark under the
user's rule.
