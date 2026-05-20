# test23-vit-cifar10-p2-deeper analysis

## Run record

- Server: `nmcc-n46h1`
- Work tree: `/data/run01/scwb923/4.3-log/test23-vit-cifar10-p2-deeper`
- Queue: `gpu_a800`

## Benchmark definition

- model: deeper compact ViT with patch size `2`
- data: real CIFAR-10 server cache
- benchmark type: non-HamGNN, non-toy, matrix-heavy visual classification
- acceptance gate: benchmark is accepted only if original `muon_ns` beats `adamw`

## Current status

Smoke run completed.

## Smoke result

### `adamw`

- `steps=780`
- `final_test_acc=0.5013671875`
- `best_test_acc=0.517578125`

### `muon_ns`

- `steps=780`
- `final_test_acc=0.52265625`
- `best_test_acc=0.534375`

## Conclusion

Under the deeper long-sequence ViT + CIFAR-10 smoke configuration, original
`muon_ns` beats `adamw`, so this benchmark is currently **accepted at the smoke
stage** and should be promoted to a formal run.

## Formal progress

Completed `adamw` summaries at `lr=0.0005`:

- `seed=0`: `final_test_acc=0.7178`, `best_test_acc=0.7195`
- `seed=1`: `final_test_acc=0.6994`, `best_test_acc=0.6994`
- `seed=2`: `final_test_acc=0.7038`, `best_test_acc=0.7056`

Completed higher-learning-rate `adamw` runs at `lr=0.001`:

- `seed=0`: `final_test_acc=0.6636`, `best_test_acc=0.6640`

Landed `muon_ns` formal summaries now available at `lr=0.0005`:

- `seed=0`: `final_test_acc=0.7349`, `best_test_acc=0.7352`
- `seed=1`: `final_test_acc=0.7337`, `best_test_acc=0.7359`
- `seed=2`: `final_test_acc=0.7355`, `best_test_acc=0.7371`

Both landed `muon_ns` formal summaries exceed every landed `adamw` formal
result:

- best landed `adamw` so far = `0.7195`

So formal acceptance is already established under the user's gate. The active
job is still running because higher-learning-rate or later method blocks remain,
but the benchmark itself is now a valid accepted non-HamGNN benchmark because
original `muon_ns` beats `adamw` across the landed low-learning-rate formal
seeds.
