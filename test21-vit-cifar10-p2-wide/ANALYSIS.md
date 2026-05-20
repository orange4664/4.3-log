# test21-vit-cifar10-p2-wide analysis

## Run record

- Server: `nmcc-n46h1`
- Work tree: `/data/run01/scwb923/4.3-log/test21-vit-cifar10-p2-wide`
- Queue: `gpu_a800`

## Benchmark definition

- model: wider compact ViT with patch size `2`
- data: real CIFAR-10 server cache
- benchmark type: non-HamGNN, non-toy, matrix-heavy visual classification
- acceptance gate: benchmark is accepted only if original `muon_ns` beats `adamw`

## Current status

Smoke run completed.

## Smoke result

### `adamw`

- `steps=780`
- `final_test_acc=0.4947265625`
- `best_test_acc=0.5017578125`

### `muon_ns`

- `steps=780`
- `final_test_acc=0.553515625`
- `best_test_acc=0.555859375`

## Conclusion

Under the wider long-sequence ViT + CIFAR-10 smoke configuration, original
`muon_ns` beats `adamw`, so this benchmark is currently **accepted at the smoke
stage** and should be promoted to a formal run.

## Formal progress

Completed `adamw` summaries so far at `lr=0.0005`:

- `seed=0`: `final_test_acc=0.6882`, `best_test_acc=0.6903`
- `seed=1`: `final_test_acc=0.6838`, `best_test_acc=0.6862`
- `seed=2`: `final_test_acc=0.6923`, `best_test_acc=0.6923`

Landed `muon_ns` formal summaries now available at `lr=0.0005`:

- `seed=0`: `final_test_acc=0.7468`, `best_test_acc=0.7486`
- `seed=1`: `final_test_acc=0.7403`, `best_test_acc=0.7414`
- `seed=2`: `final_test_acc=0.7532`, `best_test_acc=0.7532`

All landed `muon_ns` low-learning-rate formal summaries exceed every landed
`adamw` formal
result:

- best landed `adamw` so far = `0.6923`

So formal acceptance is already established under the user's gate. The active
job may still be finishing later method blocks, but the benchmark itself is now
a valid accepted non-HamGNN benchmark because original `muon_ns` beats `adamw`
across the landed low-learning-rate formal seeds.
