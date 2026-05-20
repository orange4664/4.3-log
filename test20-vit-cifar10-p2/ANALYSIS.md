# test20-vit-cifar10-p2 analysis

## Run record

- Server: `nmcc-n46h1`
- Work tree: `/data/run01/scwb923/4.3-log/test20-vit-cifar10-p2`
- Queue: `gpu_a800`

## Benchmark definition

- model: compact ViT with patch size `2`
- data: real CIFAR-10 server cache
- benchmark type: non-HamGNN, non-toy, matrix-heavy visual classification
- acceptance gate: benchmark is accepted only if original `muon_ns` beats `adamw`

## Current status

Smoke run completed.

## Smoke result

### `adamw`

- `steps=780`
- `final_test_acc=0.5046875`
- `best_test_acc=0.5283203125`

### `muon_ns`

- `steps=780`
- `final_test_acc=0.518359375`
- `best_test_acc=0.5353515625`

## Conclusion

Under the longer-sequence ViT + CIFAR-10 smoke configuration, original
`muon_ns` beats `adamw`, so this benchmark is currently **accepted at the smoke
stage** and should be promoted to a formal run.

## Formal progress

The formal run is active.

Completed `adamw` summaries so far at `lr=0.0005`:

- `seed=0`: `final_test_acc=0.7153`, `best_test_acc=0.7201`
- `seed=1`: `final_test_acc=0.7096`, `best_test_acc=0.7131`
- `seed=2`: `final_test_acc=0.7311`, `best_test_acc=0.7311`

Completed higher-learning-rate `adamw` runs at `lr=0.001`:

- `seed=0`: `final_test_acc=0.6834`, `best_test_acc=0.6847`
- `seed=1`: `final_test_acc=0.6729`, `best_test_acc=0.6757`
- `seed=2`: `final_test_acc=0.6735`, `best_test_acc=0.6779`

At this checkpoint, no `muon_ns` formal summary has landed yet, so the formal
accepted/rejected decision is still pending.

Update:

- `muon_ns`, `lr=0.0005`, `seed=0` has now landed:
  - `final_test_acc=0.7272`
  - `best_test_acc=0.7290`
- `muon_ns`, `lr=0.0005`, `seed=1` has now landed:
  - `final_test_acc=0.7259`
  - `best_test_acc=0.7280`
- `muon_ns`, `lr=0.0005`, `seed=2` has now landed:
  - `final_test_acc=0.7270`
  - `best_test_acc=0.7285`

Current read:

- `muon_ns` is competitive, but both landed formal points are still below the
  best landed `adamw` value `0.7311`
- and the third landed formal point is also below `0.7311`

## Formal conclusion

Under the formal landed runs for this benchmark:

- best `adamw`: `0.7311`
- landed `muon_ns`: `0.7290`, `0.7280`, `0.7285`

So original `muon_ns` does not beat `adamw` here, and this benchmark is
**rejected** under the user's gate.
