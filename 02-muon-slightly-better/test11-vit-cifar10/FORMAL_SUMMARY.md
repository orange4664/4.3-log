# test11-vit-cifar10 formal summary

## Decision

`test11-vit-cifar10` is an **accepted** benchmark under the user's rule.

Reason:

- original `muon_ns` beats `adamw` on the same non-HamGNN, non-toy, real-data,
  matrix-heavy benchmark
- this is supported by both smoke evidence and formal multi-seed evidence

## Smoke evidence

From `runs_smoke/all_summaries.json`:

- `adamw`: `best_test_acc = 0.535546875`
- `muon_ns`: `best_test_acc = 0.540234375`

So the smoke run already passed the gate.

## Formal evidence currently completed

### `adamw`

Completed runs:

- `lr=0.0005`, seeds `0,1,2`
- `lr=0.001`, seeds `0,1,2`

Best completed `adamw` result:

- `lr=0.0005`, `seed=2`
- `best_test_acc = 0.6948`

### `muon_ns`

Completed runs:

- `lr=0.0005`, seeds `0,1,2`
- `lr=0.001`, seed `0`

Best completed `muon_ns` result:

- `lr=0.001`, `seed=0`
- `best_test_acc = 0.7624`

### Seed-matched view at `lr=0.0005`

`adamw`:

- seed `0`: `0.6913`
- seed `1`: `0.6812`
- seed `2`: `0.6948`
- mean: `0.6891`

`muon_ns`:

- seed `0`: `0.7066`
- seed `1`: `0.7131`
- seed `2`: `0.7157`
- mean: `0.7118`

So original `muon_ns` wins all three completed seed-matched runs at
`lr=0.0005`.

## Benchmark description

- dataset: CIFAR-10
- model: compact ViT
- benchmark type: non-HamGNN, non-toy, matrix-heavy visual classifier

## Current conclusion

This is the strongest accepted benchmark found so far in this search thread.
