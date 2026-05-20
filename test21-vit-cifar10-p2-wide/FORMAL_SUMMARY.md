# test21-vit-cifar10-p2-wide formal summary

Formal acceptance is established from the landed low-learning-rate formal
results.

## `adamw` at `lr=0.0005`

- `seed=0`: `best_test_acc = 0.6903`
- `seed=1`: `best_test_acc = 0.6862`
- `seed=2`: `best_test_acc = 0.6923`

## `muon_ns` at `lr=0.0005`

- `seed=0`: `best_test_acc = 0.7486`
- `seed=1`: `best_test_acc = 0.7414`
- `seed=2`: `best_test_acc = 0.7532`

## Judgment

Original `muon_ns` beats `adamw` on every landed low-learning-rate formal seed,
so this benchmark is accepted under the user's gate.
