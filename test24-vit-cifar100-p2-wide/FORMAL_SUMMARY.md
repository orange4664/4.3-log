# test24-vit-cifar100-p2-wide formal summary

Formal acceptance is established from the landed low-learning-rate formal
results.

## `adamw` at `lr=0.0005`

- `seed=0`: `best_test_acc = 0.4562`
- `seed=1`: `best_test_acc = 0.4505`
- `seed=2`: `best_test_acc = 0.4533`

## `muon_ns` at `lr=0.0005`

- `seed=0`: `best_test_acc = 0.4660`
- `seed=1`: `best_test_acc = 0.4755`
- `seed=2`: `best_test_acc = 0.4738`

## Judgment

Original `muon_ns` beats `adamw` on every landed low-learning-rate formal seed,
so this benchmark is accepted under the user's gate.
