# test23-vit-cifar10-p2-deeper formal summary

Formal acceptance is established from the landed low-learning-rate formal
results.

## `adamw` at `lr=0.0005`

- `seed=0`: `best_test_acc = 0.7195`
- `seed=1`: `best_test_acc = 0.6994`
- `seed=2`: `best_test_acc = 0.7056`

## `muon_ns` at `lr=0.0005`

- `seed=0`: `best_test_acc = 0.7352`
- `seed=1`: `best_test_acc = 0.7359`
- `seed=2`: `best_test_acc = 0.7371`

## Judgment

Original `muon_ns` beats `adamw` on every landed low-learning-rate formal seed,
so this benchmark is accepted under the user's gate.
