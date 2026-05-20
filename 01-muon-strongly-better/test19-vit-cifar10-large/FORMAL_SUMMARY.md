# test19-vit-cifar10-large formal summary

This benchmark is formally accepted under the user's gate.

## Gate

Accept only if original `muon_ns` beats `adamw`.

## Landed formal evidence

### `adamw`

- `lr=0.0005`
  - `seed=0`: `best_test_acc=0.6577`
  - `seed=1`: `best_test_acc=0.6642`
  - `seed=2`: `best_test_acc=0.6544`
- `lr=0.001`
  - `seed=0`: `best_test_acc=0.5894`
  - `seed=1`: `best_test_acc=0.5784`
  - `seed=2`: `best_test_acc=0.5858`

### `muon_ns`

- `lr=0.0005`
  - `seed=0`: `best_test_acc=0.7316`
  - `seed=1`: `best_test_acc=0.7317`
  - `seed=2`: `best_test_acc=0.7322`

## Decision

`muon_ns` clearly beats `adamw` on the formal landed runs, so
`test19-vit-cifar10-large` is accepted.
