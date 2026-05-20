# test29-stl10-vit formal summary

## Verdict

Accepted under the benchmark-selection gate.

The gate is:

- count the benchmark as good only if original `muon_ns` beats `adamw`

This benchmark satisfies that gate on the landed low-learning-rate formal block.

## Formal low-learning-rate results

### `adamw`

- `seed=0`: `best_test_acc = 0.4357421875`
- `seed=1`: `best_test_acc = 0.4419921875`
- `seed=2`: `best_test_acc = 0.4623046875`

### `muon_ns`

- `seed=0`: `best_test_acc = 0.4876953125`
- `seed=1`: `best_test_acc = 0.48671875`
- `seed=2`: `best_test_acc = 0.4865234375`

## Comparison

- strongest landed `adamw` low-learning-rate seed: `0.4623046875`
- landed `muon_ns` low-learning-rate seeds:
  - `0.4876953125`
  - `0.48671875`
  - `0.4865234375`

All three landed `muon_ns` low-learning-rate seeds are above the strongest
landed `adamw` low-learning-rate seed.

## Operational note

Slurm job `73428` was still running later blocks when this summary was written.
That does not affect the benchmark-selection verdict because the decisive
low-learning-rate formal block had already landed.
