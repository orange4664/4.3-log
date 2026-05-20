# test30-tiny-imagenet-vit formal summary

## Verdict

Accepted under the benchmark-selection gate.

The gate is:

- count the benchmark as good only if original `muon_ns` beats `adamw`

## Smoke result

- `adamw best_test_acc = 0.0825`
- `muon_ns best_test_acc = 0.08859375`

So the smoke gate passed and the benchmark was promoted to formal.

## Formal low-learning-rate results

### `adamw` at `lr=0.0005`

- `seed=0`: `best_test_acc = 0.2513`
- `seed=1`: `best_test_acc = 0.2483`
- `seed=2`: `best_test_acc = 0.2542`

### `muon_ns` at `lr=0.0005`

- `seed=0`: `best_test_acc = 0.2571`
- `seed=1`: `best_test_acc = 0.2634`
- `seed=2`: `best_test_acc = 0.2636`

## Comparison

Every landed low-learning-rate `muon_ns` seed is above the strongest landed
low-learning-rate `adamw` seed.

- best landed `adamw` low-learning-rate accuracy: `0.2542`
- landed `muon_ns` low-learning-rate accuracies:
  - `0.2571`
  - `0.2634`
  - `0.2636`

That is sufficient to count this benchmark as accepted under the user's gate.

## Operational note

Slurm job `73432` was still running when this summary was written. The decisive
accepted evidence comes from the already-landed `lr=0.0005` formal block.
