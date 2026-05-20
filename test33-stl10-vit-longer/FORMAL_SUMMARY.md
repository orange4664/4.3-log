# test33-stl10-vit-longer formal summary

## Verdict

Accepted under the benchmark-selection gate.

The gate is:

- count the benchmark as good only if original `muon_ns` beats `adamw`

## Smoke result

- `adamw best_test_acc = 0.348828125`
- `muon_ns best_test_acc = 0.37265625`

The smoke gate passed, so the benchmark was promoted to a harder rerun.

## Landed gate-only results

### `adamw`

- `lr=0.0005`
  - `seed=0`: `0.497265625`
  - `seed=1`: `0.498046875`
  - `seed=2`: `0.4974609375`
- `lr=0.001`
  - `seed=0`: `0.4439453125`
  - `seed=1`: `0.425`
  - `seed=2`: `0.4181640625`

### `muon_ns`

- `lr=0.0005`
  - `seed=0`: `0.526953125`
  - `seed=1`: `0.5294921875`
  - `seed=2`: `0.5220703125`
- `lr=0.001`
  - `seed=0`: `0.57734375`
  - `seed=1`: `0.5810546875`

## Comparison

Strongest landed `adamw` value:

- `0.498046875`

Landed `muon_ns` values:

- `0.526953125`
- `0.5294921875`
- `0.5220703125`
- `0.57734375`
- `0.5810546875`

Every landed `muon_ns` value is above the strongest landed `adamw` value.

That is sufficient to count this benchmark as accepted under the user's gate.

## Operational note

Gate-only job `73442` is still running, and the older full-formal job `73438`
is also still present. The acceptance verdict already follows from the landed
gate-only evidence above.
