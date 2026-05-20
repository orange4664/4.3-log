# test33-stl10-vit-longer analysis

## Benchmark definition

- model: ViT with patch embedding on `96x96` images
- data: real STL10
- benchmark type: non-HamGNN, non-toy, matrix-heavy visual classification
- stress axis: longer optimization horizon than `test29-stl10-vit`
- acceptance gate: benchmark is accepted only if original `muon_ns` beats
  `adamw`

## Motivation

`test29-stl10-vit` is already an accepted non-HamGNN benchmark. This retry does
not change the dataset family or collapse into a toy setup; it simply makes the
optimization path longer. That gives us another realistic benchmark folder that
can confirm whether the STL10 signal is robust rather than schedule-specific.

## Current status

Smoke completed on `gpu_h100` and passed the user's gate.

Smoke result:

- `adamw`
  - `steps=234`
  - `best_test_acc=0.348828125`
- `muon_ns`
  - `steps=234`
  - `best_test_acc=0.37265625`

Formal has now been submitted on `gpu_h100`:

- job `73438`

Gate-only rerun has also been submitted on `gpu_h100`:

- job `73442`

Current landed gate-only evidence:

- `adamw lr=0.0005`
  - `seed=0`: `best_test_acc=0.497265625`
  - `seed=1`: `best_test_acc=0.498046875`
  - `seed=2`: `best_test_acc=0.4974609375`
- `adamw lr=0.001`
  - `seed=0`: `best_test_acc=0.4439453125`
  - `seed=1`: `best_test_acc=0.425`
  - `seed=2`: `best_test_acc=0.4181640625`
- `muon_ns lr=0.0005`
  - `seed=0`: `best_test_acc=0.526953125`
  - `seed=1`: `best_test_acc=0.5294921875`
  - `seed=2`: `best_test_acc=0.5220703125`
- `muon_ns lr=0.001`
  - `seed=0`: `best_test_acc=0.57734375`
  - `seed=1`: `best_test_acc=0.5810546875`

So the landed `muon_ns` gate-only block is not just competitive. Every landed
`muon_ns` value is already above the strongest landed `adamw` value
`0.498046875`.

## Verdict

This benchmark is accepted under the user's gate.

The acceptance rule is:

- count the benchmark as good only if original `muon_ns` beats `adamw`

That condition is already satisfied decisively on the landed gate-only runs.

## Next action

- let the remaining `muon_ns lr=0.001` seed and the older full-formal job
  finish for completeness
- retain the current acceptance verdict unless new landed `muon_ns` evidence
  contradicts the already decisive gap
