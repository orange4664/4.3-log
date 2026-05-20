# test35-tiny-imagenet-vit-deeper analysis

## Benchmark definition

- model: ViT with patch embedding on `64x64` images
- data: real Tiny-ImageNet
- benchmark type: non-HamGNN, non-toy, matrix-heavy visual classification
- stress axis: harder than `test30` via deeper/wider model and longer horizon
- acceptance gate: benchmark is accepted only if original `muon_ns` beats
  `adamw`

## Motivation

The accepted Tiny-ImageNet result in `test30` is already useful, but it still
leaves room to ask whether the same real-data advantage survives on a stronger
ViT recipe. This benchmark answers that without opening a new data dependency.

## Current status

Scaffolded locally and synced to the cluster workspace:

- `/data/run01/scwb923/4.3-log/test35-tiny-imagenet-vit-deeper`

Smoke has been submitted on `gpu_h100`:

- job `73497`

The smoke job is no longer pending. It is actively running on:

- node `d1n41d29g01`

The first landed smoke summary is:

- `adamw lr=0.0005 seed=0 best_test_acc = 0.08375`
- `muon_ns lr=0.0005 seed=0 best_test_acc = 0.13916666666666666`

So the smoke gate already passes on the first landed pair.

## Smoke verdict

Under the user's gate:

- accept promotion only if original `muon_ns` beats `adamw`

The first landed smoke pair satisfies that condition:

- `adamw`: `0.08375`
- `muon_ns`: `0.13916666666666666`

## Next action

- submit the formal run
- keep the benchmark in the candidate-good set unless later formal evidence
  reverses the signal

## Formal status

The formal run has been submitted and is actively running:

- job `73503`
- node `d1n41d29g01`

The first landed formal summaries are now available for `adamw`:

- `lr=0.0005 seed=0 best_test_acc = 0.28033854166666666`
- `lr=0.0005 seed=1 best_test_acc = 0.27239583333333334`

No landed `muon_ns` formal summary is available yet.
