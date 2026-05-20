# test36-tiny-imagenet-mlpmixer analysis

## Benchmark definition

- model: MLP-Mixer on `64x64` images
- data: real Tiny-ImageNet
- benchmark type: non-HamGNN, non-toy, matrix-heavy visual classification
- stress axis: different architecture family from the accepted ViT line
- acceptance gate: benchmark is accepted only if original `muon_ns` beats
  `adamw`

## Motivation

The current stronger results lean heavily on ViT-style models. A Tiny-ImageNet
MLP-Mixer line tests whether the Muon-family signal transfers to another
matrix-heavy architecture without paying the cost of a new dataset.

## Current status

Scaffolded locally and synced to the cluster workspace:

- `/data/run01/scwb923/4.3-log/test36-tiny-imagenet-mlpmixer`

Smoke ran on `gpu_h100`:

- job `73516`

Landed smoke summaries:

- `adamw lr=0.0005 seed=0 best_test_acc = 0.13020833333333334`
- `muon_ns lr=0.0005 seed=0 best_test_acc = 0.12145833333333333`

## Conclusion

This benchmark is rejected at smoke stage under the user's gate:

- original `muon_ns` does not beat `adamw`
- therefore this branch should not be promoted to formal multi-seed runs
