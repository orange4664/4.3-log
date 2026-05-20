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

Smoke has been submitted on `gpu_h100`:

- job `73516`

## Next action

- wait for the smoke summaries from `adamw` and `muon_ns`
- accept or reject promotion based on the smoke gate
