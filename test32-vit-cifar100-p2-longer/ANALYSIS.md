# test32-vit-cifar100-p2-longer analysis

## Benchmark definition

- model: ViT with `patch_size=2`
- data: real CIFAR-100
- benchmark type: non-HamGNN, non-toy, matrix-heavy visual classification
- stress axis: longer optimization horizon than earlier CIFAR-100 retries
- acceptance gate: benchmark is accepted only if original `muon_ns` beats
  `adamw`

## Motivation

The accepted CIFAR-100 family already suggests that real ViT training can be a
good Muon-family benchmark when the model is matrix-heavy enough. This retry
keeps the same real data and backbone family, but pushes on schedule length
instead of only width/depth.

That gives us a distinct benchmark folder with a real model and a real dataset,
without falling back to toy tasks or HamGNN-specific behavior.

## Current status

Smoke completed on `gpu_h100` and passed the user's gate.

Smoke result:

- `adamw`
  - `steps=1170`
  - `best_test_acc=0.2623046875`
- `muon_ns`
  - `steps=1170`
  - `best_test_acc=0.2734375`

Formal has now been submitted on `gpu_h100`:

- job `73437`

## Next action

- let `73437` run the full lr/seed block
- accept only if the formal results preserve `muon_ns > adamw`
