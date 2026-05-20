# Accepted Non-HamGNN Benchmarks Snapshot

This file records the currently verified accepted non-HamGNN benchmarks from
the ongoing Muon benchmark search.

Acceptance rule:

- a benchmark is accepted only if original `muon_ns` beats `adamw`

## Accepted now

### `test11-vit-cifar10`

- real CIFAR-10 visual benchmark
- compact ViT
- accepted from smoke and formal evidence

### `test16-mlpmixer-cifar10-refine`

- real CIFAR-10 visual benchmark
- refined compact MLP-Mixer
- accepted from smoke and formal evidence

### `test19-vit-cifar10-large`

- real CIFAR-10 visual benchmark
- wider/deeper ViT
- accepted from formal evidence

### `test23-vit-cifar10-p2-deeper`

- real CIFAR-10 visual benchmark
- longer-sequence deeper ViT
- accepted from landed low-learning-rate formal seeds:
  - `muon_ns`: `0.7352`, `0.7359`, `0.7371`
  - best landed `adamw`: `0.7195`

### `test24-vit-cifar100-p2-wide`

- real CIFAR-100 visual benchmark
- longer-sequence wider/deeper ViT
- accepted from landed low-learning-rate formal seeds:
  - `muon_ns`: `0.4660`, `0.4755`, `0.4738`
  - best landed `adamw`: `0.4562`

### `test21-vit-cifar10-p2-wide`

- real CIFAR-10 visual benchmark
- longer-sequence wider ViT
- accepted from landed low-learning-rate formal seeds:
  - `muon_ns`: `0.7486`, `0.7414`, `0.7532`
  - best landed `adamw`: `0.6923`

### `test29-stl10-vit`

- real STL10 visual benchmark at higher resolution
- accepted from landed low-learning-rate formal seeds:
  - `muon_ns`: `0.4876953125`, `0.48671875`, `0.4865234375`
  - landed `adamw`: `0.4357421875`, `0.4419921875`, `0.4623046875`

### `test26-vit-cifar100-p2-deeper`

- real CIFAR-100 visual benchmark
- accepted from the landed `lr=0.001` formal seeds:
  - `muon_ns`: `0.5205`, `0.5192`, `0.5277`
  - landed `adamw`: `0.417`, `0.4175`, `0.4271`

### `test30-tiny-imagenet-vit`

- real Tiny-ImageNet visual benchmark
- accepted from landed low-learning-rate formal seeds:
  - `muon_ns`: `0.2571`, `0.2634`, `0.2636`
  - landed `adamw`: `0.2513`, `0.2483`, `0.2542`

### `test33-stl10-vit-longer`

- real STL10 visual benchmark with longer optimization horizon
- accepted from landed gate-only seeds:
  - `muon_ns`: `0.526953125`, `0.5294921875`, `0.5220703125`,
    `0.57734375`, `0.5810546875`
  - strongest landed `adamw`: `0.498046875`

## Explicitly rejected examples

- `test10-pythia-tinystories`
- `test12-mlpmixer-cifar10`
- `test13-resmlp-cifar10`
- `test14-gpt2-tinystories`
- `test15-gpt2-tinystories-lrscan`
- `test17-vit-cifar100`
- `test18-gmlp-cifar10`
- `test20-vit-cifar10-p2`
- `test22-mlpmixer-cifar10-p2`
- `test25-vit-cifar100-p2`
- `test27-tabular-mlp-suite`
- `test28-svhn-vit`
