# Accepted Non-HamGNN Benchmarks Snapshot

This file records the currently verified accepted non-HamGNN benchmarks from
the ongoing Muon benchmark search.

Acceptance rule:

- a benchmark is accepted only if original `muon_ns` beats `adamw`

Path convention:

- strong wins are under `01-muon-strongly-better/`
- smaller wins are under `02-muon-slightly-better/`

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

### `test31-svhn-vit-p2-wide`

- real SVHN visual benchmark with a stronger ViT retry
- accepted from landed rerun seeds:
  - `muon_ns`: `0.9234375`, `0.91435546875`, `0.91787109375`,
    `0.94619140625`, `0.93984375`, `0.94189453125`
  - strongest landed `adamw`: `0.859765625`

### `test32-vit-cifar100-p2-longer`

- real CIFAR-100 visual benchmark with longer training
- accepted from the landed `lr=0.001` rerun block:
  - `muon_ns`: `0.5277`, `0.5258`, `0.5332`
  - strongest landed `adamw`: `0.4993`

### `test35-tiny-imagenet-vit-deeper`

- real Tiny-ImageNet-200 ViT benchmark, deeper/wider than `test30`
- best `adamw`: `0.282682`
- best `muon_ns`: `0.414323`
- best absolute gain: `+0.131641`, about `+13.16` percentage points
- best relative gain: about `+46.6%`
- mean absolute gain: about `+12.14` percentage points

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
- `test36-tiny-imagenet-mlpmixer`
