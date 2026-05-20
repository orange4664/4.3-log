# test11-vit-cifar10

Non-HamGNN visual benchmark using a matrix-heavy ViT-style model on real
CIFAR-10 data.

## Why this exists

The earlier CIFAR benchmark in this workspace used a small convnet. The user
explicitly ruled out simple models unless they are intentionally matrix-heavy.
This test replaces the small convnet with a compact ViT so the optimizer sees
more large matrix updates.

## Acceptance gate

This benchmark is only accepted if original `muon_ns` beats `adamw`.

## Planned methods

- `adamw`
- `muon_ns`
- `rt_v43_stream`
- `rt_v43_ns`
- `rt_v6_fdt_metric`

## Data

- dataset: CIFAR-10
- server cache candidate:
  `/data/run01/scwb923/hyz/cifar10_optimizer_benchmark/data/cifar-10-batches-py`
