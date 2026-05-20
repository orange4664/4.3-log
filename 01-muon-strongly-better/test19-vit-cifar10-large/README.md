# test19-vit-cifar10-large

Non-HamGNN visual benchmark using a matrix-heavy ViT-style model on real
CIFAR-10 data.

## Why this exists

`test11-vit-cifar10` already showed that original `muon_ns` can beat `adamw` on
a compact ViT. This benchmark keeps the same real-data setup and makes the
model harder by scaling the ViT width/depth upward while staying small enough
for quick cluster screening.

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
