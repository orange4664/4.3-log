# test20-vit-cifar10-p2

Non-HamGNN visual benchmark using a matrix-heavy ViT-style model on real
CIFAR-10 data.

## Why this exists

`test11-vit-cifar10` already passed the user's gate. This benchmark keeps the
same real CIFAR-10 data and compact ViT family but makes the sequence much
longer by reducing patch size from `4` to `2`, increasing the number of visual
tokens from `64` to `256`.

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
