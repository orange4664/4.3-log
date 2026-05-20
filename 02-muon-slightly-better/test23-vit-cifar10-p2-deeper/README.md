# test23-vit-cifar10-p2-deeper

Non-HamGNN visual benchmark using a matrix-heavy ViT-style model on real
CIFAR-10 data.

## Why this exists

This benchmark extends `test20-vit-cifar10-p2` by keeping the longer token
sequence (`patch_size=2`) while increasing depth to `10`, testing a deeper
matrix-heavy ViT without widening the hidden size.

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
