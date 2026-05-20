# test21-vit-cifar10-p2-wide

Non-HamGNN visual benchmark using a matrix-heavy ViT-style model on real
CIFAR-10 data.

## Why this exists

This benchmark extends `test20-vit-cifar10-p2` by combining the longer token
sequence (`patch_size=2`) with a wider ViT (`embed_dim=384`, `num_heads=12`),
to test whether Muon's advantage survives in a more matrix-heavy long-sequence
setting.

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
