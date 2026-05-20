# test25-vit-cifar100-p2

Non-HamGNN visual benchmark using a longer-sequence ViT-style model on real
CIFAR-100 data.

## Why this exists

`test17-vit-cifar100` used the smallest compact CIFAR-100 ViT recipe and did
not pass the acceptance gate. This retry changes only one main structural
dimension first:

- `patch_size=2` for a longer token sequence

This is the closest CIFAR-100 analogue of the already smoke-accepted
`test20-vit-cifar10-p2` benchmark.

## Acceptance gate

This benchmark is only accepted if original `muon_ns` beats `adamw`.

## Planned methods

- `adamw`
- `muon_ns`
- `rt_v43_stream`
- `rt_v43_ns`
- `rt_v6_fdt_metric`

## Data

- dataset: CIFAR-100
- offline server path:
  `/data/run01/scwb923/4.3-log/test17-vit-cifar100/data`
