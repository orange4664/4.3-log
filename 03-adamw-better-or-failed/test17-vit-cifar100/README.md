# test17-vit-cifar100

Non-HamGNN visual benchmark using a matrix-heavy ViT-style model on real
CIFAR-100 data.

## Why this exists

The accepted CIFAR-10 ViT benchmark showed that original `muon_ns` can beat
`adamw` on a real matrix-heavy visual model. This test makes the task harder by
keeping the compact ViT architecture and upgrading the data from CIFAR-10 to
CIFAR-100.

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
- download target on server:
  `/data/run01/scwb923/4.3-log/test17-vit-cifar100/data`
