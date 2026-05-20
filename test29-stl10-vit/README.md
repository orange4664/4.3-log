# test29-stl10-vit

Non-HamGNN visual benchmark using a matrix-heavy ViT-style model on real
STL10 data.

## Why this exists

Most of the currently accepted non-HamGNN benchmarks in this search are based
on CIFAR-10/CIFAR-100. This benchmark adds a different real visual dataset with
higher image resolution:

- dataset: STL10
- image size: `96x96`
- model family: ViT

This keeps the benchmark real and matrix-heavy while avoiding an oversized
ImageNet-scale setup.

## Acceptance gate

This benchmark is only accepted if original `muon_ns` beats `adamw`.

## Planned methods

- `adamw`
- `muon_ns`
- `rt_v43_stream`
- `rt_v43_ns`
- `rt_v6_fdt_metric`

## Data

- dataset: STL10
- storage path on server:
  `/data/run01/scwb923/4.3-log/test29-stl10-vit/data`
