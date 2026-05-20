# test34-flowers102-vit

Non-HamGNN visual benchmark using a matrix-heavy ViT-style model on real
Flowers102 data.

## Why this exists

This benchmark pushes the accepted ViT line to a more realistic medium-scale
image dataset without jumping all the way to ImageNet.

- dataset: Flowers102
- image size: `224x224`
- model family: ViT

This keeps the benchmark real and matrix-heavy while remaining practical on the
current cluster budget.

## Acceptance gate

This benchmark is only accepted if original `muon_ns` beats `adamw`.

## Planned methods

- `adamw`
- `muon_ns`
- `rt_v43_stream`
- `rt_v43_ns`
- `rt_v6_fdt_metric`

## Data

- dataset: Flowers102 from `torchvision.datasets.Flowers102`
- storage path on server:
  `/data/run01/scwb923/4.3-log/test34-flowers102-vit/data`
- note:
  prefetch the dataset before submitting training jobs; the cluster training job
  should not rely on online download
