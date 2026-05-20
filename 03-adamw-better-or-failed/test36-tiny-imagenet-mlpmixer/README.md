# test36-tiny-imagenet-mlpmixer

Non-HamGNN visual benchmark using a matrix-heavy MLP-Mixer model on real
Tiny-ImageNet data.

## Why this exists

Most of the stronger accepted non-HamGNN results in this search have come from
ViT-style models. This benchmark opens a harder real-data branch with a
different matrix-heavy architecture family while reusing the same practical
Tiny-ImageNet dataset.

Why this is useful:

- real `200`-class image classification
- no new external data dependency
- MLP-Mixer is highly matrix-dominated, which is relevant to Muon-family
  optimizers

## Acceptance gate

This benchmark is only accepted if original `muon_ns` beats `adamw`.

## Planned methods

- `adamw`
- `muon_ns`
- `rt_v43_stream`
- `rt_v43_ns`
- `rt_v6_fdt_metric`

## Data

- dataset: Tiny-ImageNet-200
- expected server path:
  `/data/run01/scwb923/4.3-log/test36-tiny-imagenet-mlpmixer/data/tiny-imagenet-200`
- reused source archive:
  `tiny-imagenet-200.zip`
