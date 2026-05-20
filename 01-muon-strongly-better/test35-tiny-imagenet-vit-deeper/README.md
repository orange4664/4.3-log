# test35-tiny-imagenet-vit-deeper

Non-HamGNN visual benchmark using a stronger matrix-heavy ViT-style model on
real Tiny-ImageNet data.

## Why this exists

`test30-tiny-imagenet-vit` already showed an accepted Tiny-ImageNet result.
This follow-up pushes the same real-data family to a harder setting without
introducing new external data dependencies.

Harder axes:

- deeper ViT backbone
- wider embedding dimension
- longer optimization horizon than the earlier smoke setup

This keeps the benchmark real, matrix-heavy, and cluster-practical.

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
  `/data/run01/scwb923/4.3-log/test35-tiny-imagenet-vit-deeper/data/tiny-imagenet-200`
- reused source archive:
  `tiny-imagenet-200.zip`
