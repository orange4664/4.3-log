# test32-vit-cifar100-p2-longer

Non-HamGNN visual benchmark using a longer-training ViT-style model on real
CIFAR-100 data.

## Why this exists

Several accepted benchmarks already show that Muon-family methods like
matrix-heavy ViT training on real image data. This benchmark keeps a proven
real-model / real-data recipe but makes the optimization path longer, to test
whether the original `muon_ns` keeps an advantage once the schedule has more
time to separate stable matrix updates from AdamW.

This benchmark keeps:

- `patch_size=2`
- CIFAR-100
- ViT backbone

and changes the stress axis to:

- longer training horizon

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
