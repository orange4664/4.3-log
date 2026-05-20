# test33-stl10-vit-longer

Non-HamGNN visual benchmark using a longer-training matrix-heavy ViT-style
model on real STL10 data.

## Why this exists

`test29-stl10-vit` already showed that STL10 can satisfy the benchmark gate
with a real `96x96` visual task. This follow-up keeps the same real dataset and
ViT family, but extends the training horizon so the optimizer comparison is not
limited to a shorter schedule.

This keeps the benchmark:

- real
- matrix-heavy
- medium scale
- harder than the accepted short STL10 recipe

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
