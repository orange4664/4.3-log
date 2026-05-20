# test24-vit-cifar100-p2-wide

Non-HamGNN visual benchmark using a stronger matrix-heavy ViT-style model on
real CIFAR-100 data.

## Why this exists

The accepted CIFAR-10 ViT benchmarks showed that original `muon_ns` can beat
`adamw` on real matrix-heavy visual models. `test17-vit-cifar100` kept the
small compact recipe and failed the user's gate on CIFAR-100.

This test keeps the benchmark real and non-toy, but pushes the model further
toward the matrix-heavy regime:

- `patch_size=2` for a longer token sequence
- wider embedding dimension
- deeper encoder

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
