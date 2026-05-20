# test31-svhn-vit-p2-wide

Non-HamGNN visual benchmark using a stronger matrix-heavy ViT-style model on
real SVHN data.

## Why this exists

`test28-svhn-vit` already showed that a smaller SVHN ViT recipe does not pass
the user's gate. This retry keeps the dataset real and unchanged, but pushes
the model closer to the already successful CIFAR/Tiny-ImageNet family:

- `patch_size=2`
- wider embedding dimension
- deeper encoder

The goal is to test whether the accepted "matrix-heavier ViT" pattern transfers
to SVHN when the backbone is scaled into the same regime.

## Acceptance gate

This benchmark is only accepted if original `muon_ns` beats `adamw`.

## Planned methods

- `adamw`
- `muon_ns`
- `rt_v43_stream`
- `rt_v43_ns`
- `rt_v6_fdt_metric`

## Data

- dataset: SVHN
- cached server path:
  `/data/run01/scwb923/4.3-log/_datasets/svhn`
