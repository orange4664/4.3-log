# test26-vit-cifar100-p2-deeper

Non-HamGNN visual benchmark using a deeper longer-sequence ViT-style model on
real CIFAR-100 data.

## Why this exists

We now have two CIFAR-100 retries with different outcomes:

- `test24-vit-cifar100-p2-wide`: smoke accepted
- `test25-vit-cifar100-p2`: smoke rejected

This benchmark isolates another structural axis from the successful CIFAR-10
family:

- `patch_size=2`
- deeper encoder

It is the CIFAR-100 counterpart of `test23-vit-cifar10-p2-deeper`.

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
