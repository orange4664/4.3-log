# test30-tiny-imagenet-vit

Non-HamGNN visual benchmark using a matrix-heavy ViT-style model on real
Tiny-ImageNet data.

## Why this exists

The accepted CIFAR and STL10 benchmarks show that `muon_ns` can beat `adamw`
on real visual workloads. The next step should be harder without becoming a
full ImageNet-scale project.

Tiny-ImageNet fits that slot:

- real image classification benchmark
- `200` classes
- `64x64` resolution
- much harder than CIFAR/STL10
- still small enough to run on one cluster GPU

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
  `/data/run01/scwb923/4.3-log/test30-tiny-imagenet-vit/data/tiny-imagenet-200`
- official source used by the prep script:
  `https://cs231n.stanford.edu/tiny-imagenet-200.zip`
