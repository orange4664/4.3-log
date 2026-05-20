# test28-svhn-vit

Non-HamGNN visual benchmark using a matrix-heavy ViT-style model on real SVHN
data.

## Why this exists

The CIFAR-10 ViT line already gave several accepted Muon benchmarks. This test
moves to a different real visual distribution:

- street-view house numbers
- real images
- still small enough to run on the cluster without turning into a giant
  benchmark
- same matrix-heavy ViT backbone family, so optimizer differences remain easy
  to attribute

## Acceptance gate

This benchmark is only accepted if original `muon_ns` beats `adamw`.

## Planned methods

- `adamw`
- `muon_ns`
- `rt_v43_stream`
- `rt_v43_ns`

## Data

- dataset: SVHN
- source: `torchvision.datasets.SVHN`
- server download probe already succeeded
