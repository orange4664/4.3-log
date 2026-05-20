# test13-resmlp-cifar10

Non-HamGNN visual benchmark using a compact `ResMLP`-style architecture on real
CIFAR-10 data.

## Why this exists

This benchmark stays on real vision data but uses a second dense, matrix-heavy
model family distinct from ViT and MLP-Mixer:

- patch embedding projection
- token-mixing linear layers across patches
- channel-expansion MLP blocks
- no ordinary small convnet backbone

The goal is to test whether original `muon_ns` keeps its edge on another dense
vision architecture rather than only one model family.

## Acceptance gate

This benchmark is only accepted if original `muon_ns` beats `adamw`.

## Planned methods

- `adamw`
- `muon_ns`
- `rt_v43_stream`
- `rt_v43_ns`
- `rt_v6_fdt_metric`

## Data

- dataset: CIFAR-10
- server cache candidate:
  `/data/run01/scwb923/hyz/cifar10_optimizer_benchmark/data/cifar-10-batches-py`
