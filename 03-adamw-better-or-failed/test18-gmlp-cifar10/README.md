# test18-gmlp-cifar10

Non-HamGNN visual benchmark using a compact but matrix-heavy `gMLP` model
on real CIFAR-10 data.

## Why this exists

The user ruled out ordinary simple models. This benchmark intentionally favors a
matrix-dominant architecture with spatial gating:

- patch embedding projection
- dense feed-forward projections
- spatial-gating linear mixers across patch positions

Compared with a small convnet or a tiny PINN MLP, this setup gives Muon-family
optimizers many more large dense matrix updates in a third visual architecture
family.

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
