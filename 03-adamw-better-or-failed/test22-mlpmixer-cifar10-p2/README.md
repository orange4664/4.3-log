# test22-mlpmixer-cifar10-p2

Non-HamGNN visual benchmark using a compact but matrix-heavy `MLP-Mixer` model
on real CIFAR-10 data.

## Why this exists

This benchmark keeps the same MLP-Mixer family but makes the token-mixing
matrix much larger by reducing patch size from `4` to `2`. That increases the
number of visual patches from `64` to `256` and directly hardens the token
mixer workload.

The user ruled out ordinary simple models. This benchmark intentionally favors a
matrix-dominant architecture:

- patch embedding projection
- token-mixing MLPs across patches
- channel-mixing MLPs across embedding dimensions

Compared with a small convnet or a tiny PINN MLP, this setup gives Muon-family
optimizers many more large dense matrix updates.

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
