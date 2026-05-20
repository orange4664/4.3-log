# test14-gpt2-tinystories

Non-HamGNN language-model benchmark using a small GPT-2 style decoder trained
from scratch on real TinyStories text files already cached on the server.

## Why this exists

Muon-family papers most often succeed on dense transformer training rather than
small PDE MLPs. This benchmark stays close to that literature pattern while
keeping the scale moderate:

- decoder-only transformer
- real local text data
- from-scratch initialization
- no pretrained checkpoint dependence

## Acceptance gate

This benchmark is only accepted if original `muon_ns` beats `adamw`.

## Planned methods

- `adamw`
- `muon_ns`
- `rt_v43_stream`
- `rt_v43_ns`
- `rt_v6_fdt_metric`

## Data

- training text: `/data/run01/scwb923/hyz/pythia_optimizer_benchmark/data/train.txt`
- validation text: `/data/run01/scwb923/hyz/pythia_optimizer_benchmark/data/val.txt`
