# test10-pythia-tinystories

Non-HamGNN Muon-family benchmark based on a small real LLM training setup:

- model family: `Pythia-160M`
- task: causal LM training
- data: local TinyStories text files on the server
- acceptance gate: this benchmark is only accepted if original `muon_ns` beats `adamw`

This folder is the first non-HamGNN replacement benchmark after the user ruled
out HamGNN as an acceptable candidate.

## Design constraints

- real text data only; no synthetic fallback is allowed
- model must start from config initialization, not pretrained weights
- benchmark size should stay tractable on one A800

## Expected server assets

- training text: `/data/run01/scwb923/hyz/pythia_optimizer_benchmark/data/train.txt`
- validation text: `/data/run01/scwb923/hyz/pythia_optimizer_benchmark/data/val.txt`

## Methods for the first pass

- `adamw`
- `muon_ns`
- `rt_v43_stream`
- `rt_v43_ns`
- `rt_v6_fdt_metric`

The benchmark is only considered "good" if `muon_ns` beats `adamw`.
