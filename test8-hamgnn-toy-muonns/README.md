# test8-hamgnn-toy-muonns

This folder contains a server-run HamGNN toy Hamiltonian benchmark intended to
screen for a benchmark that satisfies the user's acceptance rule:

- accept only if original `muon_ns` beats `adamw`

## Why this benchmark

Historical server artifacts in `hamgnn_workspace/hamgnn_toy_ablation_muonns`
already suggested:

- `muon_ns` slightly beats `adamw` on toy Hamiltonian MAE

So this is a high-priority benchmark to rerun cleanly under the current
repository-tracked workflow.

## Task

- Model family: HamGNN
- Data: toy Hamiltonian graph dataset
- Main metric: `test_mae`

## Methods

- `adamw`
- `muon_ns`
- `rt_v43_stream`
- `rt_v6_fdt_metric`

## Server dependency

This benchmark reuses the existing server-side toy graph dataset at:

```text
/data/run01/scwb923/hamgnn_workspace/hamgnn_toy_ablation_muonns/data
```

The benchmark code, configs, logs, and results live under this repository
folder; only the dataset path is external.
