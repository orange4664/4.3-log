# test35-tiny-imagenet-vit-deeper

Non-HamGNN visual benchmark using a stronger matrix-heavy ViT-style model on
real Tiny-ImageNet data.

## Why this exists

`test30-tiny-imagenet-vit` already showed an accepted Tiny-ImageNet result.
This follow-up pushes the same real-data family to a harder setting without
introducing new external data dependencies.

Harder axes:

- deeper ViT backbone
- wider embedding dimension
- longer optimization horizon than the earlier smoke setup

This keeps the benchmark real, matrix-heavy, and cluster-practical.

## Acceptance gate

This benchmark is only accepted if original `muon_ns` beats `adamw`.

## Original accepted gate methods

- `adamw`
- `muon_ns`

## Five-optimizer follow-up benchmark

The follow-up benchmark requested after choosing `test35` compares:

- `adamw`
- `muon_ns`
- `muon_adamw`
- `rt_v43_stream`
- `rt_v43_adamw`

Here `muon_adamw` is the HamGNN-style Muon+AdamW hybrid: Muon-friendly matrix
parameters use `muon_ns`, while the remaining parameters use AdamW.
`rt_v43_adamw` applies the same hybrid split to the v4.3 streaming RT
optimizer.

Run script:

```bash
sbatch scripts/submit_test35_five_optimizers_h100.sbatch
```

Analysis output:

- `FIVE_OPTIMIZER_BENCHMARK_ANALYSIS.md`
- `runs_five_optimizers/five_optimizer_summary.csv`

## Data

- dataset: Tiny-ImageNet-200
- expected server path:
  `/data/run01/scwb923/4.3-log/test35-tiny-imagenet-vit-deeper/data/tiny-imagenet-200`
- reused source archive:
  `tiny-imagenet-200.zip`
