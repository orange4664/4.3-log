# test27-tabular-mlp-suite

This benchmark adds a non-HamGNN, non-toy, real-data MLP suite for tabular
learning.

Why this direction:

- recent public Muon benchmark work explicitly reports strong Muon results on
  tabular deep learning MLPs
- the backbone is matrix-heavy by construction
- the datasets are real supervised datasets shipped directly inside
  `scikit-learn`, so the suite can run fully on the server without external
  downloads

## Datasets

- `breast_cancer`
- `wine`
- `digits`

These are not synthetic datasets and they are not toy matrix surrogates. The
model itself is intentionally a deeper, wider MLP so the optimizer comparison
stays focused on matrix updates rather than tiny shallow baselines.

## Model

- deep tabular MLP
- configurable width/depth
- GELU activations
- LayerNorm + dropout

## Optimizers

- `adamw`
- `muon_ns`
- `rt_v43_stream`
- `rt_v43_ns`

Acceptance rule for this benchmark family:

- count this benchmark as good only if original `muon_ns` beats `adamw`

## Aggregation rule

For each dataset and optimizer:

1. average each learning rate across seeds
2. choose the learning rate with the best mean validation accuracy
3. compare optimizers using the corresponding mean test accuracy at the best
   validation checkpoint

The suite summary then averages the chosen per-dataset test accuracies across
all datasets and records dataset-level win counts.

## Files

- `models.py`
- `train_tabular_mlp_suite.py`
- `scripts/submit_tabular_mlp_suite_smoke_a800.sbatch`
- `scripts/submit_tabular_mlp_suite_a800.sbatch`
- `STATUS.md`
- `ANALYSIS.md`
