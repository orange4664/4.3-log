# test27-tabular-mlp-suite status

This folder is reserved for a real tabular deep learning benchmark suite using
a matrix-heavy MLP backbone.

Acceptance rule:

- only accept this benchmark if original `muon_ns` beats `adamw`

Current contents:

- `models.py`
- `train_tabular_mlp_suite.py`
- `scripts/submit_tabular_mlp_suite_smoke_a800.sbatch`
- `scripts/submit_tabular_mlp_suite_a800.sbatch`
- `scripts/submit_tabular_mlp_suite_smoke_h100.sbatch`
- `scripts/submit_tabular_mlp_suite_h100.sbatch`

Planned datasets:

- `breast_cancer`
- `wine`
- `digits`

Results and analysis will be added after the completed server run.

Smoke run status:

- server: `nmcc-n46h1`
- job id: `73422`
- status: completed enough to analyze smoke outputs
- result: smoke rejected under the user's gate
