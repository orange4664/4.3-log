# test28-svhn-vit status

This folder is reserved for a non-HamGNN visual benchmark on real SVHN data.

Acceptance rule:

- only accept this benchmark if original `muon_ns` beats `adamw`

Current contents:

- `models.py`
- `train_vit_svhn.py`
- `scripts/submit_vit_svhn_smoke_a800.sbatch`
- `scripts/submit_vit_svhn_smoke_h100.sbatch`
- `scripts/submit_vit_svhn_a800.sbatch`

Results and analysis will be added after the completed server run.

Current server note:

- first smoke job `73423` failed on dataset download timeout
- SVHN has now been pre-downloaded to:
  - `/data/run01/scwb923/4.3-log/_datasets/svhn`
- smoke rerun submitted as `73424`
- current rerun state: smoke completed enough to analyze
- rerun result: smoke rejected under the user's gate
