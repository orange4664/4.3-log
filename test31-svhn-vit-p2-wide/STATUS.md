# test31-svhn-vit-p2-wide status

- benchmark type: harder non-HamGNN real-data vision benchmark
- model type: stronger SVHN ViT with `patch_size=2`
- current state:
  - smoke completed on `gpu_h100` as job `73433`
  - older formal submitted on `gpu_h100` as job `73434`
  - gate-only rerun was first submitted on `gpu_h100` as job `73440`
  - gate-only rerun `73440` was cancelled by Slurm due to node failure on
    `d1n41e21g01`
  - gate-only rerun resubmitted with node exclusion as job `73490`
  - `73490` is currently running on `d1n41e21g02`
- acceptance gate: requires original `muon_ns` to beat `adamw`
- current evidence:
  - landed `adamw` smoke summary:
    - `best_test_acc = 0.484375`
  - landed `muon_ns` smoke summary:
    - `best_test_acc = 0.5533203125`
  - first landed gate-only summary:
    - `adamw lr=0.0005 seed=0 best_test_acc = 0.83662109375`
    - `adamw lr=0.0005 seed=1 best_test_acc = 0.80537109375`
    - `adamw lr=0.0005 seed=2 best_test_acc = 0.859765625`
  - landed rerun gate-only summary:
    - `adamw lr=0.0005 seed=1 best_test_acc = 0.8296875`
    - `adamw lr=0.0005 seed=2 best_test_acc = 0.7994140625`
    - `adamw lr=0.001 seed=0 best_test_acc = 0.75400390625`
- next action:
  - wait for rerun `muon_ns` summaries from job `73490`
  - compare the landed `muon_ns` block against the much stronger gate-only
    `adamw` baseline before accepting or rejecting
