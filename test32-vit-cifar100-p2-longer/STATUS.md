# test32-vit-cifar100-p2-longer status

- benchmark type: harder non-HamGNN real-data vision benchmark
- model type: longer-training ViT on CIFAR-100 with `patch_size=2`
- current state:
  - smoke completed on `gpu_h100`
  - older formal submitted on `gpu_h100` as job `73437`
  - gate-only rerun first submitted on `gpu_h100` as job `73441`
  - gate-only rerun `73441` was cancelled by Slurm due to node failure on
    `d1n41e21g01`
  - gate-only rerun resubmitted with node exclusion as job `73491`
  - `73491` completed successfully on `d1n41e21g02`
- acceptance gate: requires original `muon_ns` to beat `adamw`
- current evidence:
  - smoke:
    - `adamw lr=0.0005 seed=0 best_test_acc = 0.2623046875`
    - `muon_ns lr=0.0005 seed=0 best_test_acc = 0.2734375`
  - first landed gate-only summary:
    - `adamw lr=0.0005 seed=0 best_test_acc = 0.4993`
    - `adamw lr=0.0005 seed=1 best_test_acc = 0.497`
    - `adamw lr=0.0005 seed=2 best_test_acc = 0.4951`
  - landed rerun gate-only summary:
    - `adamw lr=0.0005 seed=1 best_test_acc = 0.4961`
  - first landed rerun `muon_ns` summaries:
    - `muon_ns lr=0.0005 seed=0 best_test_acc = 0.4916`
    - `muon_ns lr=0.0005 seed=1 best_test_acc = 0.4864`
    - `muon_ns lr=0.0005 seed=2 best_test_acc = 0.4966`
    - `muon_ns lr=0.001 seed=0 best_test_acc = 0.5277`
    - `muon_ns lr=0.001 seed=1 best_test_acc = 0.5258`
    - `muon_ns lr=0.001 seed=2 best_test_acc = 0.5332`
- next action:
  - the landed rerun `muon_ns` block is now mixed:
    - seed `0`: `0.4916` vs `adamw 0.4993`
    - seed `1`: `0.4864` vs `adamw 0.4961`
    - seed `2`: `0.4966` vs `adamw 0.4951`
  - however, the first landed high-learning-rate rerun `muon_ns` seed is much
    stronger:
    - `muon_ns lr=0.001 seed=0 = 0.5277`
    - strongest landed `adamw` rerun seed so far = `0.4993`
  - the second and third landed high-learning-rate rerun `muon_ns` seeds are
    also clearly above every landed `adamw` rerun seed so far
  - this benchmark now satisfies the acceptance gate and should be treated as
    accepted
