# test32-vit-cifar100-p2-longer status

- benchmark type: harder non-HamGNN real-data vision benchmark
- model type: longer-training ViT on CIFAR-100 with `patch_size=2`
- current state:
  - smoke completed on `gpu_h100`
  - older formal submitted on `gpu_h100` as job `73437`
  - gate-only rerun running on `gpu_h100` as job `73441`
  - gate-only rerun `73441` was cancelled by Slurm due to node failure on
    `d1n41e21g01`
  - gate-only rerun resubmitted with node exclusion as job `73491`
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
- next action:
  - wait for rerun `muon_ns` summaries from job `73491`
  - compare the landed `muon_ns` block against the gate-only `adamw` baseline
    before accepting or rejecting
