# test36-tiny-imagenet-mlpmixer status

- benchmark type: harder non-HamGNN real-data vision benchmark
- model type: MLP-Mixer on Tiny-ImageNet
- current state:
  - scaffolded locally
  - synced to cluster path `/data/run01/scwb923/4.3-log/test36-tiny-imagenet-mlpmixer`
  - smoke submitted on `gpu_h100` as job `73516`
  - `73516` completed successfully on `d1n41e21g02`
  - landed smoke summaries:
    - `adamw lr=0.0005 seed=0 best_test_acc = 0.13020833333333334`
    - `muon_ns lr=0.0005 seed=0 best_test_acc = 0.12145833333333333`
- acceptance gate: requires original `muon_ns` to beat `adamw`
- next action:
  - smoke failed the gate because `muon_ns < adamw`
  - do not promote this benchmark to formal
