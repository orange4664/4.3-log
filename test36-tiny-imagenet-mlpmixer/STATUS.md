# test36-tiny-imagenet-mlpmixer status

- benchmark type: harder non-HamGNN real-data vision benchmark
- model type: MLP-Mixer on Tiny-ImageNet
- current state:
  - scaffolded locally
  - synced to cluster path `/data/run01/scwb923/4.3-log/test36-tiny-imagenet-mlpmixer`
  - smoke submitted on `gpu_h100` as job `73516`
- acceptance gate: requires original `muon_ns` to beat `adamw`
- next action:
  - wait for smoke summaries from `adamw` and `muon_ns`
  - promote only if smoke keeps `muon_ns > adamw`
