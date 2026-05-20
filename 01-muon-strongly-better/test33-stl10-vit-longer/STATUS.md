# test33-stl10-vit-longer status

- benchmark type: harder non-HamGNN real-data vision benchmark
- model type: longer-training ViT on higher-resolution STL10
- current state:
  - smoke completed on `gpu_h100`
  - formal submitted on `gpu_h100` as job `73438`
  - gate-only rerun still running on `gpu_h100` as job `73442`
- acceptance gate: requires original `muon_ns` to beat `adamw`
- current evidence:
  - smoke:
    - `adamw lr=0.0005 seed=0 best_test_acc = 0.348828125`
    - `muon_ns lr=0.0005 seed=0 best_test_acc = 0.37265625`
  - landed gate-only `adamw`:
    - `lr=0.0005`: `0.497265625`, `0.498046875`, `0.4974609375`
    - `lr=0.001`: `0.4439453125`, `0.425`, `0.4181640625`
  - landed gate-only `muon_ns`:
    - `lr=0.0005`: `0.526953125`, `0.5294921875`, `0.5220703125`
    - `lr=0.001`: `0.57734375`, `0.5810546875`
- verdict:
  - accepted under the user's gate based on landed gate-only evidence
  - every landed `muon_ns` value is above the strongest landed `adamw` value
- next action:
  - let `73442` and `73438` finish for completeness
  - keep the acceptance verdict unless a later landed `muon_ns` run drops below
    the strongest landed `adamw` seed
