# test29-stl10-vit status

- benchmark type: non-HamGNN real-data vision benchmark
- model type: ViT on higher-resolution STL10
- current state: formal low-learning-rate evidence accepted; job `73428` is
  still running later blocks
- acceptance gate: requires original `muon_ns` to beat `adamw`
- current evidence:
  - smoke:
    - `adamw lr=0.0005 seed=0 best_test_acc = 0.303515625`
    - `muon_ns lr=0.0005 seed=0 best_test_acc = 0.318359375`
  - formal `adamw` low-learning-rate block:
    - `seed=0 best_test_acc = 0.4357421875`
    - `seed=1 best_test_acc = 0.4419921875`
    - `seed=2 best_test_acc = 0.4623046875`
  - formal `muon_ns` low-learning-rate block:
    - `seed=0 best_test_acc = 0.4876953125`
    - `seed=1 best_test_acc = 0.48671875`
    - `seed=2 best_test_acc = 0.4865234375`
- interpretation:
  - smoke gate passed
  - promoted to formal run as job `73428`
  - all landed low-learning-rate `muon_ns` seeds are above the strongest landed
    low-learning-rate `adamw` seed
  - accepted under the benchmark-selection gate
