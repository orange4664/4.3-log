# test30-tiny-imagenet-vit status

- benchmark type: harder non-HamGNN real-data vision benchmark
- model type: ViT on Tiny-ImageNet `64x64`
- current state: accepted from the landed low-learning-rate formal block; job
  `73432` is still running later blocks
- acceptance gate: requires original `muon_ns` to beat `adamw`
- current evidence:
  - landed `adamw` smoke summary:
    - `best_test_acc = 0.0825`
  - landed `muon_ns` smoke summary:
    - `best_test_acc = 0.08859375`
  - landed formal `adamw` low-learning-rate block:
    - `seed=0 best_test_acc = 0.2513`
    - `seed=1 best_test_acc = 0.2483`
    - `seed=2 best_test_acc = 0.2542`
  - landed formal `muon_ns` low-learning-rate block:
    - `seed=0 best_test_acc = 0.2571`
    - `seed=1 best_test_acc = 0.2634`
    - `seed=2 best_test_acc = 0.2636`
- interpretation:
  - smoke gate passed because `0.08859375 > 0.0825`
  - every landed low-learning-rate `muon_ns` seed is above the strongest
    landed low-learning-rate `adamw` seed
  - accepted under the benchmark-selection gate
