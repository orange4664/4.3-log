# test26-vit-cifar100-p2-deeper status

- benchmark type: harder non-HamGNN real-data vision benchmark
- model type: longer-sequence deeper ViT
- current state: formal acceptance established from the landed `lr=0.001`
  multi-seed block
- acceptance gate: requires original `muon_ns` to beat `adamw`
- current evidence:
  - smoke:
    - `adamw lr=0.0005 seed=0 best_test_acc = 0.205078125`
    - `muon_ns lr=0.0005 seed=0 best_test_acc = 0.2236328125`
  - formal `adamw` so far:
    - `seed=0 best_test_acc = 0.4495`
    - `seed=1 best_test_acc = 0.4470`
    - `seed=2 best_test_acc = 0.4440`
  - formal `muon_ns` so far:
    - `seed=0 best_test_acc = 0.4472`
    - `seed=1 best_test_acc = 0.4482`
    - `seed=2 best_test_acc = 0.4546`
  - formal `adamw` at `lr=0.001`:
    - `seed=0 best_test_acc = 0.417`
    - `seed=1 best_test_acc = 0.4175`
    - `seed=2 best_test_acc = 0.4271`
  - formal `muon_ns` at `lr=0.001`:
    - `seed=0 best_test_acc = 0.5205`
    - `seed=1 best_test_acc = 0.5192`
    - `seed=2 best_test_acc = 0.5277`
- interpretation:
  - smoke gate passed
  - promoted to formal run as job `73425`
  - the landed `lr=0.0005` formal block is mixed
  - the landed `lr=0.001` formal block is decisively favorable to `muon_ns`
  - this benchmark now satisfies the benchmark-selection gate and should be
    counted as accepted
