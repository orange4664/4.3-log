# test21-vit-cifar10-p2-wide status

- benchmark type: non-HamGNN real-data vision benchmark
- model type: wider ViT with longer token sequence
- current state: formal acceptance established from landed low-learning-rate
  `muon_ns` seeds `0/1/2`
- acceptance gate: requires original `muon_ns` to beat `adamw`
- current evidence:
  - `muon_ns lr=0.0005 seed=0 best_test_acc = 0.7486`
  - `muon_ns lr=0.0005 seed=1 best_test_acc = 0.7414`
  - `muon_ns lr=0.0005 seed=2 best_test_acc = 0.7532`
  - best landed `adamw` formal result = `0.6923`
- interpretation:
  - formal acceptance is established
  - every landed low-learning-rate `muon_ns` formal summary is above every
    landed `adamw` formal summary
