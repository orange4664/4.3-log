# test23-vit-cifar10-p2-deeper status

- benchmark type: non-HamGNN real-data vision benchmark
- model type: deeper compact ViT with longer token sequence
- current state: formal run active, three `muon_ns` formal summaries landed
- acceptance gate: requires original `muon_ns` to beat `adamw`
- current evidence:
  - `muon_ns lr=0.0005 seed=0 best_test_acc = 0.7352`
  - `muon_ns lr=0.0005 seed=1 best_test_acc = 0.7359`
  - `muon_ns lr=0.0005 seed=2 best_test_acc = 0.7371`
  - best landed `adamw` formal result = `0.7195`
- interpretation:
  - formal acceptance is established for the low-learning-rate block
  - all landed `muon_ns` low-learning-rate formal summaries are above every
    landed `adamw` formal summary
