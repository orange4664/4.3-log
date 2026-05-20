# test24-vit-cifar100-p2-wide status

- benchmark type: harder non-HamGNN real-data vision benchmark
- model type: longer-sequence wider/deeper ViT
- current state: smoke accepted, formal acceptance established from landed
  `muon_ns` low-lr seeds `0/1/2`
- acceptance gate: requires original `muon_ns` to beat `adamw`
- current evidence:
  - `muon_ns lr=0.0005 seed=0 best_test_acc = 0.466`
  - `muon_ns lr=0.0005 seed=1 best_test_acc = 0.4755`
  - `muon_ns lr=0.0005 seed=2 best_test_acc = 0.4738`
  - best landed `adamw` formal result = `0.4562`
- interpretation:
  - formal acceptance is established
  - every landed low-lr `muon_ns` formal summary is above every landed `adamw`
    formal summary
