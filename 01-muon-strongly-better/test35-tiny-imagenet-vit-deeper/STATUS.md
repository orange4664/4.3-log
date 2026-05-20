# test35-tiny-imagenet-vit-deeper status

- benchmark type: harder non-HamGNN real-data vision benchmark
- model type: deeper and wider ViT on Tiny-ImageNet
- current state:
  - scaffolded locally
  - original run synced to cluster path `/data/run01/scwb923/4.3-log/test35-tiny-imagenet-vit-deeper`
  - smoke submitted on `gpu_h100` as job `73497`
  - smoke completed on cluster node `d1n41d29g01`
  - first landed smoke summary:
    - `adamw lr=0.0005 seed=0 best_test_acc = 0.08375`
    - `muon_ns lr=0.0005 seed=0 best_test_acc = 0.13916666666666666`
  - formal submitted on `gpu_h100` as job `73503`
  - formal is now running on cluster node `d1n41d29g01`
  - first landed formal summaries:
    - `adamw lr=0.0005 seed=0 best_test_acc = 0.28033854166666666`
    - `adamw lr=0.0005 seed=1 best_test_acc = 0.27239583333333334`
    - `adamw lr=0.0005 seed=2 best_test_acc = 0.28268229166666666`
    - `adamw lr=0.001 seed=0 best_test_acc = 0.21328125`
    - `adamw lr=0.001 seed=1 best_test_acc = 0.21692708333333333`
    - `adamw lr=0.001 seed=2 best_test_acc = 0.22838541666666667`
  - first landed formal `muon_ns` summary:
    - `muon_ns lr=0.0005 seed=0 best_test_acc = 0.3326822916666667`
    - `muon_ns lr=0.0005 seed=1 best_test_acc = 0.33151041666666664`
    - `muon_ns lr=0.0005 seed=2 best_test_acc = 0.3359375`
    - `muon_ns lr=0.001 seed=0 best_test_acc = 0.408203125`
    - `muon_ns lr=0.001 seed=1 best_test_acc = 0.399609375`
    - `muon_ns lr=0.001 seed=2 best_test_acc = 0.41432291666666665`
  - first landed post-`muon_ns` formal summary:
    - `rt_v43_stream lr=0.0005 seed=0 best_test_acc = 0.31940104166666666`
    - `steps = 6246`
    - `time_sec = 2527.1007554531097`
- acceptance gate: requires original `muon_ns` to beat `adamw`
- next action:
  - smoke gate passed
  - every landed formal `muon_ns` value is above every landed formal `adamw`
    value
  - best `adamw = 0.28268229166666666`
  - best `muon_ns = 0.41432291666666665`
  - best absolute gain is `+0.131640625`, about `+13.16` percentage points
  - best relative gain is about `+46.6%`
  - mean `adamw = 0.2490017361111111`
  - mean `muon_ns = 0.37037760416666664`
  - mean absolute gain is `+0.12137586805555554`, about `+12.14` percentage
    points
  - mean relative gain is about `+48.7%`
  - the original `muon_ns` formal sweep is now complete
  - the first `rt_v43_stream lr=0.0005 seed=0` formal block has also completed
  - the remaining `rt_v43_stream / rt_v43_ns / rt_v6_fdt_metric` comparison
    was manually stopped at the user's request
  - accepted conclusion for this folder is only `muon_ns > adamw`; it should
    not be used as a completed `rt_v43_*` comparison
