# Non-HamGNN Benchmark Summary

This document summarizes the non-HamGNN benchmark search requested in this
thread.

Acceptance gate for every benchmark:

- a benchmark is only counted as good if original `muon_ns` beats `adamw`

Repository grouping after cleanup:

- `01-muon-strongly-better/`: large or very robust `muon_ns` wins
- `02-muon-slightly-better/`: accepted but smaller `muon_ns` wins
- `03-adamw-better-or-failed/`: AdamW better, failed gate, or insufficient
  evidence
- `04-archive-hamgnn-pinn-and-incomplete/`: HamGNN/PINN/incomplete archive

## Accepted

### `test11-vit-cifar10`

- real CIFAR-10 visual benchmark
- compact ViT
- accepted under smoke and formal evidence

### `test16-mlpmixer-cifar10-refine`

- real CIFAR-10 visual benchmark
- refined compact MLP-Mixer
- accepted under smoke and formal evidence

### `test19-vit-cifar10-large`

- real CIFAR-10 visual benchmark
- wider/deeper ViT
- formal `muon_ns` best accuracies:
  - `0.7316`
  - `0.7317`
  - `0.7322`
- best landed `adamw` accuracy:
  - `0.6642`

### `test23-vit-cifar10-p2-deeper`

- real CIFAR-10 visual benchmark
- longer-sequence deeper ViT
- formal `muon_ns` best accuracies:
  - `0.7352`
  - `0.7359`
  - `0.7371`
- best landed `adamw` formal accuracy:
  - `0.7195`

### `test24-vit-cifar100-p2-wide`

- real CIFAR-100 visual benchmark
- longer-sequence wider/deeper ViT
- formal `muon_ns` best accuracies:
  - `0.4660`
  - `0.4755`
  - `0.4738`
- best landed `adamw` formal accuracy:
  - `0.4562`

### `test21-vit-cifar10-p2-wide`

- real CIFAR-10 visual benchmark
- longer-sequence wider ViT
- formal `muon_ns` best accuracies:
  - `0.7486`
  - `0.7414`
  - `0.7532`
- best landed `adamw` formal accuracy:
  - `0.6923`

### `test29-stl10-vit`

- real visual benchmark on STL10
- matrix-heavy ViT on `96x96` images
- added to move beyond CIFAR while staying far below ImageNet-scale cost
- formal low-learning-rate `adamw` best accuracies:
  - `0.4357421875`
  - `0.4419921875`
  - `0.4623046875`
- formal low-learning-rate `muon_ns` best accuracies:
  - `0.4876953125`
  - `0.48671875`
  - `0.4865234375`
- accepted because all landed low-learning-rate `muon_ns` seeds are above the
  strongest landed low-learning-rate `adamw` seed

### `test26-vit-cifar100-p2-deeper`

- real CIFAR-100 visual benchmark
- longer-sequence deeper ViT
- landed `lr=0.0005` formal block was mixed
- landed `lr=0.001` formal `adamw` best accuracies:
  - `0.417`
  - `0.4175`
  - `0.4271`
- landed `lr=0.001` formal `muon_ns` best accuracies:
  - `0.5205`
  - `0.5192`
  - `0.5277`
- accepted because the fully landed `lr=0.001` formal block clearly favors
  original `muon_ns`

### `test30-tiny-imagenet-vit`

- real Tiny-ImageNet visual benchmark
- matrix-heavy ViT on `64x64` images with `200` classes
- added to move beyond STL10 while staying below full ImageNet-scale cost
- smoke passed:
  - `adamw best_test_acc = 0.0825`
  - `muon_ns best_test_acc = 0.08859375`
- landed low-learning-rate formal `adamw` best accuracies:
  - `0.2513`
  - `0.2483`
  - `0.2542`
- landed low-learning-rate formal `muon_ns` best accuracies:
  - `0.2571`
  - `0.2634`
  - `0.2636`
- accepted because every landed low-learning-rate `muon_ns` seed is above the
  strongest landed low-learning-rate `adamw` seed

### `test33-stl10-vit-longer`

- real STL10 visual benchmark
- longer-training ViT on `96x96` images
- added as a harder follow-up to the already accepted `test29-stl10-vit`
- landed gate-only `adamw` best accuracies:
  - `0.497265625`
  - `0.498046875`
  - `0.4974609375`
  - `0.4439453125`
  - `0.425`
  - `0.4181640625`
- landed gate-only `muon_ns` best accuracies:
  - `0.526953125`
  - `0.5294921875`
  - `0.5220703125`
  - `0.57734375`
  - `0.5810546875`
- accepted because every landed `muon_ns` value is above the strongest landed
  `adamw` value `0.498046875`

### `test31-svhn-vit-p2-wide`

- real visual benchmark on SVHN
- stronger matrix-heavy ViT retry with `patch_size=2`, `embed_dim=256`,
  `depth=10`
- added because the weaker SVHN ViT recipe in `test28` failed the gate
- smoke passed:
  - `adamw best_test_acc = 0.484375`
  - `muon_ns best_test_acc = 0.5533203125`
- rerun job `73490` completed successfully after the earlier node-failure retry
- landed gate-only `adamw` best accuracies:
  - `0.83662109375`
  - `0.80537109375`
  - `0.859765625`
  - `0.8296875`
  - `0.7994140625`
  - `0.75400390625`
- landed rerun `muon_ns` best accuracies:
  - `0.9234375`
  - `0.91435546875`
  - `0.91787109375`
  - `0.94619140625`
  - `0.93984375`
  - `0.94189453125`
- accepted because every landed rerun `muon_ns` seed is above the strongest
  landed rerun `adamw` seed `0.859765625`

### `test32-vit-cifar100-p2-longer`

- real CIFAR-100 visual benchmark
- longer-training ViT retry with `patch_size=2`
- added to test whether the accepted CIFAR-100 ViT family stays favorable to
  original `muon_ns` over a longer optimization horizon
- smoke passed:
  - `adamw best_test_acc = 0.2623046875`
  - `muon_ns best_test_acc = 0.2734375`
- rerun job `73491` completed successfully after the earlier node-failure retry
- landed gate-only `adamw` best accuracies:
  - `0.4993`
  - `0.497`
  - `0.4951`
  - `0.4961`
  - `0.4826`
- landed rerun `muon_ns` best accuracies:
  - `0.4916`
  - `0.4864`
  - `0.4966`
  - `0.5277`
  - `0.5258`
  - `0.5332`
- accepted because the fully landed `lr=0.001` rerun `muon_ns` block is
  decisively above every landed `adamw` rerun seed

### `test35-tiny-imagenet-vit-deeper`

- harder Tiny-ImageNet ViT follow-up
- same real `200`-class Tiny-ImageNet family as `test30`, but with a deeper
  and wider ViT
- smoke passed:
  - `adamw best_test_acc = 0.08375`
  - `muon_ns best_test_acc = 0.13916666666666666`
- landed formal `adamw` best accuracies:
  - `0.28033854166666666`
  - `0.27239583333333334`
  - `0.28268229166666666`
  - `0.21328125`
  - `0.21692708333333333`
  - `0.22838541666666667`
- landed formal `muon_ns` best accuracies:
  - `0.3326822916666667`
  - `0.33151041666666664`
  - `0.3359375`
  - `0.408203125`
  - `0.399609375`
  - `0.41432291666666665`
- accepted because every landed `muon_ns` value is above every landed `adamw`
  value
- best `adamw = 0.282682`, best `muon_ns = 0.414323`
- best absolute gain: `+0.131641`, about `+13.16` percentage points
- best relative gain: about `+46.6%`
- mean `adamw = 0.249002`, mean `muon_ns = 0.370378`
- mean absolute gain: `+0.121376`, about `+12.14` percentage points
- mean relative gain: about `+48.7%`
- a later comparison run continued into `rt_v43_stream`, but that larger
  optimizer sweep was manually cancelled after the first landed `rt_v43_stream`
  seed because the user chose to stop optimizer-tuning work for now

## Rejected

### Text / LLM

- `test10-pythia-tinystories`
- `test14-gpt2-tinystories`
- `test15-gpt2-tinystories-lrscan`

### CIFAR-10 visual

- `test12-mlpmixer-cifar10`
- `test13-resmlp-cifar10`
- `test18-gmlp-cifar10`
- `test20-vit-cifar10-p2`
- `test22-mlpmixer-cifar10-p2`

### CIFAR-100 visual

- `test17-vit-cifar100`
- `test25-vit-cifar100-p2`

### Tiny-ImageNet MLP-Mixer

- `test36-tiny-imagenet-mlpmixer`
  - smoke rejected
  - `adamw best_test_acc = 0.13020833333333334`
  - `muon_ns best_test_acc = 0.12145833333333333`

### `test27-tabular-mlp-suite`

- real tabular deep learning benchmark suite
- matrix-heavy MLP on:
  - `breast_cancer`
  - `wine`
  - `digits`
- added because public Muon benchmark work reports tabular MLPs as a positive
  direction for Muon-family optimizers
- smoke rejected
- aggregate smoke:
  - `adamw mean_test_acc_at_best_val = 0.9688596491228071`
  - `muon_ns mean_test_acc_at_best_val = 0.9585282651072125`
  - `muon_ns win_count_vs_adamw = 1 / 3`

### `test28-svhn-vit`

- real visual benchmark on SVHN
- matrix-heavy ViT backbone
- added to test whether the accepted CIFAR ViT pattern transfers to another
  real visual distribution
- first smoke failed on download timeout, then reran against cached local SVHN
  data
- smoke rejected on rerun:
  - `adamw best_test_acc = 0.61171875`
  - `muon_ns best_test_acc = 0.5779296875`

## Excluded From Final Non-HamGNN Set

- `test1-hamgnn-si`
- `test2-pinn`
- `test3-burgers-pinn`
- `test4-poisson-pinn`
- `test5-allen-cahn-pinn`
- `test6-reaction-diffusion-pinn`
- `test7-helmholtz-pinn`
- `test8-hamgnn-toy-muonns`
- `test9-hamgnn-sacada`
