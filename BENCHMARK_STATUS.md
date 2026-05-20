# Muon Harder Benchmark Status

This file tracks the non-HamGNN benchmark search requested in this thread.

Acceptance rule:

- a benchmark is counted as good only if original `muon_ns` beats `adamw`

Repository grouping:

- strong Muon wins live under `01-muon-strongly-better/`
- smaller accepted Muon wins live under `02-muon-slightly-better/`
- rejected or AdamW-favored runs live under `03-adamw-better-or-failed/`
- HamGNN/PINN/incomplete material lives under
  `04-archive-hamgnn-pinn-and-incomplete/`

## Accepted

- `test11-vit-cifar10`
  - compact ViT on real CIFAR-10
  - smoke accepted
  - formal accepted

- `test16-mlpmixer-cifar10-refine`
  - refined compact MLP-Mixer on real CIFAR-10
  - smoke/formal evidence accepted

- `test19-vit-cifar10-large`
  - wider/deeper ViT on real CIFAR-10
  - formal evidence accepted
  - landed `muon_ns` best accuracies `0.7316`, `0.7317`, `0.7322`
  - clearly above landed `adamw` best accuracy `0.6642`

- `test23-vit-cifar10-p2-deeper`
  - longer-sequence deeper ViT on real CIFAR-10
  - formal evidence accepted
  - landed `muon_ns` best accuracies `0.7352`, `0.7359`, `0.7371`
  - all are above landed `adamw` formal best `0.7195`

- `test24-vit-cifar100-p2-wide`
  - stronger real CIFAR-100 ViT benchmark
  - formal evidence accepted
  - landed `muon_ns` best accuracies `0.4660`, `0.4755`, `0.4738`
  - all are above landed `adamw` formal best `0.4562`

- `test21-vit-cifar10-p2-wide`
  - longer-sequence wider ViT on real CIFAR-10
  - formal evidence accepted
  - landed `muon_ns` best accuracies `0.7486`, `0.7414`, `0.7532`
  - all are above landed `adamw` formal best `0.6923`

- `test29-stl10-vit`
  - higher-resolution real-vision benchmark on STL10
  - formal low-learning-rate evidence accepted
  - landed `adamw` low-learning-rate best accuracies `0.4357421875`,
    `0.4419921875`, `0.4623046875`
  - landed `muon_ns` low-learning-rate best accuracies `0.4876953125`,
    `0.48671875`, `0.4865234375`
  - all landed `muon_ns` low-learning-rate seeds are above the strongest
    landed `adamw` low-learning-rate seed

- `test26-vit-cifar100-p2-deeper`
  - harder CIFAR-100 ViT retry
  - formal evidence accepted
  - landed `lr=0.001` `adamw` best accuracies `0.417`, `0.4175`, `0.4271`
  - landed `lr=0.001` `muon_ns` best accuracies `0.5205`, `0.5192`, `0.5277`
  - every landed `muon_ns` seed at that formal block is above every landed
    `adamw` seed at the same learning rate

- `test30-tiny-imagenet-vit`
  - harder Tiny-ImageNet ViT benchmark
  - real `64x64`, `200`-class visual classification
  - smoke accepted
  - low-learning-rate formal `adamw` best accuracies `0.2513`, `0.2483`,
    `0.2542`
  - low-learning-rate formal `muon_ns` best accuracies `0.2571`, `0.2634`,
    `0.2636`
  - every landed low-learning-rate `muon_ns` seed is above the strongest
    landed low-learning-rate `adamw` seed

- `test33-stl10-vit-longer`
  - harder real-data STL10 ViT benchmark with longer optimization horizon
  - smoke accepted
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
  - every landed `muon_ns` value is above the strongest landed `adamw`
    value `0.498046875`

- `test31-svhn-vit-p2-wide`
  - stronger SVHN ViT retry with `patch_size=2`
  - smoke and gate-only rerun now completed
  - landed rerun `adamw` best accuracies:
    - `0.83662109375`
    - `0.8296875`
    - `0.7994140625`
    - `0.75400390625`
    - `0.80537109375`
    - `0.859765625`
  - landed rerun `muon_ns` best accuracies:
    - `0.9234375`
    - `0.91435546875`
    - `0.91787109375`
    - `0.94619140625`
    - `0.93984375`
    - `0.94189453125`
  - every landed `muon_ns` rerun seed is above the strongest landed `adamw`
    rerun seed `0.859765625`

- `test32-vit-cifar100-p2-longer`
  - longer-training CIFAR-100 ViT retry with `patch_size=2`
  - smoke and gate-only rerun now completed
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
  - accepted because the fully landed `lr=0.001` rerun `muon_ns` block
    decisively beats every landed `adamw` rerun seed

- `test35-tiny-imagenet-vit-deeper`
  - harder Tiny-ImageNet ViT follow-up
  - real `64x64`, `200`-class visual classification
  - smoke accepted
  - formal landed `adamw` best accuracies:
    - `0.28033854166666666`
    - `0.27239583333333334`
    - `0.28268229166666666`
    - `0.21328125`
    - `0.21692708333333333`
    - `0.22838541666666667`
  - formal landed `muon_ns` best accuracies:
    - `0.3326822916666667`
    - `0.33151041666666664`
    - `0.3359375`
    - `0.408203125`
    - `0.399609375`
    - `0.41432291666666665`
  - every landed `muon_ns` value is above every landed `adamw` value
  - best `adamw = 0.282682`, best `muon_ns = 0.414323`
  - best absolute gain is `+0.131641`, about `+13.16` percentage points
  - best relative gain is about `+46.6%`
  - mean `adamw = 0.249002`, mean `muon_ns = 0.370378`
  - mean absolute gain is `+0.121376`, about `+12.14` percentage points
  - mean relative gain is about `+48.7%`
  - the later optimizer comparison job was manually cancelled after the first
    landed `rt_v43_stream` seed because the user decided to stop optimizer
    tuning and keep only the already-validated `muon_ns vs adamw` conclusion

## Pending

- no current benchmark is pending in the reorganized non-HamGNN status table

## Rejected

- `test10-pythia-tinystories`
  - real-model LLM benchmark using `Pythia-160M` config init on local
    TinyStories text
  - rejected under the user's gate
  - `adamw best_val_loss = 1.028679771348834`
  - `muon_ns best_val_loss = 1.4042991697788239`

- `test12-mlpmixer-cifar10`
- `test13-resmlp-cifar10`
- `test14-gpt2-tinystories`
- `test15-gpt2-tinystories-lrscan`
- `test17-vit-cifar100`
- `test18-gmlp-cifar10`
- `test22-mlpmixer-cifar10-p2`
- `test20-vit-cifar10-p2`
  - formal rejected
  - `adamw` best landed value: `0.7311`
  - `muon_ns` landed values: `0.7290`, `0.7280`, `0.7285`
- `test27-tabular-mlp-suite`
  - smoke rejected
  - real tabular deep learning benchmark suite on:
    - `breast_cancer`
    - `wine`
    - `digits`
  - smoke aggregate:
    - `adamw mean_test_acc_at_best_val = 0.9688596491228071`
    - `muon_ns mean_test_acc_at_best_val = 0.9585282651072125`
    - `muon_ns win_count_vs_adamw = 1 / 3`
- `test28-svhn-vit`
  - smoke rejected after rerun with cached local SVHN data
  - real visual benchmark on SVHN with matrix-heavy ViT
  - rerun smoke:
    - `adamw best_test_acc = 0.61171875`
    - `muon_ns best_test_acc = 0.5779296875`
- `test25-vit-cifar100-p2`
  - smoke rejected
  - `adamw`: `0.2291015625`
  - `muon_ns`: `0.2228515625`
- `test36-tiny-imagenet-mlpmixer`
  - smoke rejected
  - `adamw`: `0.13020833333333334`
  - `muon_ns`: `0.12145833333333333`

## Out Of Scope / Deprecated For This Search

- `test1-hamgnn-si`
- `test8-hamgnn-toy-muonns`
- `test9-hamgnn-sacada`
- `test2-pinn`
- `test3-burgers-pinn`
- `test4-poisson-pinn`
- `test5-allen-cahn-pinn`
- `test6-reaction-diffusion-pinn`
- `test7-helmholtz-pinn`
- earlier HamGNN-specific repair notes
