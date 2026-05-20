# test30-tiny-imagenet-vit analysis

## Benchmark definition

- model: ViT with patch embedding on `64x64` images
- data: real Tiny-ImageNet-200
- benchmark type: non-HamGNN, non-toy, matrix-heavy visual classification
- acceptance gate: benchmark is accepted only if original `muon_ns` beats
  `adamw`

## Motivation

The current accepted set already covers CIFAR-10, CIFAR-100, and STL10. To push
the benchmark search further without jumping to full ImageNet cost, Tiny-ImageNet
is a good intermediate step:

- harder label space than STL10
- higher resolution than CIFAR
- still practical for single-GPU smoke and formal runs

## Current status

Local scaffold prepared and uploaded to `nmcc-n46h1`.

Server note:

- first smoke job `73429` stalled in server-side `wget`, leaving a zero-byte
  archive
- Tiny-ImageNet was then downloaded locally and uploaded to the server
- smoke job `73430` then failed because `val/images` was still visible to
  `ImageFolder` as a bogus class directory
- validation-layout handling was fixed locally
- smoke rerun on `gpu_h100` as job `73431`

Current smoke progress:

- landed `adamw` smoke summary:
  - `steps=500`
  - `best_test_acc=0.0825`
- landed `muon_ns` smoke summary:
  - `steps=500`
  - `best_test_acc=0.08859375`

So the smoke gate passes:

- `muon_ns best_test_acc = 0.08859375`
- `adamw best_test_acc = 0.0825`

## Formal progress

The landed `adamw lr=0.0005` formal block is now:

- `seed=0`
  - `final_test_acc = 0.2513`
  - `best_test_acc = 0.2513`
- `seed=1`
  - `final_test_acc = 0.2483`
  - `best_test_acc = 0.2483`
- `seed=2`
  - `final_test_acc = 0.2542`
  - `best_test_acc = 0.2542`

So the current `adamw` low-learning-rate formal baseline for comparison is the
range `0.2483` to `0.2542`.

The landed `muon_ns lr=0.0005` formal block is now:

- `seed=0`
  - `final_test_acc = 0.2569`
  - `best_test_acc = 0.2571`
- `seed=1`
  - `final_test_acc = 0.2634`
  - `best_test_acc = 0.2634`
- `seed=2`
  - `final_test_acc = 0.2636`
  - `best_test_acc = 0.2636`

Relative to the strongest landed `adamw` low-learning-rate result:

- best `adamw` low-learning-rate accuracy: `0.2542`
- `muon_ns` low-learning-rate accuracies: `0.2571`, `0.2634`, `0.2636`

So every landed low-learning-rate `muon_ns` seed is above every landed
low-learning-rate `adamw` seed. That is sufficient to accept this benchmark
under the user's gate even though job `73432` is still running later blocks.
