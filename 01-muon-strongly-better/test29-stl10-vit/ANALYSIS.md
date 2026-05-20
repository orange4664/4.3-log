# test29-stl10-vit analysis

## Benchmark definition

- model: ViT with patch embedding on `96x96` images
- data: real STL10
- benchmark type: non-HamGNN, non-toy, matrix-heavy visual classification
- acceptance gate: benchmark is accepted only if original `muon_ns` beats
  `adamw`

## Motivation

The currently strongest accepted benchmarks in this search are mostly CIFAR
variants. STL10 is a useful next step because it changes the visual domain and
resolution without jumping to a very large ImageNet-scale workload.

## Current status

Local scaffold prepared and synced to the server.

First server smoke run on `gpu_h100`:

- job `73426`
- result: failed before training because online STL10 download timed out

Recovery:

- a complete STL10 archive was downloaded locally
- the archive was uploaded to the cluster
- STL10 was extracted into `data/stl10_binary`

Offline rerun:

- job `73427`

## Smoke result

The offline rerun completed and passed the user's gate.

### `adamw`

- `steps=156`
- `final_test_acc=0.287109375`
- `best_test_acc=0.303515625`

### `muon_ns`

- `steps=156`
- `final_test_acc=0.318359375`
- `best_test_acc=0.318359375`

## Conclusion

Under this higher-resolution STL10 + ViT smoke configuration, original
`muon_ns` beats `adamw`, so this benchmark is currently **accepted at the smoke
stage** and is worth promoting to a formal run.

## Formal progress

Formal has now been submitted on `gpu_h100`:

- job `73428`

The landed low-learning-rate `adamw` formal block is:

- `seed=0`: `final_test_acc=0.4357421875`, `best_test_acc=0.4357421875`
- `seed=1`: `final_test_acc=0.4419921875`, `best_test_acc=0.4419921875`
- `seed=2`: `final_test_acc=0.4623046875`, `best_test_acc=0.4623046875`

The landed low-learning-rate `muon_ns` formal block is:

- `seed=0`: `final_test_acc=0.4876953125`, `best_test_acc=0.4876953125`
- `seed=1`: `final_test_acc=0.4826171875`, `best_test_acc=0.48671875`
- `seed=2`: `final_test_acc=0.4865234375`, `best_test_acc=0.4865234375`

Relative to the strongest landed `adamw` low-learning-rate result:

- best `adamw` low-learning-rate accuracy: `0.4623046875`
- `muon_ns` low-learning-rate accuracies: `0.4876953125`, `0.48671875`,
  `0.4865234375`

So this benchmark now satisfies the user's acceptance gate. It should be
counted as an accepted non-HamGNN benchmark even though job `73428` is still
running later blocks after the already-decisive low-learning-rate formal block.
