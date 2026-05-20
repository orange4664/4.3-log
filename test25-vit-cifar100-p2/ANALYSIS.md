# test25-vit-cifar100-p2 analysis

## Benchmark definition

- model: ViT with `patch_size=2`
- data: real CIFAR-100
- benchmark type: non-HamGNN, non-toy, matrix-heavy visual classification
- acceptance gate: benchmark is accepted only if original `muon_ns` beats `adamw`

## Motivation

This benchmark is the CIFAR-100 counterpart of the already smoke-accepted
`test20-vit-cifar10-p2` benchmark. It isolates the longer-sequence effect
without simultaneously widening and deepening the architecture.

## Current status

Local scaffold prepared.

Server note:

- initial smoke submission on `gpu_a800` stayed pending because that partition
  had zero free GPUs at the time
- smoke was resubmitted to `gpu_h100` as job `73419`

## Smoke result

The `gpu_h100` smoke rerun completed successfully, but it did not pass the
user's gate.

### `adamw`

- `steps=780`
- `final_test_acc=0.22890625`
- `best_test_acc=0.2291015625`

### `muon_ns`

- `steps=780`
- `final_test_acc=0.22109375`
- `best_test_acc=0.2228515625`

## Conclusion

Under this CIFAR-100 `patch_size=2` configuration, original `muon_ns` is below
`adamw`, so this benchmark is currently **rejected** under the user's gate.
