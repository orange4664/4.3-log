# test31-svhn-vit-p2-wide analysis

## Benchmark definition

- model: ViT with `patch_size=2`, wider embedding, deeper encoder
- data: real SVHN
- benchmark type: non-HamGNN, non-toy, matrix-heavy visual classification
- acceptance gate: benchmark is accepted only if original `muon_ns` beats
  `adamw`

## Motivation

The first SVHN ViT attempt (`test28`) used a weaker configuration and failed
the gate. This retry keeps the same real dataset but moves the backbone closer
to the already-successful accepted family:

- more tokens via `patch_size=2`
- more matrix capacity via larger embedding width
- deeper transformer stack

## Current status

Local scaffold prepared and uploaded to `nmcc-n46h1`.

Server note:

- smoke completed on `gpu_h100` as job `73433`
- formal submitted on `gpu_h100` as job `73434`

Current smoke progress:

- landed `adamw` smoke summary:
  - `steps=1144`
  - `best_test_acc=0.484375`
- landed `muon_ns` smoke summary:
  - `steps=1144`
  - `best_test_acc=0.5533203125`

This is a strong enough smoke win to justify promotion:

- absolute gain over `adamw`: `+0.0689453125`

## Next action

- let `73434` run the full lr/seed block on cached SVHN data
- accept only if the formal results preserve `muon_ns > adamw`
