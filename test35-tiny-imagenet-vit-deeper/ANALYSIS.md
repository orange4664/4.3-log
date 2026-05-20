# test35-tiny-imagenet-vit-deeper analysis

## Benchmark definition

- model: ViT with patch embedding on `64x64` images
- data: real Tiny-ImageNet
- benchmark type: non-HamGNN, non-toy, matrix-heavy visual classification
- stress axis: harder than `test30` via deeper/wider model and longer horizon
- acceptance gate: benchmark is accepted only if original `muon_ns` beats
  `adamw`

## Motivation

The accepted Tiny-ImageNet result in `test30` is already useful, but it still
leaves room to ask whether the same real-data advantage survives on a stronger
ViT recipe. This benchmark answers that without opening a new data dependency.

## Current status

Scaffolded locally and synced to the cluster workspace:

- `/data/run01/scwb923/4.3-log/test35-tiny-imagenet-vit-deeper`

Smoke has been submitted on `gpu_h100`:

- job `73497`

The smoke job is no longer pending. It is actively running on:

- node `d1n41d29g01`

The first landed smoke summary is:

- `adamw lr=0.0005 seed=0 best_test_acc = 0.08375`
- `muon_ns lr=0.0005 seed=0 best_test_acc = 0.13916666666666666`

So the smoke gate already passes on the first landed pair.

## Smoke verdict

Under the user's gate:

- accept promotion only if original `muon_ns` beats `adamw`

The first landed smoke pair satisfies that condition:

- `adamw`: `0.08375`
- `muon_ns`: `0.13916666666666666`

## Formal status

The formal run has been submitted and is actively running:

- job `73503`
- node `d1n41d29g01`

The landed formal summaries for `adamw` are:

- `lr=0.0005 seed=0 best_test_acc = 0.28033854166666666`
- `lr=0.0005 seed=1 best_test_acc = 0.27239583333333334`
- `lr=0.0005 seed=2 best_test_acc = 0.28268229166666666`
- `lr=0.001 seed=0 best_test_acc = 0.21328125`
- `lr=0.001 seed=1 best_test_acc = 0.21692708333333333`
- `lr=0.001 seed=2 best_test_acc = 0.22838541666666667`

The landed formal summaries for original `muon_ns` are:

- `lr=0.0005 seed=0 best_test_acc = 0.3326822916666667`
- `lr=0.0005 seed=1 best_test_acc = 0.33151041666666664`
- `lr=0.0005 seed=2 best_test_acc = 0.3359375`
- `lr=0.001 seed=0 best_test_acc = 0.408203125`
- `lr=0.001 seed=1 best_test_acc = 0.399609375`
- `lr=0.001 seed=2 best_test_acc = 0.41432291666666665`

These landed formal results already show a clean `muon_ns > adamw` signal under
the user's gate:

- every landed `muon_ns` value is above every landed `adamw` value
- the strongest landed `adamw` value is `0.28268229166666666`
- the weakest landed `muon_ns` value is `0.33151041666666664`

## What is still pending

This benchmark cannot be treated as fully closed yet, because the submitted
formal job did not stop after `adamw` and `muon_ns`.

The same Slurm job is still continuing through later optimizer blocks:

- `rt_v43_stream`
- `rt_v43_ns`
- `rt_v6_fdt_metric`

Cluster-side evidence already shows that the run has moved into
`rt_v43_stream`:

- path exists:
  - `runs/rt_v43_stream/lr_0.0005/seed_0/diag/`
- first landed formal summary:
  - `rt_v43_stream lr=0.0005 seed=0 best_test_acc = 0.31940104166666666`
  - `steps = 6246`
  - `time_sec = 2527.1007554531097`
- next seed directory now exists:
  - `runs/rt_v43_stream/lr_0.0005/seed_1/diag/`

So the correct status is:

- original `muon_ns` has already beaten `adamw` decisively on the landed formal
  part of this benchmark
- the first post-`muon_ns` `rt_v43_stream` formal seed has now fully landed
- but the overall benchmark job is still running, so the repository-level final
  summary should still wait for the full sweep to end

## Next action

- wait for job `73503` to finish
- once all later method blocks land, update:
  - `STATUS.md`
  - `ANALYSIS.md`
  - top-level benchmark summary files
  - repository `README.md`
