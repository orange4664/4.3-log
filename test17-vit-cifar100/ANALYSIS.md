# test17-vit-cifar100 analysis

## Run record

- Server: `nmcc-n46h1`
- Work tree: `/data/run01/scwb923/4.3-log/test17-vit-cifar100`
- Queue: `gpu_a800`

## Benchmark definition

- model: compact ViT
- data: real CIFAR-100
- benchmark type: non-HamGNN, non-toy, matrix-heavy visual classification
- acceptance gate: benchmark is accepted only if original `muon_ns` beats `adamw`

## Motivation

`test11-vit-cifar10` is already an accepted benchmark. This benchmark keeps the
same matrix-heavy ViT family and raises the task difficulty by replacing
CIFAR-10 with CIFAR-100.

## Current status

Local scaffold prepared and server smoke rerun is active.

## Data setup note

The first smoke submission for this benchmark failed for a non-benchmark
reason: the cluster job tried to download CIFAR-100 directly through
`torchvision`, and the remote HTTPS request timed out.

This was fixed by:

- downloading `cifar-100-python.tar.gz` locally
- uploading it to `/data/run01/scwb923/4.3-log/test17-vit-cifar100/data`
- extracting it on the cluster
- removing `--download` from the smoke and formal sbatch scripts

So the currently running smoke rerun is now a valid offline benchmark run.

## Smoke result

The offline smoke rerun completed successfully and is valid for benchmark
screening.

### `adamw`

- `steps=780`
- `final_test_acc=0.237890625`
- `best_test_acc=0.237890625`

### `muon_ns`

- `steps=780`
- `final_test_acc=0.2099609375`
- `best_test_acc=0.2099609375`

## Conclusion

Under the current compact ViT + CIFAR-100 smoke configuration, original
`muon_ns` is clearly worse than `adamw`, so this benchmark is currently
**rejected** under the user's gate.
