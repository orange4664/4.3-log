# test22-mlpmixer-cifar10-p2 analysis

## Run record

- Server: `nmcc-n46h1`
- Work tree: `/data/run01/scwb923/4.3-log/test22-mlpmixer-cifar10-p2`
- Queue: `gpu_a800`

## Benchmark definition

- model: compact MLP-Mixer with patch size `2`
- data: real CIFAR-10 server cache
- benchmark type: non-HamGNN, non-toy, matrix-heavy visual classification
- acceptance gate: benchmark is accepted only if original `muon_ns` beats `adamw`

## Current server status

Smoke run completed.

## Smoke result

### `adamw`

- `steps=780`
- `final_test_acc=0.583984375`
- `best_test_acc=0.583984375`

### `muon_ns`

- `steps=780`
- `final_test_acc=0.5298828125`
- `best_test_acc=0.5298828125`

## Conclusion

Under the longer-sequence MLP-Mixer + CIFAR-10 smoke configuration, original
`muon_ns` remains worse than `adamw`, so this benchmark is currently
**rejected** under the user's gate.
