# test10-pythia-tinystories analysis

## Run record

- Server: `nmcc-n46h1`
- Work tree: `~/run/4.3-log/test10-pythia-tinystories`
- Queue: `gpu_a800`
- Jobs:
  - `73370`: `adamw`
  - `73371`: `muon_ns`

## Benchmark definition

- model: `Pythia-160M` config
- initialization: config-only, not pretrained weights
- data: local TinyStories text files on the server
- acceptance gate: benchmark is accepted only if original `muon_ns` beats `adamw`

## Result

This benchmark is **not accepted** under the user's rule.

From the server-side `summary.json` files:

### `adamw`

- `completed_steps=600`
- `best_val_loss=1.028679771348834`
- `last_val_loss=1.5758180655539036`
- `train_loss_mean=1.5165613722250175`
- `elapsed_s=273.8238055706024`

### `muon_ns`

- `completed_steps=600`
- `best_val_loss=1.4042991697788239`
- `last_val_loss=1.5998346619307995`
- `train_loss_mean=6.39175776163737`
- `elapsed_s=392.1874587535858`

## Conclusion

- `test10-pythia-tinystories` is a valid non-HamGNN real-model benchmark
- it ran on real local text data, not synthetic fallback
- but original `muon_ns` is clearly worse than `adamw` on the main validation-loss metric
- therefore this benchmark is rejected under the user's acceptance gate
