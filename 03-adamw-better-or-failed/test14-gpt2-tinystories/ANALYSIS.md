# test14-gpt2-tinystories analysis

## Run record

- Server: `nmcc-n46h1`
- Work tree: `/data/run01/scwb923/4.3-log/test14-gpt2-tinystories`
- Queue: `gpu_a800`

## Benchmark definition

- model: GPT-2 style decoder
- initialization: config-only, from scratch
- data: local TinyStories text files on the server
- acceptance gate: benchmark is accepted only if original `muon_ns` beats `adamw`

## Current status

Smoke runs were submitted and completed on the server.

## Smoke result

Under the current configuration, this benchmark is **rejected** by the user's
gate.

From the completed smoke summaries:

### `adamw`

- `completed_steps=300`
- `best_val_loss=0.9214051812887192`
- `last_val_loss=1.1508001536130905`

### `muon_ns`

- `completed_steps=300`
- `best_val_loss=1.86994019895792`
- `last_val_loss=1.86994019895792`

So original `muon_ns` is clearly worse than `adamw` on validation loss in this
configuration.

## Current conclusion

- benchmark scaffold is valid
- the current smoke configuration is rejected
- if this language-model direction is kept, it should be revisited with a more
  explicit LR screening setup rather than a single fixed learning rate
