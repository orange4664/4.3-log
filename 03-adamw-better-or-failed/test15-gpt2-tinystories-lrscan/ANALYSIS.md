# test15-gpt2-tinystories-lrscan analysis

## Run record

- Server: `nmcc-n46h1`
- Work tree: `/data/run01/scwb923/4.3-log/test15-gpt2-tinystories-lrscan`
- Queue: `gpu_a800`

## Benchmark definition

- model: GPT-2 style decoder
- initialization: config-only, from scratch
- data: local TinyStories text files on the server
- comparison: `adamw` vs original `muon_ns`
- decision rule: compare each optimizer at its best screened learning rate

## Current status

The small LR screening run was submitted and completed for:

- `adamw` at `1e-4`
- `adamw` at `3e-4`
- `muon_ns` at `3e-5`
- `muon_ns` at `1e-4`

## Screening result

The benchmark is **rejected** under the user's gate for the screened learning
rates.

Best completed `adamw` result:

- `lr=3e-4`
- `best_val_loss=0.9214051812887192`

Second `adamw` point:

- `lr=1e-4`
- `best_val_loss=1.005099631845951`

Best completed `muon_ns` result:

- `lr=1e-4`
- `best_val_loss=4.6742969155311584`

Second `muon_ns` point:

- `lr=3e-5`
- `best_val_loss=8.631155490875244`

So even after introducing a small learning-rate screen, original `muon_ns`
remains far worse than `adamw` on this benchmark.

## Current conclusion

- benchmark scaffold is valid
- the screened language-model setup is rejected under the user's gate
- this particular small GPT TinyStories direction is not currently a good
  accepted benchmark for original `muon_ns`
