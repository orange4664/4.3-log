# test11-vit-cifar10 analysis

## Run record

- Server: `nmcc-n46h1`
- Work tree: `/data/run01/scwb923/4.3-log/test11-vit-cifar10`
- Queue: `gpu_a800`
- Smoke job:
  - `73372`
- Formal job:
  - `73376`

## Benchmark definition

- model: compact ViT
- data: real CIFAR-10 server cache
- benchmark type: non-HamGNN, non-toy, matrix-heavy visual classification
- acceptance gate: benchmark is accepted only if original `muon_ns` beats `adamw`

## Smoke result

The smoke run already satisfied the acceptance gate.

From `runs_smoke/all_summaries.json`:

### `adamw`

- `steps=780`
- `final_test_acc=0.4990234375`
- `best_test_acc=0.535546875`

### `muon_ns`

- `steps=780`
- `final_test_acc=0.5185546875`
- `best_test_acc=0.540234375`

So on the smoke run, original `muon_ns` was better than `adamw`.

## Formal run result

At the current checkpoint, `adamw` has completed all three seeds at `lr=0.0005`:

- seed `0`: `final_test_acc=0.6913`, `best_test_acc=0.6913`
- seed `1`: `final_test_acc=0.6812`, `best_test_acc=0.6812`
- seed `2`: `final_test_acc=0.6908`, `best_test_acc=0.6948`

Mean `adamw` best accuracy over the finished three seeds:

- `mean_best_test_acc = 0.6891`

The first completed formal `muon_ns` run is already better than the matching
`adamw` seed at the same learning rate:

### `muon_ns`, `lr=0.0005`, `seed=0`

- `final_test_acc=0.7058`
- `best_test_acc=0.7066`

For comparison, the corresponding `adamw`, `lr=0.0005`, `seed=0` run was:

- `final_test_acc=0.6913`
- `best_test_acc=0.6913`

So the formal run is currently consistent with the smoke-stage conclusion that
original `muon_ns` can beat `adamw` on this benchmark.

The second completed formal `muon_ns` run also stays ahead:

### `muon_ns`, `lr=0.0005`, `seed=1`

- `final_test_acc=0.7130`
- `best_test_acc=0.7131`

For comparison, the corresponding `adamw`, `lr=0.0005`, `seed=1` run was:

- `final_test_acc=0.6812`
- `best_test_acc=0.6812`

So two completed formal seeds both support the same conclusion.

The third completed formal `muon_ns` run also stays ahead:

### `muon_ns`, `lr=0.0005`, `seed=2`

- `final_test_acc=0.7114`
- `best_test_acc=0.7157`

For comparison, the corresponding `adamw`, `lr=0.0005`, `seed=2` run was:

- `final_test_acc=0.6908`
- `best_test_acc=0.6948`

Across the three completed `muon_ns` seeds at `lr=0.0005`:

- `muon_ns` mean `best_test_acc = 0.7118`

For the matching completed `adamw` runs at `lr=0.0005`:

- `adamw` mean `best_test_acc = 0.6891`

There are now also completed higher-learning-rate `muon_ns` runs:

### `muon_ns`, `lr=0.001`

- seed `0`: `final_test_acc=0.7616`, `best_test_acc=0.7624`
- seed `1`: `final_test_acc=0.7738`, `best_test_acc=0.7738`
- seed `2`: `final_test_acc=0.7706`, `best_test_acc=0.7725`

Compared with the best completed `adamw` result across its finished runs:

- best completed `adamw` = `0.6948`

So the formal margin is no longer only small or seed-local; the full `lr=0.001`
three-seed block is substantially ahead of `adamw`.

Mean `muon_ns` best accuracy over the three `lr=0.001` seeds:

- `mean_best_test_acc = 0.7696`

## Current conclusion

- `test11-vit-cifar10` is an accepted benchmark at both smoke and formal stage
- all three completed formal `muon_ns` seeds at `lr=0.0005` are ahead of the
  matching `adamw` runs at the same learning rate
- all three completed formal `muon_ns` seeds at `lr=0.001` are also ahead of
  the best `adamw` run
- this benchmark should now be treated as a clearly accepted benchmark under
  the user's gate
