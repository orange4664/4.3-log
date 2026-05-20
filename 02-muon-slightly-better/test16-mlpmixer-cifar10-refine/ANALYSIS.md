# test16-mlpmixer-cifar10-refine analysis

## Run record

- Server: `nmcc-n46h1`
- Work tree: `/data/run01/scwb923/4.3-log/test16-mlpmixer-cifar10-refine`
- Queue: `gpu_a800`

## Benchmark definition

- model: compact MLP-Mixer
- data: real CIFAR-10 server cache
- comparison: focused `adamw` vs original `muon_ns`
- goal: resolve the near-miss gap seen in `test12`

## Current status

Server jobs have completed.

- `73392`: `adamw`, `lr=5e-4` completed
- `73393`: `adamw`, `lr=7e-4` completed
- `73394`: `muon_ns`, `lr=1e-3` completed
- `73395`: `muon_ns`, `lr=1.2e-3` completed

Completed `adamw` results currently set a strong bar:

- best completed `adamw` = `0.7790` at `lr=0.0007`, `seed=2`

Completed `muon_ns` results clear that bar:

- `lr=0.0012`, `seed=0`: `best_test_acc=0.7842`
- `lr=0.0012`, `seed=1`: `best_test_acc=0.7857`
- `lr=0.0012`, `seed=2`: `best_test_acc=0.7867`

Mean `muon_ns` best accuracy at `lr=0.0012`:

- `mean_best_test_acc = 0.7855`

For comparison, the strongest completed `adamw` block was `lr=0.0007`:

- `seed=0`: `0.7760`
- `seed=1`: `0.7760`
- `seed=2`: `0.7790`
- `mean_best_test_acc = 0.7770`

## Conclusion

`test16-mlpmixer-cifar10-refine` is an **accepted** benchmark under the user's
gate.

This matters because the original `test12` MLP-Mixer line was only a near-miss.
The focused refine pass found a `muon_ns` regime that is now clearly better
than `adamw` on the same non-HamGNN, real-data, matrix-heavy benchmark family.
