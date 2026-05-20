# test26-vit-cifar100-p2-deeper analysis

## Benchmark definition

- model: ViT with `patch_size=2`, deeper encoder
- data: real CIFAR-100
- benchmark type: non-HamGNN, non-toy, matrix-heavy visual classification
- acceptance gate: benchmark is accepted only if original `muon_ns` beats `adamw`

## Motivation

This benchmark is the CIFAR-100 counterpart of `test23-vit-cifar10-p2-deeper`.
It helps separate the effect of extra depth from the effect of extra width.

## Current status

Local scaffold prepared.

Server note:

- smoke completed on `gpu_h100` as job `73421`

## Smoke result

Landed smoke summaries at `lr=0.0005`, `seed=0`:

### `adamw`

- `final_test_acc=0.1927734375`
- `best_test_acc=0.205078125`

### `muon_ns`

- `final_test_acc=0.2162109375`
- `best_test_acc=0.2236328125`

## Current conclusion

Under the current smoke configuration, original `muon_ns` beats `adamw`, so
this benchmark is currently **accepted at the smoke stage** and is worth
continuing.

## Next action

This benchmark has now been promoted to a formal run on `gpu_h100` as job
`73425`, so the CIFAR-100 deeper recipe can be judged under the same multi-seed
standard as `test23` and `test24`.

## Formal progress

Landed `adamw` formal summaries so far at `lr=0.0005`:

- `seed=0`: `final_test_acc=0.4495`, `best_test_acc=0.4495`
- `seed=1`: `final_test_acc=0.4456`, `best_test_acc=0.4470`
- `seed=2`: `final_test_acc=0.4440`, `best_test_acc=0.4440`

The first landed `muon_ns` formal summary is now available at `lr=0.0005`,
`seed=0`:

- `final_test_acc=0.4468`
- `best_test_acc=0.4472`

The second landed `muon_ns` formal summary is now available at `lr=0.0005`,
`seed=1`:

- `final_test_acc=0.4480`
- `best_test_acc=0.4482`

The third landed `muon_ns` formal summary is now available at `lr=0.0005`,
`seed=2`:

- `final_test_acc=0.4546`
- `best_test_acc=0.4546`

The currently landed `muon_ns` formal block is still mixed:

- `muon_ns seed=0 best_test_acc = 0.4472`
- `muon_ns seed=1 best_test_acc = 0.4482`
- `muon_ns seed=2 best_test_acc = 0.4546`

Relative to `adamw`:

- `muon_ns seed=0` is below `adamw seed=0 best_test_acc = 0.4495`
- `muon_ns seed=1` is above `adamw seed=1 best_test_acc = 0.4470`
- `muon_ns seed=2` is above `adamw seed=2 best_test_acc = 0.4440`

So the accepted/rejected decision remains pending. The low-learning-rate formal
block is no longer incomplete, but the evidence is mixed rather than cleanly
favorable.

## Additional formal progress

The higher-learning-rate `lr=0.001` formal block has now fully landed.

### `adamw` at `lr=0.001`

- `seed=0`: `final_test_acc=0.4170`, `best_test_acc=0.4170`
- `seed=1`: `final_test_acc=0.4157`, `best_test_acc=0.4175`
- `seed=2`: `final_test_acc=0.4238`, `best_test_acc=0.4271`

### `muon_ns` at `lr=0.001`

- `seed=0`: `final_test_acc=0.5205`, `best_test_acc=0.5205`
- `seed=1`: `final_test_acc=0.5187`, `best_test_acc=0.5192`
- `seed=2`: `final_test_acc=0.5246`, `best_test_acc=0.5277`

This higher-learning-rate block is decisive:

- every landed `muon_ns` seed at `lr=0.001` is far above every landed `adamw`
  seed at `lr=0.001`

So even though the `lr=0.0005` block was mixed, the benchmark itself now
passes the user's acceptance gate because there exists a fully landed formal
original-`muon_ns` setting that clearly beats `adamw`.
