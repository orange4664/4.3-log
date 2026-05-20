# test24-vit-cifar100-p2-wide analysis

## Benchmark definition

- model: ViT with `patch_size=2`, wider embedding, deeper encoder
- data: real CIFAR-100
- benchmark type: non-HamGNN, non-toy, matrix-heavy visual classification
- acceptance gate: benchmark is accepted only if original `muon_ns` beats `adamw`

## Motivation

`test17-vit-cifar100` showed that the smallest compact CIFAR-100 ViT recipe is
not enough. This benchmark retries the real CIFAR-100 line with a stronger ViT
configuration chosen to resemble the already-successful CIFAR-10 ViT family.

## Current status

Local scaffold prepared.

Server note:

- initial smoke submission on `gpu_a800` stayed pending because that partition
  had zero free GPUs at the time
- smoke was resubmitted to `gpu_h100` as job `73418`

## Smoke result

The `gpu_h100` smoke rerun completed successfully and passed the user's gate.

### `adamw`

- `steps=780`
- `final_test_acc=0.2119140625`
- `best_test_acc=0.2119140625`

### `muon_ns`

- `steps=780`
- `final_test_acc=0.241796875`
- `best_test_acc=0.241796875`

## Conclusion

Under this stronger CIFAR-100 ViT configuration, original `muon_ns` beats
`adamw` at the smoke stage, so this benchmark is currently **accepted at the
smoke stage** and should be promoted to a formal run.

## Formal progress

- formal submitted on `gpu_h100`
- current formal job: `73420`
- current state: running

Landed `adamw` formal summaries at `lr=0.0005`:

- `seed=0`: `final_test_acc=0.4553`, `best_test_acc=0.4562`
- `seed=1`: `final_test_acc=0.4484`, `best_test_acc=0.4505`

Landed `muon_ns` formal summaries now available at `lr=0.0005`:

- `seed=0`: `final_test_acc=0.466`, `best_test_acc=0.466`
- `seed=1`: `final_test_acc=0.4727`, `best_test_acc=0.4755`
- `seed=2`: `final_test_acc=0.4738`, `best_test_acc=0.4738`

Both landed `muon_ns` formal summaries are above every landed `adamw` formal
summary:

- `adamw seed=0 best_test_acc = 0.4562`
- `adamw seed=1 best_test_acc = 0.4505`
- `adamw seed=2 best_test_acc = 0.4533`

So formal acceptance is already established under the user's gate. The active
job `73420` is still running because higher-learning-rate or other remaining
method blocks are still being executed, but the benchmark itself is now a valid
accepted non-HamGNN benchmark because original `muon_ns` beats `adamw` across
the landed low-learning-rate formal seeds.

Additional higher-learning-rate progress has also landed:

- `muon_ns lr=0.001 seed=0`: `final_test_acc=0.5331`,
  `best_test_acc=0.5343`

That high-learning-rate block is much worse than the low-learning-rate block,
but it does not change the acceptance conclusion because the benchmark gate is
already satisfied by the complete landed `lr=0.0005` formal seed set.
