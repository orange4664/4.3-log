# 4.3-log benchmark repository

This repository records the Muon / AdamW benchmark work from this thread. The
current non-HamGNN acceptance rule is deliberately strict:

- a benchmark is counted as positive only when original `muon_ns` beats `adamw`
- later `rt_v43_*` / `rt_v6_*` optimizer tuning is not used to promote a
  benchmark unless that comparison actually finished

The repository is now organized by result strength instead of by raw `testXX`
order.

## Directory layout

### `01-muon-strongly-better/`

Benchmarks where `muon_ns` is clearly better than `adamw`, usually by a large
absolute gap or by a robust all-seed win.

Included folders:

- `test19-vit-cifar10-large`
- `test21-vit-cifar10-p2-wide`
- `test26-vit-cifar100-p2-deeper`
- `test29-stl10-vit`
- `test31-svhn-vit-p2-wide`
- `test32-vit-cifar100-p2-longer`
- `test33-stl10-vit-longer`
- `test35-tiny-imagenet-vit-deeper`

`test35-tiny-imagenet-vit-deeper` is the strongest Tiny-ImageNet result:

- best `adamw`: `0.282682`
- best `muon_ns`: `0.414323`
- best absolute gain: `+0.131641` accuracy, about `+13.16` percentage points
- best relative gain: about `+46.6%`
- mean `adamw`: `0.249002`
- mean `muon_ns`: `0.370378`
- mean absolute gain: `+0.121376`, about `+12.14` percentage points
- mean relative gain: about `+48.7%`

The later `rt_v43_stream / rt_v43_ns / rt_v6_fdt_metric` sweep for `test35` was
manually stopped at the user's request. The accepted conclusion for this folder
is therefore only `muon_ns > adamw`.

### `02-muon-slightly-better/`

Benchmarks where `muon_ns` passes the acceptance gate, but the margin is smaller
or less clean than the strong-win group.

Included folders:

- `test11-vit-cifar10`
- `test16-mlpmixer-cifar10-refine`
- `test23-vit-cifar10-p2-deeper`
- `test24-vit-cifar100-p2-wide`
- `test30-tiny-imagenet-vit`

### `03-adamw-better-or-failed/`

Benchmarks where `adamw` is better, `muon_ns` does not clear the gate, or the
run is not strong enough to claim a Muon win.

Included folders:

- `test10-pythia-tinystories`
- `test12-mlpmixer-cifar10`
- `test13-resmlp-cifar10`
- `test14-gpt2-tinystories`
- `test15-gpt2-tinystories-lrscan`
- `test17-vit-cifar100`
- `test18-gmlp-cifar10`
- `test20-vit-cifar10-p2`
- `test22-mlpmixer-cifar10-p2`
- `test25-vit-cifar100-p2`
- `test27-tabular-mlp-suite`
- `test28-svhn-vit`
- `test36-tiny-imagenet-mlpmixer`

### `04-archive-hamgnn-pinn-and-incomplete/`

Archive for HamGNN repair work, PINN/PDE exploratory work, incomplete dataset
work, and audit notes. These are useful historical records, but they are not
part of the non-HamGNN `muon_ns > adamw` grouping above.

Included folders and notes:

- `test1-hamgnn-si`
- `test2-pinn`
- `test3-burgers-pinn`
- `test4-poisson-pinn`
- `test5-allen-cahn-pinn`
- `test6-reaction-diffusion-pinn`
- `test7-helmholtz-pinn`
- `test8-hamgnn-toy-muonns`
- `test34-flowers102-vit`
- `v4_3_hamgnn_repair_and_73239_audit.md`
- `v4_3_repair_73239_and_test2_pinn_audit.md`

## Benchmark families

- CIFAR-10 ViT family: `test11`, `test19`, `test21`, `test23`.
  These are the same CIFAR-10 ViT line with different model size, patch size,
  depth/width, and training settings.
- CIFAR-100 ViT family: `test17`, `test24`, `test25`, `test26`, `test32`.
  These are harder CIFAR-100 image-classification runs with different ViT
  recipes and training lengths.
- Tiny-ImageNet ViT family: `test30`, `test35`.
  Both use Tiny-ImageNet-200; `test35` is the deeper/wider harder follow-up.
- STL10 ViT family: `test29`, `test33`.
  Both use STL10; `test33` is the longer/harder training follow-up.
- SVHN ViT family: `test28`, `test31`.
  The weaker SVHN recipe failed, while the stronger `p2-wide` retry passed.
- MLP-Mixer family: `test12`, `test16`, `test22`, `test36`.
  This family is mixed; only the refined CIFAR-10 setting passed.
- Text / LLM family: `test10`, `test14`, `test15`.
  Current landed results do not support `muon_ns > adamw`.
- Tabular MLP: `test27`.
  Real tabular datasets were tried and rejected under the gate.
- HamGNN/PINN archive: `test1` to `test8` plus audit notes.
  These are separated from the non-HamGNN benchmark categorization.

## Main status files

- `BENCHMARK_STATUS.md`: authoritative accepted/rejected status table.
- `NON_HAMGNN_BENCHMARK_SUMMARY.md`: readable benchmark summary.
- `ACCEPTED_BENCHMARKS_SNAPSHOT.md`: compact accepted benchmark snapshot.
- `NEXT_BENCHMARK_CANDIDATES.md`: future benchmark ideas, not current results.
