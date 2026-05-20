# GitHub Repository Layout

This file records the repository organization after cleanup. The old raw
`testXX-*` layout has been replaced with result-based folders.

## Top-level groups

- `BENCHMARK_STATUS.md`
- `NON_HAMGNN_BENCHMARK_SUMMARY.md`
- `GITHUB_PUSH_PLAN.md`
- `01-muon-strongly-better/`
- `02-muon-slightly-better/`
- `03-adamw-better-or-failed/`
- `04-archive-hamgnn-pinn-and-incomplete/`

## Strong Muon wins

- `01-muon-strongly-better/test19-vit-cifar10-large/`
- `01-muon-strongly-better/test21-vit-cifar10-p2-wide/`
- `01-muon-strongly-better/test26-vit-cifar100-p2-deeper/`
- `01-muon-strongly-better/test29-stl10-vit/`
- `01-muon-strongly-better/test31-svhn-vit-p2-wide/`
- `01-muon-strongly-better/test32-vit-cifar100-p2-longer/`
- `01-muon-strongly-better/test33-stl10-vit-longer/`
- `01-muon-strongly-better/test35-tiny-imagenet-vit-deeper/`

## Smaller accepted Muon wins

- `02-muon-slightly-better/test11-vit-cifar10/`
- `02-muon-slightly-better/test16-mlpmixer-cifar10-refine/`
- `02-muon-slightly-better/test23-vit-cifar10-p2-deeper/`
- `02-muon-slightly-better/test24-vit-cifar100-p2-wide/`
- `02-muon-slightly-better/test30-tiny-imagenet-vit/`

## Rejected or AdamW-favored

- `03-adamw-better-or-failed/test10-pythia-tinystories/`
- `03-adamw-better-or-failed/test12-mlpmixer-cifar10/`
- `03-adamw-better-or-failed/test13-resmlp-cifar10/`
- `03-adamw-better-or-failed/test14-gpt2-tinystories/`
- `03-adamw-better-or-failed/test15-gpt2-tinystories-lrscan/`
- `03-adamw-better-or-failed/test17-vit-cifar100/`
- `03-adamw-better-or-failed/test18-gmlp-cifar10/`
- `03-adamw-better-or-failed/test20-vit-cifar10-p2/`
- `03-adamw-better-or-failed/test22-mlpmixer-cifar10-p2/`
- `03-adamw-better-or-failed/test25-vit-cifar100-p2/`
- `03-adamw-better-or-failed/test27-tabular-mlp-suite/`
- `03-adamw-better-or-failed/test28-svhn-vit/`
- `03-adamw-better-or-failed/test36-tiny-imagenet-mlpmixer/`

## Archive

- `04-archive-hamgnn-pinn-and-incomplete/test1-hamgnn-si/`
- `04-archive-hamgnn-pinn-and-incomplete/test2-pinn/`
- `04-archive-hamgnn-pinn-and-incomplete/test3-burgers-pinn/`
- `04-archive-hamgnn-pinn-and-incomplete/test4-poisson-pinn/`
- `04-archive-hamgnn-pinn-and-incomplete/test5-allen-cahn-pinn/`
- `04-archive-hamgnn-pinn-and-incomplete/test6-reaction-diffusion-pinn/`
- `04-archive-hamgnn-pinn-and-incomplete/test7-helmholtz-pinn/`
- `04-archive-hamgnn-pinn-and-incomplete/test8-hamgnn-toy-muonns/`
- `04-archive-hamgnn-pinn-and-incomplete/test34-flowers102-vit/`

## Notes

- `test35` is now accepted under the user's gate because the landed original
  `muon_ns` formal block decisively beats the landed `adamw` formal block; the
  later `rt_v43_stream / rt_v43_ns / rt_v6_fdt_metric` comparison sweep was
  manually cancelled at the user's request after the first landed
  `rt_v43_stream` seed.
- `test34` is blocked on dataset provisioning and remains in the archive.
