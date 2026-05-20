# GitHub Push Plan

This file is a staging plan for the eventual GitHub push for the non-HamGNN
benchmark search in this thread.

## Include Now

- `BENCHMARK_STATUS.md`
- `NON_HAMGNN_BENCHMARK_SUMMARY.md`
- `GITHUB_PUSH_PLAN.md`
- `test10-pythia-tinystories/`
- `test11-vit-cifar10/`
- `test12-mlpmixer-cifar10/`
- `test13-resmlp-cifar10/`
- `test14-gpt2-tinystories/`
- `test15-gpt2-tinystories-lrscan/`
- `test16-mlpmixer-cifar10-refine/`
- `test17-vit-cifar100/`
- `test18-gmlp-cifar10/`
- `test19-vit-cifar10-large/`
- `test20-vit-cifar10-p2/`
- `test21-vit-cifar10-p2-wide/`
- `test22-mlpmixer-cifar10-p2/`
- `test23-vit-cifar10-p2-deeper/`
- `test24-vit-cifar100-p2-wide/`
- `test25-vit-cifar100-p2/`
- `test26-vit-cifar100-p2-deeper/`
- `test27-tabular-mlp-suite/`
- `test28-svhn-vit/`
- `test29-stl10-vit/`
- `test30-tiny-imagenet-vit/`
- `test31-svhn-vit-p2-wide/`
- `test32-vit-cifar100-p2-longer/`
- `test33-stl10-vit-longer/`
- `test34-flowers102-vit/`
- `test35-tiny-imagenet-vit-deeper/`
- `test36-tiny-imagenet-mlpmixer/`

## Accepted Benchmarks In This Set

- `test11-vit-cifar10/`
- `test16-mlpmixer-cifar10-refine/`
- `test19-vit-cifar10-large/`
- `test21-vit-cifar10-p2-wide/`
- `test23-vit-cifar10-p2-deeper/`
- `test24-vit-cifar100-p2-wide/`
- `test26-vit-cifar100-p2-deeper/`
- `test29-stl10-vit/`
- `test30-tiny-imagenet-vit/`
- `test33-stl10-vit-longer/`

## Pending Benchmarks In This Set

- `test31-svhn-vit-p2-wide/`
- `test32-vit-cifar100-p2-longer/`
- `test34-flowers102-vit/`
- `test35-tiny-imagenet-vit-deeper/`
- `test36-tiny-imagenet-mlpmixer/`

## Rejected Benchmarks In This Set

- `test10-pythia-tinystories/`
- `test12-mlpmixer-cifar10/`
- `test13-resmlp-cifar10/`
- `test14-gpt2-tinystories/`
- `test15-gpt2-tinystories-lrscan/`
- `test17-vit-cifar100/`
- `test18-gmlp-cifar10/`
- `test20-vit-cifar10-p2/`
- `test22-mlpmixer-cifar10-p2/`
- `test25-vit-cifar100-p2/`
- `test27-tabular-mlp-suite/`
- `test28-svhn-vit/`

## Exclude

- `test1-hamgnn-si/`
- `test2-pinn/`
- `test3-burgers-pinn/`
- `test4-poisson-pinn/`
- `test5-allen-cahn-pinn/`
- `test6-reaction-diffusion-pinn/`
- `test7-helmholtz-pinn/`
- `test8-hamgnn-toy-muonns/`
- `test9-hamgnn-sacada/`
- earlier HamGNN repair notes that are not needed for the non-HamGNN benchmark
  search

## Notes

- `test23` and `test24` already have enough landed formal evidence to count as
  accepted under the user's gate even though their Slurm jobs may still be
  finishing higher-learning-rate or later method blocks.
- `test26` is now accepted because the fully landed `lr=0.001` formal block
  clearly favors original `muon_ns`.
- `test29` now has decisive accepted low-learning-rate formal evidence even
  though job `73428` is still running later blocks.
- `test30` is now accepted from the landed low-learning-rate formal block.
- `test33` is now accepted from landed gate-only evidence.
- `test31` and `test32` passed smoke and still need landed `muon_ns`
  gate-only evidence.
- `test35` passed smoke and is in formal.
- `test36` smoke is submitted and still pending in queue.
- `test34` is blocked on dataset provisioning and should not be treated as a
  completed run.
- Prefer selective `git add` of the directories above instead of broad staging,
  because the worktree contains unrelated benchmark and HamGNN changes.
