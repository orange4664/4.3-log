# 02 - Muon slightly better

This folder contains benchmarks where original `muon_ns` beats `adamw`, but the
margin is smaller or less clean than the strong-win group.

## Included benchmarks

| Folder | Benchmark | Main conclusion |
| --- | --- | --- |
| `test11-vit-cifar10` | CIFAR-10 compact ViT | Passed smoke/formal gate. |
| `test16-mlpmixer-cifar10-refine` | CIFAR-10 refined MLP-Mixer | Passed after recipe refinement. |
| `test23-vit-cifar10-p2-deeper` | CIFAR-10 ViT, deeper patch-size 2 | `muon_ns` beats `adamw`, but margin is moderate. |
| `test24-vit-cifar100-p2-wide` | CIFAR-100 ViT, wider patch-size 2 | `muon_ns` passes, margin is moderate. |
| `test30-tiny-imagenet-vit` | Tiny-ImageNet ViT | Accepted, but smaller gain than `test35`. |

These runs are still valid positive evidence for `muon_ns > adamw`; they are
separated from `01-muon-strongly-better` only to make the result strength easier
to read.
