# 03 - AdamW better or failed

This folder contains benchmarks where `adamw` is better, `muon_ns` fails the
acceptance gate, or the run is not strong enough to claim a Muon win.

## Included benchmarks

| Folder | Benchmark family | Reason for this group |
| --- | --- | --- |
| `test10-pythia-tinystories` | Text / LLM | `adamw` lower validation loss. |
| `test12-mlpmixer-cifar10` | CIFAR-10 MLP-Mixer | Did not pass gate. |
| `test13-resmlp-cifar10` | CIFAR-10 ResMLP | Did not pass gate. |
| `test14-gpt2-tinystories` | Text / LLM | Did not pass gate. |
| `test15-gpt2-tinystories-lrscan` | Text / LLM | Did not pass gate. |
| `test17-vit-cifar100` | CIFAR-100 ViT | Weaker recipe failed. |
| `test18-gmlp-cifar10` | CIFAR-10 gMLP | Did not pass gate. |
| `test20-vit-cifar10-p2` | CIFAR-10 ViT | Formal rejected. |
| `test22-mlpmixer-cifar10-p2` | CIFAR-10 MLP-Mixer | Did not pass gate. |
| `test25-vit-cifar100-p2` | CIFAR-100 ViT | Smoke rejected. |
| `test27-tabular-mlp-suite` | Tabular MLP | Aggregate favors `adamw`. |
| `test28-svhn-vit` | SVHN ViT | Weaker recipe rejected. |
| `test36-tiny-imagenet-mlpmixer` | Tiny-ImageNet MLP-Mixer | Smoke rejected. |

This group is important because it shows `muon_ns` is not uniformly better.
The positive evidence is concentrated mostly in matrix-heavy ViT settings, not
in every architecture family.
