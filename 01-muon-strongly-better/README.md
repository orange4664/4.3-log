# 01 - Muon strongly better

This folder contains benchmarks where original `muon_ns` is clearly better
than `adamw` under the current acceptance rule.

## Included benchmarks

| Folder | Benchmark | Main conclusion |
| --- | --- | --- |
| `test19-vit-cifar10-large` | CIFAR-10 ViT, larger model | Strong `muon_ns` win over `adamw`. |
| `test21-vit-cifar10-p2-wide` | CIFAR-10 ViT, patch-size 2 wide model | Strong all-seed `muon_ns` win. |
| `test26-vit-cifar100-p2-deeper` | CIFAR-100 ViT, deeper model | Strong `lr=0.001` `muon_ns` block. |
| `test29-stl10-vit` | STL10 ViT | `muon_ns` beats strongest landed `adamw` seed. |
| `test31-svhn-vit-p2-wide` | SVHN ViT, stronger retry | Large `muon_ns` win after weaker SVHN recipe failed. |
| `test32-vit-cifar100-p2-longer` | CIFAR-100 ViT, longer training | Strong accepted `muon_ns` rerun block. |
| `test33-stl10-vit-longer` | STL10 ViT, longer training | Large `muon_ns` win over strongest `adamw`. |
| `test35-tiny-imagenet-vit-deeper` | Tiny-ImageNet ViT, deeper/wider | Largest Tiny-ImageNet win in this repository. |

## `test35` improvement

`test35-tiny-imagenet-vit-deeper` compares landed formal `adamw` and
`muon_ns` runs on real Tiny-ImageNet-200.

- best `adamw`: `0.282682`
- best `muon_ns`: `0.414323`
- best absolute gain: `+0.131641`, about `+13.16` percentage points
- best relative gain: about `+46.6%`
- mean `adamw`: `0.249002`
- mean `muon_ns`: `0.370378`
- mean absolute gain: `+0.121376`, about `+12.14` percentage points
- mean relative gain: about `+48.7%`

The later `rt_v43_stream / rt_v43_ns / rt_v6_fdt_metric` optimizer comparison
was stopped manually, so it is not part of the accepted claim here.
