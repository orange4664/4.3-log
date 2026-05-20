# test34-flowers102-vit analysis

## Benchmark definition

- model: ViT with patch embedding on `224x224` images
- data: real Flowers102
- benchmark type: non-HamGNN, non-toy, matrix-heavy visual classification
- stress axis: more realistic image scale than CIFAR/STL10 while still far
  below ImageNet cost
- acceptance gate: benchmark is accepted only if original `muon_ns` beats
  `adamw`

## Motivation

The current accepted set already shows a strong ViT pattern on CIFAR, STL10,
and Tiny-ImageNet. Flowers102 extends that line to a more realistic image
setting with larger inputs and a nontrivial label structure, but without the
operational cost of full ImageNet.

This is also a better fit for the user's constraint than returning to toy PINN
setups or tiny text models.

## Current status

Scaffolded locally and synced to the cluster workspace:

- `/data/run01/scwb923/4.3-log/test34-flowers102-vit`

Smoke has been submitted on `gpu_h100`:

- job `73445`

That first smoke attempt did not reach training. It failed during
`torchvision.datasets.Flowers102(..., download=True)` because the cluster-side
HTTP request timed out.

## Next action

- provision Flowers102 data outside the training job
- rerun smoke against cached local data only
- accept or reject promotion based on the smoke gate once summaries exist
