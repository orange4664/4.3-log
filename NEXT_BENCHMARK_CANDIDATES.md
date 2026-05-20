# Next benchmark candidates

This note records harder non-HamGNN benchmark directions that are still small
enough to be practical on the current cluster budget.

Selection constraints from the thread:

- avoid HamGNN because the user already validated it separately
- prefer real datasets and real models
- avoid toy models unless the matrix path is intentionally made complex
- accept a benchmark only if original `muon_ns` beats `adamw`

## Priority 1: do not add new runs until comparison goals are explicit

The previous real-vision gate runs have already been classified in the
repository:

- `test31-svhn-vit-p2-wide` is accepted under `01-muon-strongly-better/`
- `test32-vit-cifar100-p2-longer` is accepted under
  `01-muon-strongly-better/`
- `test35-tiny-imagenet-vit-deeper` is accepted under
  `01-muon-strongly-better/`

The next benchmark should be chosen only after deciding whether the goal is
architecture coverage, physics/PDE relevance, or optimizer tuning beyond
`muon_ns vs adamw`.

## Priority 2: small scientific benchmark with stronger precedent in Muon papers

### Candidate: operator-learning PDE benchmark

Use a small neural-operator or transformer recipe on a subset of PDEBench /
PDEArena-style data rather than a tiny PINN toy.

Why this is worth trying:

- BCAT reports Muon plus AdamW ablations on real fluid-dynamics foundation-model
  training rather than only on toy equations
- this is much closer to the scientific-learning use case discussed in Muon
  follow-up papers than the earlier simple PINN tests

Practical compromise:

- do not start from the full BCAT scale
- choose one downstream task and one smaller model
- keep each optimizer sweep in its own folder

## Priority 3: one more medium real-vision dataset beyond STL10 and Tiny-ImageNet

### Candidate: Oxford-IIIT Pet or Flowers102 with ViT

Why:

- still real image datasets
- larger and less trivial than CIFAR-style setups
- far cheaper than full ImageNet
- compatible with the same matrix-heavy ViT family already giving accepted
  benchmarks

This path is lower risk than inventing a new architecture family immediately.

## Priority 4: material-science benchmark only if setup cost stays bounded

### Candidate: a small QM9/QH9-style graph benchmark

Why:

- real scientific data
- closer to the physical-learning motivation in the thread
- much cheaper than full Matbench-Discovery or large interatomic-potential
  recipes

Why not first:

- it needs a clean non-HamGNN graph baseline design
- setup cost is higher than continuing the current ViT line

## Source anchors

- BCAT: https://arxiv.org/abs/2501.18972
- SpecMuon: https://arxiv.org/abs/2602.16167
- AdaMuon: https://arxiv.org/abs/2507.11005

## Immediate next action

1. keep the current GitHub layout stable
2. decide whether the next comparison is `muon_ns vs adamw` only or includes
   `rt_v43_*`
3. add one new real-data benchmark folder only after that decision
