# test35 five-optimizer benchmark analysis

Status: scaffolded and submitted; final formal results have not landed yet.

## Benchmark

- dataset: Tiny-ImageNet-200
- model: deeper/wider ViT from `test35`
- default shape: patch size `4`, embed dim `320`, depth `10`, heads `10`
- default sweep: learning rates `0.0005,0.001`, seeds `0,1,2`, epochs `6`

## Optimizers

- `adamw`: full AdamW baseline.
- `muon_ns`: original Muon Newton-Schulz optimizer from the HamGNN adapter.
- `muon_adamw`: HamGNN-style MuonAdamW hybrid. Muon-friendly matrix params use
  `muon_ns`; non-matrix/norm/bias params use AdamW.
- `rt_v43_stream`: original v4.3 streaming RT optimizer.
- `rt_v43_adamw`: v4.3 streaming RT on Muon-friendly matrix params plus AdamW
  on remaining params.

## How to run

On the N46H1 cluster, from the synced repository:

```bash
sbatch scripts/submit_test35_five_optimizers_h100.sbatch
```

The script writes raw results under `runs_five_optimizers/` and regenerates this
analysis file after the job finishes.

## Current analysis

No five-optimizer run has landed yet. The earlier accepted result remains:

- `muon_ns` strongly beats `adamw` on `test35`
- best `adamw`: `0.282682`
- best `muon_ns`: `0.414323`
- best absolute gain: about `+13.16` percentage points

The five-optimizer benchmark is intended to compare whether the hybrid variants
preserve or improve that signal.

## Cluster status

- smoke job `73811`: completed; all five requested optimizers wrote
  `summary.json`
- formal job `73812`: submitted on `gpu_h100`; final results pending
