# SQ-RT-Muon v4.1: Fine-Temperature Self-Quench

v4.1 is a small targeted update to v4. It does **not** change the RT-Muon geometry or introduce a new algorithm family.

The goal is to keep the v3/v4 large-learning-rate AUC advantage while removing the small residual final-loss gap against exact Muon in lowrank/mixed toys.

## What changed from v4

1. **Finer beta grid**

v4 used:

```text
0, 0.25, 0.5, 1, 2, 4
```

v4.1 defaults to:

```text
0, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 4
```

The motivation is that late-stage / small-LR regimes often should be extremely close to Muon, but not necessarily exactly beta=0. A coarse grid forces the thermostat to jump between exact Muon and a relatively large spectral perturbation.

2. **Stronger self-quench threshold presets**

v4 showed that self-quench improves final loss. v4.1 explicitly includes higher thresholds:

```text
phi_threshold = 0, 0.001, 0.005, 0.01, 0.02
```

inside a 9-config budget-matched SQ grid. The highest-threshold entries are intended to produce beta-zero-rate near 0.95--1.0 in final-loss-selected low-signal regimes.

3. **LR-regime scorecards**

v4.1 adds:

```text
lr_regime_scorecard.csv
useful_lr_window.csv
```

These make the observed regime structure explicit: very small LR can favor exact Muon final loss, while medium/large LR favors RT/SQ-RT in both AUC and stability.

## Core update remains unchanged

For each candidate beta, v4.1 still computes:

```text
phi(beta) = crossfit_gain(beta)
            - lambda_diff   * diffusion_cost(beta)
            - lambda_uncert * replica_uncertainty(beta)
            - lambda_tube   * tube_penalty(beta)
```

and selects beta by maximizing phi. If the best phi is not above the threshold, the method restores exact Muon:

```text
beta = 0, w_i = 1, Q = U V^T.
```

## Main files

```text
experiments/run_sq_rt_v41_bench.py
rtmuon/optimizers.py
rtmuon/toy.py
scripts/submit_v41_budgetmatched.sbatch
scripts/submit_v41_threshold_sweep.sbatch
scripts/submit_v41_with_ablations.sbatch
```

## Main evaluation questions

1. Does v4.1 preserve the medium/large-LR AUC advantage over Muon and positive-p HTMuon-like baselines?
2. Does v4.1 reduce the lowrank/mixed exact-final gap below 0?
3. At eps = 0, 0.5%, 1%, and 2%, does `muon_quality_constrained_auc.csv` show lower constrained AUC than Muon?
4. Do the high-threshold configurations show larger `beta_zero_rate` and smaller `mean_abs_w_minus_1` in final-loss-selected small-LR regimes?

## Interpreting outcomes

A strong v4.1 result is:

```text
AUC(SQ-RT v4.1) < AUC(Muon)
final(SQ-RT v4.1) <= final(Muon)
```

in all four toy settings. A near-win is still useful if exact final is within 0.1% of Muon while constrained AUC is substantially lower.
