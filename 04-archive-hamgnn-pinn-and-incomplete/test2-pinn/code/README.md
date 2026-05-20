# SQ-RT-Muon v4.3 selector bench

This package continues the v4.x SQ-RT-Muon line.  It does **not** introduce a new penalty or a new singular-value transform.  It adds a clean selector benchmark for the empirical regime found in v1--v4.2:

- small learning rate / exact-final-loss regime: hard-quench SQ-RT v4.1 is safest, often matching Muon exactly;
- medium-large learning rate / speed regime: soft-quench SQ-RT v4.2 gives better AUC and wider useful learning-rate window.

The new family is:

```text
sq_selector_v43
```

It uses hard-quench when `lr < lr_switch`, and soft-quench when `lr >= lr_switch`.  The default switch is `0.05`, matching the observed transition where SQ/RT starts to dominate Muon and HTMuon-like baselines on both AUC and final loss.

This is a selector/scorecard step.  It tests whether the hard/soft tradeoff can be turned into a fair, budget-matched optimizer family before changing the theory again.

## What is compared

The default benchmark includes:

```text
muon
htmuon_like p-grid
rt_fixed
sq_hard_v41
sq_soft_v42
sq_selector_v43
```

Non-Muon families have the same number of algorithm configs by default.

## Run on BSCC-M9 CPU queue

```bash
cd /publicfs10/$USER
unzip rt_muon_v4_3_selectorbench.zip
cd rt_muon_v4_3_selectorbench
python -m pip install --user -r requirements.txt
bash scripts/smoke_test_v43.sh
sbatch scripts/submit_v43_selector_budgetmatched.sbatch
```

The Slurm script requests 32 CPU cores and the Python script uses `--workers ${SLURM_CPUS_PER_TASK:-32}`.  BLAS threads are pinned to one thread to avoid oversubscription.

## Key outputs

The main result directory is normally:

```text
results/sq_rt_v43_selector_<JOBID>
```

Look at:

```text
oracle_comparison.csv
positive_ht_comparison.csv
muon_quality_constrained_auc.csv
lr_regime_scorecard.csv
lr_regime_win_counts.csv
selector_mode_diagnostics.csv
search_budget.csv
summary_by_config.csv
records.csv
```

The key questions are:

1. Does `sq_selector_v43` keep v4.1's exact-final stability at small LR?
2. Does it keep v4.2's AUC / LR robustness at medium-large LR?
3. In `muon_quality_constrained_auc.csv`, does it beat Muon at epsilon = 0 or 0.5%?
4. In `lr_regime_scorecard.csv`, does it beat Muon and positive-p HTMuon-like for `lr >= 0.05` on AUC and final loss?

