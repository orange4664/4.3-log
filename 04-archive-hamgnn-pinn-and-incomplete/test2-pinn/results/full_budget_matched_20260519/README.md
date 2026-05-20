# full_budget_matched_20260519

This directory contains the completed local full `budget_matched` run of the `v4.3` selector benchmark.

Run command:

```bash
python experiments/run_sq_rt_v43_selector_bench.py --mode budget_matched --outdir ..\results\full_budget_matched_20260519 --workers 16 --seeds 16 --steps 200
```

## High-level conclusion

The `sq_selector_v43` family completed the full benchmark and showed strong gains over Muon on AUC:

- In `lr_regime_win_counts.csv`, `selector_beats_muon` is:
  - `24/24` for `auc_loss` over all LR/setting pairs
  - `16/16` for `auc_loss` on `lr >= 0.05`
- For `final_loss`, `selector_beats_muon` is:
  - `22/24` over all LR/setting pairs
  - `16/16` on `lr >= 0.05`

## Positive-HT comparison

From `positive_ht_comparison.csv`:

- For `auc_loss`, `sq_selector_v43` beats the best positive-p HT baseline on all four settings:
  - `fullrank`
  - `lowrank`
  - `mixed`
  - `spiked_noise`
- For `final_loss`, the selector is competitive, but not universally better than the constrained HT comparison in the same way.

## Files to inspect first

- `positive_ht_comparison.csv`
- `lr_regime_win_counts.csv`
- `oracle_comparison.csv`
- `muon_quality_constrained_auc.csv`
- `selector_mode_diagnostics.csv`
- `summary_by_config.csv`
- `records.csv`
