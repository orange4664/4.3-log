# HamGNN RT/FDT-Metric integration plan

HamGNN training is launched with `HamGNN2.0 --config config.yaml`.  The public config documentation places learning rate, LR decay, gradient clipping and epoch controls under `optim_params`, and the default optimizer in current `Model.py` is AdamW.  This adapter preserves the HamGNN training loop and only replaces `configure_optimizers()` at runtime.

## Why temporal FDT for HamGNN

HamGNN primary training commonly uses batch size 1, so explicit two-microbatch replicas are not generally available without modifying the training step and data pipeline.  The first HamGNN experiment therefore uses temporal replicas:

- signal: current gradient / momentum statistics;
- fluctuation: innovation relative to EMA gradient;
- metric temperature: row/column reliability from signal-to-noise.

This keeps the integration source-local and compatible with Lightning automatic optimization.

## Headline comparison

1. AdamW: current HamGNN default.
2. Muon-stream: streaming-power Muon baseline.
3. RT-v4.3-stream: spectral thermostat control.
4. FDT-Metric Muon: marginal temperature metric only.
5. FDT-Metric RT-v6: marginal metric + gated spectral excitation.
6. FDT-Metric RT-v6 period4: cost-controlled variant.

Primary metrics should be Hamiltonian validation MAE, training loss/AUC if logs are dense enough, final test Hamiltonian MAE, runtime, and optimizer diagnostics.
