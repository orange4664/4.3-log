# HamGNN RT/FDT-Metric optimizer adapter

This package adds optimizer-only controls for HamGNN without modifying HamGNN source code.  It runs HamGNN through a wrapper:

```bash
python -m hamgnn_rt.run_hamgnn_rt --config config.yaml --diag-dir rt_diag --run-name run1
```

The wrapper monkey-patches `Model.configure_optimizers()` and keeps the normal HamGNN training entry point and callbacks.  HamGNN's public README uses `HamGNN2.0 --config config.yaml` for training, with optimizer settings under `optim_params`; this wrapper preserves that config style.

## Methods

Set `optim_params.optimizer` in the YAML:

```yaml
optim_params:
  lr: 0.01
  lr_decay: 0.5
  lr_patience: 4
  gradient_clip_val: 0.0
  max_epochs: 3000
  min_epochs: 30
  stop_patience: 10
  optimizer: rt_v6_fdt_metric   # adamw, muon_stream, rt_v43_stream, muon_stream_fdt_metric, rt_v6_fdt_metric
  rt_stream_k: 0                # 0 = full small-side rank; use e.g. 32 if memory is high
  rt_power_iters: 1
  rt_period: 1
  rt_lr_switch: 0.02
  metric_alpha_row: 0.25
  metric_alpha_col: 0.25
  metric_tmin: 0.5
  metric_tmax: 2.0
```

Recommended comparison:

```text
adamw
muon_stream
rt_v43_stream
muon_stream_fdt_metric
rt_v6_fdt_metric
rt_v6_fdt_metric_period4
```

Because HamGNN often uses `batch_size=1`, this adapter estimates replica/FDT quantities from temporal gradient innovations rather than explicit two-microbatch gradients.  It is the correct first integration target for Lightning automatic optimization.

## Install/use inside a HamGNN environment

```bash
cd hamgnn_rt_adapter
pip install -e .
python scripts/smoke_optimizer.py
```

Run one config:

```bash
CONFIG=/path/to/config.yaml RUN_NAME=rt_v6 sbatch scripts/submit_hamgnn_rt_single_n46h_h200.sbatch
```

Create and submit a comparison grid:

```bash
BASE_CONFIG=/path/to/base_config.yaml \
METHODS=adamw,muon_stream,rt_v43_stream,muon_stream_fdt_metric,rt_v6_fdt_metric,rt_v6_fdt_metric_period4 \
LRS=0.003,0.01,0.03 \
MAX_PARALLEL=2 \
bash scripts/launch_hamgnn_rt_compare_n46h.sh h200
```

Collect TensorBoard scalars and optimizer diagnostics:

```bash
python tools/collect_hamgnn_tensorboard.py --root $(cat latest_hamgnn_rt_root.txt) --out hamgnn_rt_summary.json
```

## Optimizer diagnostics

Each run writes JSONL diagnostics such as beta, accept, metric row/column temperature CV, update RMS, and matrix shape into `HAMGNN_RT_DIAG_DIR`.

## Important first-pass interpretation

- `muon_stream` is the streaming-power Muon baseline.
- `rt_v43_stream` is the spectral thermostat control.
- `muon_stream_fdt_metric` tests whether the FDT metric helps without spectral RT.
- `rt_v6_fdt_metric` tests whether spectral RT adds value inside the FDT metric.
- `rt_period=4` is the first cost-control variant.


## GPU monitor

The Slurm scripts start `scripts/gpu_monitor.py` automatically and write `gpu_monitor.csv` under each run diagnostics directory. The collector summarizes mean/max GPU utilization, memory, power and temperature in `hamgnn_rt_summary.json`.
