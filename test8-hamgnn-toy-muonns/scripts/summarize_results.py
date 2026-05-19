#!/usr/bin/env python3
from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read_scalars(run_dir: str):
    events = glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True)
    out = {}
    for ev in events:
        try:
            ea = EventAccumulator(ev, size_guidance={"scalars": 0})
            ea.Reload()
            for tag in ea.Tags().get("scalars", []):
                vals = ea.Scalars(tag)
                if vals:
                    out.setdefault(tag, []).extend((v.step, v.value) for v in vals)
        except Exception:
            pass
    return {tag: sorted(vals, key=lambda x: x[0]) for tag, vals in out.items()}


def metric(scalars, *tags):
    vals = []
    for tag in tags:
        vals.extend(scalars.get(tag, []))
    if not vals:
        return None, None, None
    vals = sorted(vals, key=lambda x: x[0])
    return vals[-1][1], min(v for _, v in vals), vals[-1][0]


def gpu_stats(path: str):
    if not os.path.exists(path):
        return {}
    vals, mem, pwr = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            def getnum(keys):
                for k in keys:
                    if k in row and row[k] not in ("", "[N/A]"):
                        try:
                            return float(str(row[k]).strip().split()[0])
                        except Exception:
                            pass
                return None

            u = getnum(["utilization.gpu [%]", "utilization.gpu"])
            m = getnum(["memory.used [MiB]", "memory.used"])
            p = getnum(["power.draw [W]", "power.draw"])
            if u is not None:
                vals.append(u)
            if m is not None:
                mem.append(m)
            if p is not None:
                pwr.append(p)
    return {
        "gpu_util_mean": float(np.mean(vals)) if vals else None,
        "gpu_util_max": float(np.max(vals)) if vals else None,
        "mem_used_max_mib": float(np.max(mem)) if mem else None,
        "power_mean_w": float(np.mean(pwr)) if pwr else None,
    }


def main() -> None:
    root = Path(os.environ["TEST8_ROOT"]).resolve()
    methods = ["adamw", "muon_ns", "rt_v43_stream", "rt_v6_fdt_metric"]
    rows = []
    for method in methods:
        run_dir = root / "train" / method / "version_0"
        pred = run_dir / "prediction_hamiltonian.npy"
        targ = run_dir / "target_hamiltonian.npy"
        test_mae = None
        if pred.exists() and targ.exists():
            test_mae = float(np.mean(np.abs(np.load(pred) - np.load(targ))))
        scalars = read_scalars(str(run_dir))
        train_last, train_best, train_step = metric(scalars, "training/total_loss_epoch", "training/total_loss")
        val_last, val_best, _ = metric(scalars, "validation/total_loss")
        ckpts = glob.glob(str(run_dir / "checkpoints" / "*.ckpt"))
        diag_dirs = sorted(glob.glob(str(root / "diag" / f"toy_{method}_*")))
        diag_dir = Path(diag_dirs[-1]) if diag_dirs else None
        opt_diag = list(diag_dir.glob("*optimizer_diag.jsonl")) if diag_dir else []
        last_diag = {}
        diag_count = 0
        if opt_diag:
            lines = [ln for ln in opt_diag[0].read_text(encoding="utf-8").splitlines() if ln.strip()]
            diag_count = len(lines)
            if lines:
                try:
                    last_diag = json.loads(lines[-1])
                except Exception:
                    last_diag = {}
        row = {
            "method": method,
            "test_mae": test_mae,
            "train_last": train_last,
            "train_best": train_best,
            "val_last": val_last,
            "val_best": val_best,
            "last_epoch_step": train_step,
            "checkpoint": os.path.basename(ckpts[0]) if ckpts else "",
            "diag_records": diag_count,
            "diag_beta": last_diag.get("beta"),
            "diag_accept": last_diag.get("accept"),
            "diag_update_rms": last_diag.get("update_rms"),
            "diag_metric_row_temp_cv": last_diag.get("metric_row_temp_cv"),
            "diag_metric_col_temp_cv": last_diag.get("metric_col_temp_cv"),
        }
        if diag_dir:
            row.update(gpu_stats(str(diag_dir / "gpu_monitor.csv")))
        rows.append(row)

    rows_sorted = sorted(rows, key=lambda r: (float("inf") if r["test_mae"] is None else r["test_mae"]))
    summary_json = root / "toy_muonns_benchmark_summary.json"
    summary_csv = root / "toy_muonns_benchmark_summary.csv"
    summary_json.write_text(json.dumps(rows_sorted, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow(row)
    print(json.dumps(rows_sorted, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
