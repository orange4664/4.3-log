#!/usr/bin/env python3
"""Lightweight result collector for HamGNN TensorBoard event dirs.

It extracts scalar tags from TensorBoard event files when tensorboard is installed.
If unavailable, it still summarizes optimizer JSONL diagnostics.
"""
from __future__ import annotations
import argparse, glob, json, os
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
parser.add_argument("--out", default="hamgnn_rt_summary.json")
args = parser.parse_args()
summary = {"root": args.root, "runs": []}
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception:
    EventAccumulator = None
for event in glob.glob(os.path.join(args.root, "**", "events.out.tfevents.*"), recursive=True):
    run_dir = os.path.dirname(event)
    rec = {"run_dir": run_dir, "event_file": event, "scalars": {}}
    if EventAccumulator is not None:
        try:
            ea = EventAccumulator(run_dir)
            ea.Reload()
            for tag in ea.Tags().get("scalars", []):
                vals = ea.Scalars(tag)
                if vals:
                    rec["scalars"][tag] = {"last": vals[-1].value, "min": min(v.value for v in vals), "max": max(v.value for v in vals), "n": len(vals)}
        except Exception as e:
            rec["error"] = str(e)
    summary["runs"].append(rec)
# optimizer diagnostics
opt_diags = []
for path in glob.glob(os.path.join(args.root, "**", "*_optimizer_diag.jsonl"), recursive=True):
    vals = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            for k, v in d.items():
                if isinstance(v, (int, float)):
                    vals[k].append(float(v))
    opt_diags.append({"path": path, "mean": {k: sum(v)/len(v) for k, v in vals.items() if v}})
summary["optimizer_diagnostics"] = opt_diags

# GPU monitor summaries
gpu_logs = []
for path in glob.glob(os.path.join(args.root, "**", "gpu_monitor.csv"), recursive=True):
    vals = defaultdict(list)
    try:
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            for line in f:
                parts = [x.strip() for x in line.strip().split(",")]
                if len(parts) < len(header):
                    continue
                row = dict(zip(header, parts))
                for key in ["utilization.gpu", "utilization.memory", "memory.used", "memory.total", "power.draw", "temperature.gpu"]:
                    try:
                        vals[key].append(float(row.get(key, "nan")))
                    except Exception:
                        pass
    except Exception:
        continue
    gpu_logs.append({"path": path, "mean": {k: sum(v)/len(v) for k, v in vals.items() if v}, "max": {k: max(v) for k, v in vals.items() if v}})
summary["gpu_monitor"] = gpu_logs

with open(args.out, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, sort_keys=True)
print(args.out)
