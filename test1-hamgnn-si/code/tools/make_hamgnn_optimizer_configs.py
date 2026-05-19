#!/usr/bin/env python3
"""Create HamGNN config variants for optimizer comparison."""
from __future__ import annotations
import argparse, copy, os, yaml

METHOD_BLOCKS = {
    "adamw": {"optimizer": "adamw"},
    "muon_ns": {"optimizer": "muon_ns", "muon_momentum": 0.95, "muon_ns_steps": 5, "muon_nesterov": True},
    "muon_stream": {"optimizer": "muon_stream", "rt_stream_k": 0, "rt_power_iters": 1},
    "ht_stream": {"optimizer": "ht_stream", "rt_stream_k": 0, "rt_power_iters": 1, "ht_power": 0.125},
    "mclip_stream": {"optimizer": "mclip_stream", "rt_stream_k": 0, "rt_power_iters": 1, "mclip_delta": 0.25},
    "rt_v43_stream": {
        "optimizer": "rt_v43_stream", "rt_stream_k": 0, "rt_power_iters": 1,
        "rt_period": 1, "rt_lr_switch": 0.02, "rt_hard_delta": 0.2, "rt_soft_delta": 0.5,
        "rt_hard_phi_threshold": 0.0, "rt_soft_phi_threshold": 0.0,
    },
    "muon_stream_fdt_metric": {
        "optimizer": "muon_stream_fdt_metric", "rt_stream_k": 0, "rt_power_iters": 1,
        "metric_alpha_row": 0.25, "metric_alpha_col": 0.25,
        "metric_tmin": 0.5, "metric_tmax": 2.0, "metric_lambda_noise": 1.0,
    },
    "rt_v6_fdt_metric": {
        "optimizer": "rt_v6_fdt_metric", "rt_stream_k": 0, "rt_power_iters": 1,
        "rt_period": 1, "rt_lr_switch": 0.02, "rt_hard_delta": 0.2, "rt_soft_delta": 0.5,
        "rt_hard_phi_threshold": 0.0, "rt_soft_phi_threshold": 0.0,
        "metric_alpha_row": 0.25, "metric_alpha_col": 0.25,
        "metric_tmin": 0.5, "metric_tmax": 2.0, "metric_lambda_noise": 1.0,
    },
    "rt_v6_fdt_metric_period4": {
        "optimizer": "rt_v6_fdt_metric", "rt_stream_k": 0, "rt_power_iters": 1,
        "rt_period": 4, "rt_lr_switch": 0.02, "rt_hard_delta": 0.2, "rt_soft_delta": 0.5,
        "rt_hard_phi_threshold": 0.0, "rt_soft_phi_threshold": 0.0,
        "metric_alpha_row": 0.25, "metric_alpha_col": 0.25,
        "metric_tmin": 0.5, "metric_tmax": 2.0, "metric_lambda_noise": 1.0,
    },
}

parser = argparse.ArgumentParser()
parser.add_argument("--base-config", required=True)
parser.add_argument("--outdir", required=True)
parser.add_argument("--methods", default="adamw,muon_ns,rt_v43_stream,rt_v6_fdt_metric")
parser.add_argument("--lrs", default="0.003,0.01,0.03")
parser.add_argument("--tag", default="hamgnn_rt")
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)
with open(args.base_config, "r", encoding="utf-8") as f:
    base = yaml.safe_load(f) or {}
methods = [x.strip() for x in args.methods.split(",") if x.strip()]
lrs = [float(x) for x in args.lrs.split(",") if x.strip()]
manifest = []
for method in methods:
    if method not in METHOD_BLOCKS:
        raise KeyError(f"Unknown method {method}. Known: {sorted(METHOD_BLOCKS)}")
    for lr in lrs:
        cfg = copy.deepcopy(base)
        cfg.setdefault("optim_params", {})
        cfg["optim_params"].update(METHOD_BLOCKS[method])
        cfg["optim_params"]["lr"] = lr
        train_dir = cfg.setdefault("profiler_params", {}).get("train_dir", "./train")
        cfg["profiler_params"]["train_dir"] = os.path.join(train_dir, f"{args.tag}_{method}_lr{lr:g}")
        out = os.path.join(args.outdir, f"config_{method}_lr{lr:g}.yaml")
        with open(out, "w", encoding="utf-8") as fo:
            yaml.safe_dump(cfg, fo, sort_keys=False, allow_unicode=True)
        manifest.append((method, lr, out))
manifest_path = os.path.join(args.outdir, "manifest.tsv")
with open(manifest_path, "w", encoding="utf-8") as f:
    f.write("method\tlr\tconfig\n")
    for method, lr, out in manifest:
        f.write(f"{method}\t{lr:g}\t{out}\n")
print(manifest_path)
