from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def scalar(values: Iterable[Any]) -> List[float]:
    out = []
    for value in values:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            out.append(float(value))
    return out


def method_from_diag_name(path: Path) -> str:
    summary = path / "summary.json"
    if summary.exists():
        try:
            return str(read_json(summary).get("optimizer", path.name))
        except Exception:
            pass
    name = path.name
    prefix = "pythia160_"
    if name.startswith(prefix):
        body = name[len(prefix) :]
        parts = body.rsplit("_", 1)
        return parts[0]
    return name


def gpu_stats(path: Path) -> Dict[str, Any]:
    csv_path = path / "gpu_monitor.csv"
    if not csv_path.exists():
        return {}
    rows = []
    with csv_path.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return {}

    def first_existing(*names: str) -> List[float]:
        for name in names:
            vals = scalar(row.get(name) for row in rows)
            if vals:
                return vals
        return []

    util = first_existing("utilization.gpu [%]", "gpu_util", "util_gpu", "gpu")
    mem = first_existing("memory.used [MiB]", "memory_used_mb", "mem_used", "mem")
    return {
        "gpu_util_mean": mean(util) if util else None,
        "gpu_util_max": max(util) if util else None,
        "gpu_mem_mb_mean": mean(mem) if mem else None,
        "gpu_mem_mb_max": max(mem) if mem else None,
    }


def collect_run(diag_dir: Path) -> Dict[str, Any] | None:
    summary_path = diag_dir / "summary.json"
    metrics_path = diag_dir / "train_metrics.jsonl"
    if not summary_path.exists() and not metrics_path.exists():
        return None

    summary = read_json(summary_path) if summary_path.exists() else {}
    metrics = read_jsonl(metrics_path)
    last = metrics[-1] if metrics else {}
    method = str(summary.get("optimizer") or method_from_diag_name(diag_dir))
    train_losses = scalar(row.get("train_loss") for row in metrics)
    val_rows = [row for row in metrics if "val_loss" in row]
    val_losses = scalar(row.get("val_loss") for row in val_rows)
    step_times = scalar(row.get("step_time_s") for row in metrics)
    grad_norms = scalar(row.get("grad_norm") for row in metrics)

    out = {
        "method": method,
        "diag_dir": str(diag_dir),
        "completed_steps": summary.get("completed_steps", summary.get("last_step", last.get("step"))),
        "train_loss_last": summary.get("train_loss_last", last.get("train_loss")),
        "train_loss_best": summary.get("train_loss_best", min(train_losses) if train_losses else None),
        "train_loss_mean": summary.get("train_loss_mean", mean(train_losses) if train_losses else None),
        "train_loss_auc_step": summary.get("train_loss_auc_step"),
        "last_val_loss": summary.get("last_val_loss", val_losses[-1] if val_losses else None),
        "best_val_loss": summary.get("best_val_loss", min(val_losses) if val_losses else None),
        "val_loss_auc_step": summary.get("val_loss_auc_step"),
        "last_val_ppl": summary.get("last_val_ppl"),
        "best_val_ppl": summary.get("best_val_ppl"),
        "elapsed_s": summary.get("elapsed_s", last.get("elapsed_s")),
        "tokens_seen": summary.get("tokens_seen", last.get("tokens_seen")),
        "tokens_per_s": summary.get("tokens_per_s", last.get("tokens_per_s")),
        "step_time_s_mean": mean(step_times) if step_times else None,
        "step_time_s_std": pstdev(step_times) if len(step_times) > 1 else 0.0 if step_times else None,
        "grad_norm_last": grad_norms[-1] if grad_norms else None,
        "grad_norm_mean": mean(grad_norms) if grad_norms else None,
        "peak_cuda_allocated_mb": summary.get("peak_cuda_allocated_mb"),
        "peak_cuda_reserved_mb": summary.get("peak_cuda_reserved_mb"),
        "diverged": summary.get("diverged", False),
        "diverged_reason": summary.get("diverged_reason", ""),
        "data_source": summary.get("data_source"),
    }
    out.update(gpu_stats(diag_dir))
    return out


def rank_key(row: Dict[str, Any]) -> tuple:
    diverged = bool(row.get("diverged"))
    best_val = row.get("best_val_loss")
    train_mean = row.get("train_loss_mean")
    elapsed = row.get("elapsed_s")
    return (
        1 if diverged else 0,
        float(best_val) if isinstance(best_val, (int, float)) else math.inf,
        float(train_mean) if isinstance(train_mean, (int, float)) else math.inf,
        float(elapsed) if isinstance(elapsed, (int, float)) else math.inf,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    diag_root = root / "diag"
    runs = []
    for path in sorted(diag_root.glob("pythia160_*")):
        if path.is_dir():
            row = collect_run(path)
            if row is not None:
                runs.append(row)

    ranked = sorted(runs, key=rank_key)
    payload = {
        "root": str(root),
        "ranked_methods": [row["method"] for row in ranked],
        "runs": ranked,
    }

    out_json = Path(args.out_json) if args.out_json else root / "benchmark_summary.json"
    out_csv = Path(args.out_csv) if args.out_csv else root / "benchmark_summary.csv"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = sorted({key for row in ranked for key in row})
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ranked)

    print(json.dumps({"out_json": str(out_json), "out_csv": str(out_csv), "ranked_methods": payload["ranked_methods"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
