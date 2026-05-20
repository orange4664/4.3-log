from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List


def load_rows(runs_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted(runs_dir.rglob("summary.json")):
        with path.open("r", encoding="utf-8") as f:
            rows.append(json.load(f))
    return rows


def write_csv(rows: Iterable[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "lr",
        "seed",
        "best_test_acc",
        "final_test_acc",
        "final_test_loss",
        "auc_train_loss",
        "steps",
        "time_sec",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def method_stats(rows: List[Dict[str, object]]) -> List[Dict[str, float | str]]:
    by_method: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        method = str(row.get("method", ""))
        if not method:
            continue
        by_method[method].append(float(row.get("best_test_acc", 0.0)))

    stats: List[Dict[str, float | str]] = []
    for method, values in sorted(by_method.items()):
        stats.append(
            {
                "method": method,
                "n": float(len(values)),
                "best": max(values),
                "mean": mean(values),
                "worst": min(values),
            }
        )
    stats.sort(key=lambda x: float(x["best"]), reverse=True)
    return stats


def write_markdown(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = method_stats(rows)
    adamw_best = next((float(s["best"]) for s in stats if s["method"] == "adamw"), None)

    lines = [
        "# test35 five-optimizer benchmark analysis",
        "",
        "## Optimizers",
        "",
        "- `adamw`: full AdamW baseline.",
        "- `muon_ns`: original Muon Newton-Schulz on all matrix-like params with AdamW fallback inside the adapter for non-matrix params.",
        "- `muon_adamw`: HamGNN-style MuonAdamW hybrid; Muon-friendly matrix params use `muon_ns`, remaining params use AdamW.",
        "- `rt_v43_stream`: original v4.3 streaming RT optimizer from the selectorbench / HamGNN adapter line.",
        "- `rt_v43_adamw`: v4.3 on Muon-friendly matrix params plus AdamW on remaining params.",
        "",
        "## Method summary",
        "",
        "| rank | method | runs | best_test_acc | mean_best_test_acc | worst_best_test_acc | best_vs_adamw |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, stat in enumerate(stats, start=1):
        best = float(stat["best"])
        delta = "" if adamw_best is None else f"{best - adamw_best:+.6f}"
        lines.append(
            f"| {i} | `{stat['method']}` | {int(stat['n'])} | {best:.6f} | "
            f"{float(stat['mean']):.6f} | {float(stat['worst']):.6f} | {delta} |"
        )

    lines.extend(
        [
            "",
            "## Raw completed runs",
            "",
            "| method | lr | seed | best_test_acc | final_test_acc | steps | time_sec |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(rows, key=lambda r: (str(r.get("method", "")), float(r.get("lr", 0.0)), int(r.get("seed", 0)))):
        lines.append(
            f"| `{row.get('method', '')}` | {float(row.get('lr', 0.0)):.6g} | "
            f"{int(row.get('seed', 0))} | {float(row.get('best_test_acc', 0.0)):.6f} | "
            f"{float(row.get('final_test_acc', 0.0)):.6f} | {int(row.get('steps', 0))} | "
            f"{float(row.get('time_sec', 0.0)):.1f} |"
        )

    if not rows:
        lines.append("| no completed runs yet |  |  |  |  |  |  |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs_five_optimizers")
    parser.add_argument("--out-md", default="FIVE_OPTIMIZER_BENCHMARK_ANALYSIS.md")
    parser.add_argument("--out-csv", default="runs_five_optimizers/five_optimizer_summary.csv")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    rows = load_rows(runs_dir)
    print(json.dumps(rows, indent=2))
    write_csv(rows, Path(args.out_csv))
    write_markdown(rows, Path(args.out_md))


if __name__ == "__main__":
    main()
