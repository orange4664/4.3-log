from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    for path in root.glob("**/summary.json"):
        if path.name != "summary.json":
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        row["summary_path"] = str(path)
        rows.append(row)

    rows = sorted(rows, key=lambda r: (r["method"], r["lr"], r["seed"]))
    (root / "all_runs.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    by_method = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)

    method_rows = []
    for method, group in sorted(by_method.items()):
        accs = [float(x["best_test_acc"]) for x in group]
        finals = [float(x["final_test_acc"]) for x in group]
        losses = [float(x["auc_train_loss"]) for x in group]
        row = {
            "method": method,
            "runs": len(group),
            "mean_best_test_acc": mean(accs),
            "std_best_test_acc": pstdev(accs) if len(accs) > 1 else 0.0,
            "mean_final_test_acc": mean(finals),
            "mean_auc_train_loss": mean(losses),
            "best_single_run_acc": max(accs),
        }
        method_rows.append(row)

    method_rows = sorted(method_rows, key=lambda r: (-r["mean_best_test_acc"], r["mean_auc_train_loss"]))
    with (root / "summary_by_method.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(method_rows[0].keys()) if method_rows else ["method"])
        writer.writeheader()
        for row in method_rows:
            writer.writerow(row)

    print(json.dumps(method_rows, indent=2))


if __name__ == "__main__":
    main()
