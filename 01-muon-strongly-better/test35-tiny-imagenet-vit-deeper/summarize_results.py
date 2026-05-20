from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    rows = []
    for path in sorted(root.rglob("summary.json")):
        if path.name != "summary.json" or "runs" not in str(path):
            continue
        with path.open("r", encoding="utf-8") as f:
            rows.append(json.load(f))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
