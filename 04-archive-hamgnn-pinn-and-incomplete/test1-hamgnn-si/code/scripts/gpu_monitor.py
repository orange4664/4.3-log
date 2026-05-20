#!/usr/bin/env python3
import argparse, csv, os, subprocess, time

FIELDS = [
    "timestamp", "gpu_index", "utilization.gpu", "utilization.memory",
    "memory.used", "memory.total", "power.draw", "temperature.gpu",
]

QUERY = "index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu"

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--interval", type=float, default=5.0)
args = parser.parse_args()
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(FIELDS + ["CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID"])
    while True:
        try:
            raw = subprocess.check_output([
                "nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"
            ], text=True, stderr=subprocess.DEVNULL)
            ts = time.time()
            for line in raw.strip().splitlines():
                vals = [x.strip() for x in line.split(",")]
                if not vals or len(vals) < 7:
                    continue
                writer.writerow([ts] + vals + [
                    os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    os.environ.get("SLURM_JOB_ID", ""),
                    os.environ.get("SLURM_ARRAY_TASK_ID", ""),
                ])
            f.flush()
        except Exception:
            pass
        time.sleep(args.interval)
