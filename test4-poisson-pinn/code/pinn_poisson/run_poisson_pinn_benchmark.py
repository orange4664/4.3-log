#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[3]
HAMGNN_OPT_PATH = REPO_ROOT / "test1-hamgnn-si" / "code" / "hamgnn_rt" / "optimizers.py"


def load_hamgnn_optimizer():
    spec = importlib.util.spec_from_file_location("hamgnn_rt_optimizers", HAMGNN_OPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load repaired optimizer from {HAMGNN_OPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.HamGNNMuonFDTOptimizer


HamGNNMuonFDTOptimizer = load_hamgnn_optimizer()


@dataclass
class RunConfig:
    optimizer: str
    seed: int
    lr: float
    steps: int
    width: int
    depth: int
    n_interior: int
    device: str
    lr_switch: float
    k_freq: int


class PoissonPINN(nn.Module):
    def __init__(self, width: int, depth: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = 1
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.Tanh())
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def exact_solution(x: torch.Tensor, k_freq: int) -> torch.Tensor:
    k = float(k_freq) * math.pi
    return torch.sin(k * x) + 0.2 * torch.sin(3.0 * k * x)


def rhs_f(x: torch.Tensor, k_freq: int) -> torch.Tensor:
    k = float(k_freq) * math.pi
    return (k * k) * torch.sin(k * x) + 0.2 * (3.0 * k) * (3.0 * k) * torch.sin(3.0 * k * x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_samples(cfg: RunConfig, device: torch.device) -> Dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(cfg.seed)
    x_i = torch.rand(cfg.n_interior, 1, generator=g)
    x_b = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    return {"x_i": x_i.to(device), "x_b": x_b.to(device)}


def full_loss(model: PoissonPINN, batch: Dict[str, torch.Tensor], k_freq: int) -> Tuple[torch.Tensor, Dict[str, float]]:
    x_req = batch["x_i"].detach().clone().requires_grad_(True)
    u = model(x_req)
    u_x = torch.autograd.grad(u, x_req, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_req, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    residual = -u_xx - rhs_f(x_req, k_freq)
    pde = residual.pow(2).mean()
    u_b = model(batch["x_b"])
    bc = u_b.pow(2).mean()
    total = pde + bc
    return total, {"pde_loss": float(pde.detach().cpu().item()), "bc_loss": float(bc.detach().cpu().item())}


def evaluate(model: PoissonPINN, k_freq: int, device: torch.device, grid_n: int = 257) -> Dict[str, float]:
    with torch.no_grad():
        x = torch.linspace(0.0, 1.0, grid_n, device=device).unsqueeze(1)
        pred = model(x)
        ref = exact_solution(x, k_freq)
        mse = (pred - ref).pow(2).mean()
        rel_l2 = torch.linalg.norm(pred - ref) / torch.linalg.norm(ref).clamp_min(1e-12)
    return {"grid_mse": float(mse.cpu().item()), "grid_rel_l2": float(rel_l2.cpu().item())}


def build_optimizer(model: PoissonPINN, optimizer_name: str, lr: float, diag_dir: Path, lr_switch: float):
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    common = dict(
        lr=lr,
        weight_decay=0.0,
        beta=0.95,
        adam_beta1=0.9,
        adam_beta2=0.999,
        ns_steps=5,
        rt_period=1,
        lr_switch=lr_switch,
        hard_delta=0.35,
        soft_delta=0.35,
        lambda_noise=2.0,
        lambda_consistency=0.25,
        lambda_diff=0.5,
        lambda_uncert=0.5,
        lambda_tube=0.05,
        phi_threshold_hard=0.0,
        phi_threshold_soft=0.0,
        soft_tau=0.01,
        log_interval=25,
        diag_dir=str(diag_dir),
        name=optimizer_name,
    )
    if optimizer_name == "muon_ns_adamw":
        return HamGNNMuonFDTOptimizer(model.parameters(), mode="muon_ns", **common)
    if optimizer_name == "rt_v43_stream_adamw":
        return HamGNNMuonFDTOptimizer(model.parameters(), mode="rt_v43_stream", **common)
    if optimizer_name == "rt_v43_ns_adamw":
        return HamGNNMuonFDTOptimizer(model.parameters(), mode="rt_v43_ns", **common)
    raise ValueError(f"Unknown optimizer {optimizer_name}")


def summarize_diag(diag_path: Path) -> Dict[str, float]:
    if not diag_path.exists():
        return {}
    rows = []
    with diag_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    out: Dict[str, float] = {}
    for col in ["beta", "accept", "phi"]:
        vals = [float(r[col]) for r in rows if col in r]
        if vals:
            out[f"diag_mean_{col}"] = float(sum(vals) / len(vals))
    return out


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    keys = sorted({k for row in rows for k in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def group_rows(rows: List[Dict[str, object]], keys: List[str]) -> Dict[Tuple[object, ...], List[Dict[str, object]]]:
    out: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        out.setdefault(key, []).append(row)
    return out


def stats(vals: List[float]) -> Tuple[float, float, float, float]:
    mean_v = float(sum(vals) / len(vals))
    std_v = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
    return mean_v, std_v, float(min(vals)), float(max(vals))


def run_one(cfg: RunConfig, outdir: Path) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    run_dir = outdir / "per_run" / f"{cfg.optimizer}__lr_{cfg.lr:g}__seed_{cfg.seed}"
    diag_dir = run_dir / "diag"
    diag_dir.mkdir(parents=True, exist_ok=True)
    model = PoissonPINN(cfg.width, cfg.depth).to(device)
    optimizer = build_optimizer(model, cfg.optimizer, cfg.lr, diag_dir, cfg.lr_switch)
    batch = make_samples(cfg, device)
    curve_rows: List[Dict[str, object]] = []
    best_loss = float("inf")
    best_rel_l2 = float("inf")

    for step in range(1, cfg.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, pieces = full_loss(model, batch, cfg.k_freq)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == cfg.steps:
            eval_metrics = evaluate(model, cfg.k_freq, device)
            total_loss = float(loss.detach().cpu().item())
            best_loss = min(best_loss, total_loss)
            best_rel_l2 = min(best_rel_l2, eval_metrics["grid_rel_l2"])
            curve_rows.append(
                {
                    "optimizer": cfg.optimizer,
                    "seed": cfg.seed,
                    "lr": cfg.lr,
                    "step": step,
                    "train_loss": total_loss,
                    "best_train_loss_so_far": best_loss,
                    "grid_mse": eval_metrics["grid_mse"],
                    "grid_rel_l2": eval_metrics["grid_rel_l2"],
                    **pieces,
                }
            )

    final_eval = evaluate(model, cfg.k_freq, device)
    final_loss, pieces = full_loss(model, batch, cfg.k_freq)
    record = {
        **asdict(cfg),
        "final_train_loss": float(final_loss.detach().cpu().item()),
        "best_train_loss": best_loss,
        "best_grid_rel_l2": best_rel_l2,
        **final_eval,
        **pieces,
        **summarize_diag(diag_dir / f"{cfg.optimizer}_optimizer_diag.jsonl"),
    }
    return record, curve_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="High-frequency Poisson PINN benchmark.")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--width", type=int, default=96)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--n-interior", type=int, default=2048)
    p.add_argument("--k-freq", type=int, default=7)
    p.add_argument("--lrs", type=str, default="0.001,0.002")
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--optimizers", type=str, default="adamw,muon_ns_adamw,rt_v43_stream_adamw,rt_v43_ns_adamw")
    p.add_argument("--lr-switch", type=float, default=0.0015)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    lrs = [float(x.strip()) for x in args.lrs.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    optimizers = [x.strip() for x in args.optimizers.split(",") if x.strip()]

    records: List[Dict[str, object]] = []
    curves: List[Dict[str, object]] = []
    for optimizer_name in optimizers:
        for lr in lrs:
            for seed in seeds:
                cfg = RunConfig(
                    optimizer=optimizer_name,
                    seed=seed,
                    lr=lr,
                    steps=args.steps,
                    width=args.width,
                    depth=args.depth,
                    n_interior=args.n_interior,
                    device=args.device,
                    lr_switch=args.lr_switch,
                    k_freq=args.k_freq,
                )
                record, curve_rows = run_one(cfg, outdir)
                records.append(record)
                curves.extend(curve_rows)

    records = sorted(records, key=lambda r: (str(r["optimizer"]), float(r["lr"]), int(r["seed"])))
    curves = sorted(curves, key=lambda r: (str(r["optimizer"]), float(r["lr"]), int(r["seed"]), int(r["step"])))
    write_csv(outdir / "records.csv", records)
    write_csv(outdir / "curves.csv", curves)

    summary_rows: List[Dict[str, object]] = []
    metrics = ["final_train_loss", "best_train_loss", "grid_mse", "grid_rel_l2"]
    for (optimizer, lr), group in sorted(group_rows(records, ["optimizer", "lr"]).items()):
        row: Dict[str, object] = {"optimizer": optimizer, "lr": lr}
        for metric in metrics:
            vals = [float(g[metric]) for g in group]
            mean_v, std_v, min_v, max_v = stats(vals)
            row[f"{metric}_mean"] = mean_v
            row[f"{metric}_std"] = std_v
            row[f"{metric}_min"] = min_v
            row[f"{metric}_max"] = max_v
        summary_rows.append(row)
    write_csv(outdir / "summary_by_optimizer_lr.csv", summary_rows)

    ranking = sorted(records, key=lambda r: (float(r["grid_rel_l2"]), float(r["grid_mse"]), float(r["final_train_loss"])))
    write_csv(outdir / "ranking_by_rel_l2.csv", ranking)

    best_by_optimizer: List[Dict[str, object]] = []
    for optimizer, group in sorted(group_rows(records, ["optimizer"]).items()):
        best = sorted(group, key=lambda r: (float(r["grid_rel_l2"]), float(r["grid_mse"]), float(r["final_train_loss"])))[0]
        best_by_optimizer.append(best)
    write_csv(outdir / "best_by_optimizer.csv", best_by_optimizer)

    with (outdir / "run_args.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "script": str(Path(__file__).resolve()),
                "repaired_v43_source": str(HAMGNN_OPT_PATH.resolve()),
                "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            },
            fh,
            indent=2,
            sort_keys=True,
        )

    lines = [
        "# poisson_pinn benchmark",
        "",
        "This benchmark uses a high-frequency Poisson target to make the function-learning task harder.",
        "",
        f"- k_freq: {args.k_freq}",
        f"- depth: {args.depth}",
        f"- width: {args.width}",
        "",
        "## Acceptance rule",
        "",
        "This benchmark is only considered a good benchmark if the original `muon_ns_adamw` beats `adamw`.",
        "",
        "## Best runs by optimizer",
        "",
    ]
    for row in best_by_optimizer:
        lines.append(
            f"- `{row['optimizer']}` at `lr={row['lr']}` seed `{int(row['seed'])}`: "
            f"`grid_rel_l2={float(row['grid_rel_l2']):.6g}`, "
            f"`grid_mse={float(row['grid_mse']):.6g}`, "
            f"`final_train_loss={float(row['final_train_loss']):.6g}`"
        )
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
