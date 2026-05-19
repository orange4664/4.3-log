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
    n_residual: int
    n_initial: int
    n_boundary: int
    nu: float
    device: str
    lr_switch: float


class HeatPINN(nn.Module):
    def __init__(self, width: int, depth: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = 2
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.Tanh())
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t], dim=1))


def exact_solution(x: torch.Tensor, t: torch.Tensor, nu: float) -> torch.Tensor:
    return torch.exp(-(math.pi ** 2) * nu * t) * torch.sin(math.pi * x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_samples(cfg: RunConfig, device: torch.device) -> Dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(cfg.seed)
    x_r = torch.rand(cfg.n_residual, 1, generator=g)
    t_r = torch.rand(cfg.n_residual, 1, generator=g)
    x_i = torch.rand(cfg.n_initial, 1, generator=g)
    t_i = torch.zeros(cfg.n_initial, 1)
    t_b = torch.rand(cfg.n_boundary, 1, generator=g)
    x_b0 = torch.zeros(cfg.n_boundary, 1)
    x_b1 = torch.ones(cfg.n_boundary, 1)
    return {
        "x_r": x_r.to(device),
        "t_r": t_r.to(device),
        "x_i": x_i.to(device),
        "t_i": t_i.to(device),
        "x_b0": x_b0.to(device),
        "x_b1": x_b1.to(device),
        "t_b": t_b.to(device),
    }


def residual_loss(model: HeatPINN, x: torch.Tensor, t: torch.Tensor, nu: float) -> torch.Tensor:
    x_req = x.detach().clone().requires_grad_(True)
    t_req = t.detach().clone().requires_grad_(True)
    u = model(x_req, t_req)
    u_x, u_t = torch.autograd.grad(u, (x_req, t_req), grad_outputs=torch.ones_like(u), create_graph=True)
    u_xx = torch.autograd.grad(u_x, x_req, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return (u_t - nu * u_xx).pow(2).mean()


def full_loss(model: HeatPINN, batch: Dict[str, torch.Tensor], nu: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    pde = residual_loss(model, batch["x_r"], batch["t_r"], nu)
    u_i = model(batch["x_i"], batch["t_i"])
    u_i_ref = exact_solution(batch["x_i"], batch["t_i"], nu)
    ic = (u_i - u_i_ref).pow(2).mean()
    u_b0 = model(batch["x_b0"], batch["t_b"])
    u_b1 = model(batch["x_b1"], batch["t_b"])
    bc = u_b0.pow(2).mean() + u_b1.pow(2).mean()
    total = pde + ic + bc
    return total, {
        "pde_loss": float(pde.detach().cpu().item()),
        "ic_loss": float(ic.detach().cpu().item()),
        "bc_loss": float(bc.detach().cpu().item()),
    }


def evaluate(model: HeatPINN, nu: float, device: torch.device, grid_n: int = 81) -> Dict[str, float]:
    with torch.no_grad():
        xs = torch.linspace(0.0, 1.0, grid_n, device=device)
        ts = torch.linspace(0.0, 1.0, grid_n, device=device)
        xx, tt = torch.meshgrid(xs, ts, indexing="ij")
        pred = model(xx.reshape(-1, 1), tt.reshape(-1, 1)).reshape(grid_n, grid_n)
        ref = exact_solution(xx, tt, nu)
        mse = (pred - ref).pow(2).mean()
        rel_l2 = torch.linalg.norm(pred - ref) / torch.linalg.norm(ref).clamp_min(1e-12)
        max_abs = (pred - ref).abs().max()
    return {
        "grid_mse": float(mse.cpu().item()),
        "grid_rel_l2": float(rel_l2.cpu().item()),
        "grid_max_abs": float(max_abs.cpu().item()),
    }


def build_optimizer(
    model: HeatPINN,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    diag_dir: Path,
    lr_switch: float,
):
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    common = dict(
        lr=lr,
        weight_decay=weight_decay,
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
    if not rows:
        return {}
    out = {}
    for col in [
        "beta",
        "beta_candidate",
        "soft_shrink",
        "accept",
        "phi",
        "raw_cv_gain",
        "diffusion_cost",
        "uncertainty_cost",
        "tube_penalty",
        "mode_uses_muon_source",
    ]:
        vals = [float(r[col]) for r in rows if col in r]
        if vals:
            out[f"diag_mean_{col}"] = float(sum(vals) / len(vals))
    return out


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = set()
        for row in rows:
            keys.update(row.keys())
        fieldnames = sorted(keys)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def group_records(rows: List[Dict[str, object]], keys: List[str]) -> Dict[Tuple[object, ...], List[Dict[str, object]]]:
    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        groups.setdefault(key, []).append(row)
    return groups


def mean_std_min_max(values: List[float]) -> Tuple[float, float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean_v = float(sum(values) / len(values))
    std_v = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    return mean_v, std_v, float(min(values)), float(max(values))


def run_one(cfg: RunConfig, outdir: Path) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    run_dir = outdir / "per_run" / f"{cfg.optimizer}__lr_{cfg.lr:g}__seed_{cfg.seed}"
    diag_dir = run_dir / "diag"
    diag_dir.mkdir(parents=True, exist_ok=True)

    model = HeatPINN(cfg.width, cfg.depth).to(device)
    optimizer = build_optimizer(model, cfg.optimizer, cfg.lr, 0.0, diag_dir, cfg.lr_switch)
    batch = make_samples(cfg, device)

    curve_rows: List[Dict[str, float]] = []
    best_loss = float("inf")
    best_rel_l2 = float("inf")

    for step in range(1, cfg.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, pieces = full_loss(model, batch, cfg.nu)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 25 == 0 or step == cfg.steps:
            eval_metrics = evaluate(model, cfg.nu, device)
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
                    "grid_max_abs": eval_metrics["grid_max_abs"],
                    **pieces,
                }
            )

    final_eval = evaluate(model, cfg.nu, device)
    final_loss, pieces = full_loss(model, batch, cfg.nu)
    diag_summary = summarize_diag(diag_dir / f"{cfg.optimizer}_optimizer_diag.jsonl")
    record = {
        **asdict(cfg),
        "final_train_loss": float(final_loss.detach().cpu().item()),
        "best_train_loss": best_loss,
        "best_grid_rel_l2": best_rel_l2,
        **final_eval,
        **pieces,
        **diag_summary,
    }
    return record, curve_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real PDE/PINN benchmark for repaired v4.3 optimizers.")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--n-residual", type=int, default=512)
    parser.add_argument("--n-initial", type=int, default=128)
    parser.add_argument("--n-boundary", type=int, default=128)
    parser.add_argument("--nu", type=float, default=0.1)
    parser.add_argument("--lrs", type=str, default="0.001,0.002")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument(
        "--optimizers",
        type=str,
        default="adamw,muon_ns_adamw,rt_v43_stream_adamw,rt_v43_ns_adamw",
    )
    parser.add_argument("--lr-switch", type=float, default=0.0015)
    return parser.parse_args()


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
                    n_residual=args.n_residual,
                    n_initial=args.n_initial,
                    n_boundary=args.n_boundary,
                    nu=args.nu,
                    device=args.device,
                    lr_switch=args.lr_switch,
                )
                record, curve_rows = run_one(cfg, outdir)
                records.append(record)
                curves.extend(curve_rows)

    records_sorted = sorted(records, key=lambda r: (str(r["optimizer"]), float(r["lr"]), int(r["seed"])))
    curves_sorted = sorted(curves, key=lambda r: (str(r["optimizer"]), float(r["lr"]), int(r["seed"]), int(r["step"])))
    write_csv(outdir / "records.csv", records_sorted)
    write_csv(outdir / "curves.csv", curves_sorted)

    summary_rows: List[Dict[str, object]] = []
    metrics = [
        "final_train_loss",
        "best_train_loss",
        "grid_mse",
        "grid_rel_l2",
        "grid_max_abs",
        "diag_mean_beta",
        "diag_mean_accept",
        "diag_mean_phi",
    ]
    for (optimizer, lr), group in sorted(group_records(records_sorted, ["optimizer", "lr"]).items()):
        row: Dict[str, object] = {"optimizer": optimizer, "lr": lr}
        for metric in metrics:
            vals = [float(g[metric]) for g in group if metric in g and g[metric] is not None]
            mean_v, std_v, min_v, max_v = mean_std_min_max(vals)
            row[f"{metric}_mean"] = mean_v
            row[f"{metric}_std"] = std_v
            row[f"{metric}_min"] = min_v
            row[f"{metric}_max"] = max_v
        summary_rows.append(row)
    write_csv(outdir / "summary_by_optimizer_lr.csv", summary_rows)

    ranking = sorted(records_sorted, key=lambda r: (float(r["grid_rel_l2"]), float(r["grid_mse"]), float(r["final_train_loss"])))
    write_csv(outdir / "ranking_by_rel_l2.csv", ranking)

    best_by_optimizer: List[Dict[str, object]] = []
    for optimizer, group in sorted(group_records(records_sorted, ["optimizer"]).items()):
        best = sorted(group, key=lambda r: (float(r["grid_rel_l2"]), float(r["grid_mse"]), float(r["final_train_loss"])))[0]
        best_by_optimizer.append(best)
    write_csv(outdir / "best_by_optimizer.csv", best_by_optimizer)

    arg_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    metadata = {
        "script": str(Path(__file__).resolve()),
        "repaired_v43_source": str(HAMGNN_OPT_PATH.resolve()),
        "args": arg_dict,
    }
    with (outdir / "run_args.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)

    summary_lines = [
        "# heat1d_pinn benchmark",
        "",
        "This is a real PDE/PINN benchmark, separate from the earlier selector toy benchmark.",
        "",
        "## PDE",
        "",
        "- Equation: `u_t = nu * u_xx` on `x in [0,1]`, `t in [0,1]`",
        "- Initial condition: `u(x,0) = sin(pi x)`",
        "- Boundary condition: `u(0,t) = u(1,t) = 0`",
        "- Exact solution: `exp(-nu*pi^2*t) * sin(pi*x)`",
        "",
        "## Optimizers",
        "",
        "- `adamw`: pure AdamW baseline",
        "- `muon_ns_adamw`: repaired Muon-NS for matrix params, AdamW fallback for non-matrix params",
        "- `rt_v43_stream_adamw`: repaired v4.3 stream path for matrix params, AdamW fallback elsewhere",
        "- `rt_v43_ns_adamw`: repaired v4.3 Muon-source path for matrix params, AdamW fallback elsewhere",
        "",
        "## Key files",
        "",
        "- `records.csv`",
        "- `summary_by_optimizer_lr.csv`",
        "- `ranking_by_rel_l2.csv`",
        "- `best_by_optimizer.csv`",
        "- `curves.csv`",
        "",
        "## Best runs by optimizer",
        "",
    ]
    for row in best_by_optimizer:
        summary_lines.append(
            f"- `{row['optimizer']}` at `lr={row['lr']}` seed `{int(row['seed'])}`: "
            f"`grid_rel_l2={float(row['grid_rel_l2']):.6g}`, `grid_mse={float(row['grid_mse']):.6g}`, "
            f"`final_train_loss={float(row['final_train_loss']):.6g}`"
        )
    summary_lines.append("")
    summary_lines.append("This benchmark uses the repaired torch implementation from `test1-hamgnn-si/code/hamgnn_rt/optimizers.py`.")
    (outdir / "README.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
