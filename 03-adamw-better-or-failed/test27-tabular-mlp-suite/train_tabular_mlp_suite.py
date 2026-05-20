from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import TabularMLP


DATASET_LOADERS: Dict[str, Callable[[], object]] = {
    "breast_cancer": datasets.load_breast_cancer,
    "wine": datasets.load_wine,
    "digits": datasets.load_digits,
}


@dataclass
class DatasetBundle:
    name: str
    in_dim: int
    num_classes: int
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def to_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    return DataLoader(
        TensorDataset(x_t, y_t),
        batch_size=min(batch_size, len(x_t)),
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )


def load_dataset(name: str, seed: int, batch_size: int, eval_batch_size: int, device: torch.device) -> DatasetBundle:
    if name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset {name}")
    raw = DATASET_LOADERS[name]()
    x = raw.data.astype(np.float32)
    y = raw.target.astype(np.int64)

    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval,
        y_trainval,
        test_size=0.25,
        random_state=seed + 17,
        stratify=y_trainval,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    return DatasetBundle(
        name=name,
        in_dim=int(x.shape[1]),
        num_classes=int(len(np.unique(y))),
        train_loader=to_loader(x_train, y_train, batch_size=batch_size, shuffle=True, device=device),
        val_loader=to_loader(x_val, y_val, batch_size=eval_batch_size, shuffle=False, device=device),
        test_loader=to_loader(x_test, y_test, batch_size=eval_batch_size, shuffle=False, device=device),
    )


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == y).float().mean().item())


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0.0
    total_n = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            total_loss += float(ce(logits, y).item())
            total_correct += float((logits.argmax(dim=-1) == y).sum().item())
            total_n += int(y.numel())
    model.train()
    return total_loss / max(total_n, 1), total_correct / max(total_n, 1)


def set_hamgnn_env(args: argparse.Namespace, optimizer_name: str, diag_dir: Path) -> None:
    os.environ["HAMGNN_RT_OPTIMIZER"] = optimizer_name
    os.environ["HAMGNN_RT_WEIGHT_DECAY"] = str(args.weight_decay)
    os.environ["HAMGNN_RT_MOMENTUM"] = str(args.momentum)
    os.environ["HAMGNN_MUON_NESTEROV"] = "true"
    os.environ["HAMGNN_MUON_NS_STEPS"] = str(args.ns_steps)
    os.environ["HAMGNN_RT_STREAM_K"] = str(args.stream_k)
    os.environ["HAMGNN_RT_POWER_ITERS"] = str(args.power_iters)
    os.environ["HAMGNN_RT_PERIOD"] = str(args.rt_period)
    os.environ["HAMGNN_RT_LR_SWITCH"] = str(args.lr_switch)
    os.environ["HAMGNN_RT_HARD_DELTA"] = str(args.hard_delta)
    os.environ["HAMGNN_RT_SOFT_DELTA"] = str(args.soft_delta)
    os.environ["HAMGNN_RT_LAMBDA_NOISE"] = str(args.lambda_noise)
    os.environ["HAMGNN_RT_LAMBDA_CONSISTENCY"] = str(args.lambda_consistency)
    os.environ["HAMGNN_RT_LAMBDA_DIFF"] = str(args.lambda_diff)
    os.environ["HAMGNN_RT_LAMBDA_UNCERT"] = str(args.lambda_uncert)
    os.environ["HAMGNN_RT_LAMBDA_TUBE"] = str(args.lambda_tube)
    os.environ["HAMGNN_RT_HARD_PHI_THRESHOLD"] = str(args.phi_threshold_hard)
    os.environ["HAMGNN_RT_SOFT_PHI_THRESHOLD"] = str(args.phi_threshold_soft)
    os.environ["HAMGNN_RT_SOFT_TAU"] = str(args.soft_tau)
    os.environ["HAMGNN_RT_LOG_INTERVAL"] = str(args.log_every)
    os.environ["HAMGNN_RT_DIAG_DIR"] = str(diag_dir)
    os.environ["HAMGNN_RT_RUN_NAME"] = optimizer_name
    os.environ["HAMGNN_RT_BETA_GRID"] = args.beta_grid


def make_optimizer(model: nn.Module, args: argparse.Namespace, method: str, lr: float, diag_dir: Path):
    if method == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    from hamgnn_rt.optimizers import build_optimizer_from_env

    set_hamgnn_env(args, method, diag_dir)
    return build_optimizer_from_env(
        list(model.named_parameters()),
        lr=lr,
        eps=1e-8,
        beta1=0.9,
        beta2=0.95,
        amsgrad=False,
    )


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def group_rows(rows: List[Dict[str, object]], keys: List[str]) -> Dict[Tuple[object, ...], List[Dict[str, object]]]:
    out: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        out.setdefault(key, []).append(row)
    return out


def mean_std(values: List[float]) -> Tuple[float, float]:
    mean = float(sum(values) / len(values))
    std = float(np.std(np.array(values, dtype=np.float64))) if len(values) > 1 else 0.0
    return mean, std


def train_one(
    args: argparse.Namespace,
    dataset_name: str,
    seed: int,
    lr: float,
    method: str,
    outdir: Path,
) -> Dict[str, object]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    bundle = load_dataset(
        dataset_name,
        seed=seed,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        device=device,
    )

    model = TabularMLP(
        in_dim=bundle.in_dim,
        num_classes=bundle.num_classes,
        width=args.width,
        depth=args.depth,
        dropout=args.dropout,
    ).to(device)

    run_dir = outdir / dataset_name / method / f"lr_{lr:g}" / f"seed_{seed}"
    diag_dir = run_dir / "diag"
    run_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)

    optimizer = make_optimizer(model, args, method, lr, diag_dir)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    ce = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    test_acc_at_best_val = -1.0
    best_val_epoch = -1
    final_train_loss = float("nan")
    final_val_loss = float("nan")
    final_val_acc = float("nan")
    final_test_loss = float("nan")
    final_test_acc = float("nan")
    auc_train_loss = 0.0
    epochs_without_improve = 0
    start = time.time()
    epoch_rows: List[Dict[str, object]] = []

    for epoch in range(args.epochs):
        model.train()
        running_train_loss = 0.0
        running_train_acc = 0.0
        batches = 0
        for x, y in bundle.train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            optimizer.step()
            running_train_loss += float(loss.detach().item())
            running_train_acc += accuracy(logits.detach(), y)
            batches += 1

        scheduler.step()

        final_train_loss = running_train_loss / max(batches, 1)
        auc_train_loss += final_train_loss
        val_loss, val_acc = evaluate(model, bundle.val_loader, device)
        test_loss, test_acc = evaluate(model, bundle.test_loader, device)
        final_val_loss, final_val_acc = val_loss, val_acc
        final_test_loss, final_test_acc = test_loss, test_acc

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc_at_best_val = test_acc
            best_val_epoch = epoch
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        epoch_rows.append(
            {
                "dataset": dataset_name,
                "method": method,
                "lr": lr,
                "seed": seed,
                "epoch": epoch,
                "train_loss": final_train_loss,
                "train_acc": running_train_acc / max(batches, 1),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "best_val_acc": best_val_acc,
                "test_acc_at_best_val": test_acc_at_best_val,
                "time_sec": time.time() - start,
            }
        )

        if args.early_stop_patience > 0 and epochs_without_improve >= args.early_stop_patience:
            break

    write_csv(run_dir / "records.csv", epoch_rows)

    summary = {
        "dataset": dataset_name,
        "method": method,
        "seed": seed,
        "lr": lr,
        "epochs_ran": len(epoch_rows),
        "auc_train_loss": auc_train_loss / max(len(epoch_rows), 1),
        "best_val_acc": best_val_acc,
        "best_val_epoch": best_val_epoch,
        "test_acc_at_best_val": test_acc_at_best_val,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "final_val_acc": final_val_acc,
        "final_test_loss": final_test_loss,
        "final_test_acc": final_test_acc,
        "time_sec": time.time() - start,
        "device": str(device),
        "width": args.width,
        "depth": args.depth,
        "dropout": args.dropout,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def build_aggregate_tables(outdir: Path, rows: List[Dict[str, object]]) -> None:
    write_csv(outdir / "all_summaries.csv", rows)

    by_dataset_method_lr: List[Dict[str, object]] = []
    grouped = group_rows(rows, ["dataset", "method", "lr"])
    for (dataset_name, method, lr), group in sorted(grouped.items()):
        best_val_acc_mean, best_val_acc_std = mean_std([float(r["best_val_acc"]) for r in group])
        best_test_mean, best_test_std = mean_std([float(r["test_acc_at_best_val"]) for r in group])
        final_test_mean, final_test_std = mean_std([float(r["final_test_acc"]) for r in group])
        epochs_mean, _ = mean_std([float(r["epochs_ran"]) for r in group])
        by_dataset_method_lr.append(
            {
                "dataset": dataset_name,
                "method": method,
                "lr": lr,
                "seed_count": len(group),
                "best_val_acc_mean": best_val_acc_mean,
                "best_val_acc_std": best_val_acc_std,
                "test_acc_at_best_val_mean": best_test_mean,
                "test_acc_at_best_val_std": best_test_std,
                "final_test_acc_mean": final_test_mean,
                "final_test_acc_std": final_test_std,
                "epochs_ran_mean": epochs_mean,
            }
        )
    write_csv(outdir / "dataset_method_lr_summary.csv", by_dataset_method_lr)

    best_by_dataset_optimizer: List[Dict[str, object]] = []
    grouped_best = group_rows(by_dataset_method_lr, ["dataset", "method"])
    for (_, _), group in sorted(grouped_best.items()):
        best_row = sorted(
            group,
            key=lambda r: (
                -float(r["best_val_acc_mean"]),
                -float(r["test_acc_at_best_val_mean"]),
                float(r["lr"]),
            ),
        )[0]
        best_by_dataset_optimizer.append(best_row)
    write_csv(outdir / "best_by_dataset_optimizer.csv", best_by_dataset_optimizer)

    overall_rows: List[Dict[str, object]] = []
    grouped_overall = group_rows(best_by_dataset_optimizer, ["method"])
    adamw_rows = {row["dataset"]: row for row in best_by_dataset_optimizer if row["method"] == "adamw"}
    for (method,), group in sorted(grouped_overall.items()):
        test_mean, test_std = mean_std([float(r["test_acc_at_best_val_mean"]) for r in group])
        val_mean, val_std = mean_std([float(r["best_val_acc_mean"]) for r in group])
        win_count_vs_adamw = 0
        compared = 0
        for row in group:
            ds = str(row["dataset"])
            if ds in adamw_rows and method != "adamw":
                compared += 1
                if float(row["test_acc_at_best_val_mean"]) > float(adamw_rows[ds]["test_acc_at_best_val_mean"]):
                    win_count_vs_adamw += 1
        overall_rows.append(
            {
                "method": method,
                "dataset_count": len(group),
                "mean_best_val_acc": val_mean,
                "std_best_val_acc": val_std,
                "mean_test_acc_at_best_val": test_mean,
                "std_test_acc_at_best_val": test_std,
                "win_count_vs_adamw": win_count_vs_adamw,
                "compared_dataset_count_vs_adamw": compared,
            }
        )
    write_csv(outdir / "overall_optimizer_summary.csv", overall_rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--datasets", default="breast_cancer,wine,digits")
    ap.add_argument("--methods", default="adamw,muon_ns,rt_v43_stream,rt_v43_ns")
    ap.add_argument("--lrs", default="0.0005,0.001,0.002")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--early-stop-patience", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.95)
    ap.add_argument("--ns-steps", type=int, default=5)
    ap.add_argument("--stream-k", type=int, default=0)
    ap.add_argument("--power-iters", type=int, default=1)
    ap.add_argument("--rt-period", type=int, default=1)
    ap.add_argument("--lr-switch", type=float, default=0.0015)
    ap.add_argument("--hard-delta", type=float, default=0.35)
    ap.add_argument("--soft-delta", type=float, default=0.35)
    ap.add_argument("--lambda-noise", type=float, default=2.0)
    ap.add_argument("--lambda-consistency", type=float, default=0.25)
    ap.add_argument("--lambda-diff", type=float, default=0.5)
    ap.add_argument("--lambda-uncert", type=float, default=0.5)
    ap.add_argument("--lambda-tube", type=float, default=0.05)
    ap.add_argument("--phi-threshold-hard", type=float, default=0.0)
    ap.add_argument("--phi-threshold-soft", type=float, default=0.0)
    ap.add_argument("--soft-tau", type=float, default=0.01)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--beta-grid", default="0.0,0.05,0.1,0.2,0.35,0.5")
    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dataset_names = [x.strip() for x in args.datasets.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    lrs = [float(x.strip()) for x in args.lrs.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    rows: List[Dict[str, object]] = []
    for dataset_name in dataset_names:
        for method in methods:
            for lr in lrs:
                for seed in seeds:
                    rows.append(train_one(args, dataset_name, seed, lr, method, outdir))

    build_aggregate_tables(outdir, rows)
    with (outdir / "run_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
