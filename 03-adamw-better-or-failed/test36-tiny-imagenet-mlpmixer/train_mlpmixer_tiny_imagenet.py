from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from models import TinyMLPMixer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ensure_val_layout(root: Path) -> None:
    val_dir = root / "val"
    images_dir = val_dir / "images"
    anno_path = val_dir / "val_annotations.txt"
    if not images_dir.exists() or not anno_path.exists():
        return

    mapping = {}
    for line in anno_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            mapping[parts[0]] = parts[1]

    for image_name, cls in mapping.items():
        src = images_dir / image_name
        dst_dir = val_dir / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / image_name
        if dst.exists():
            continue
        if src.exists():
            shutil.move(str(src), str(dst))

    try:
        if images_dir.exists() and not any(images_dir.iterdir()):
            images_dir.rmdir()
    except OSError:
        pass


def maybe_unpack_zip(data_dir: Path) -> Path:
    root = data_dir / "tiny-imagenet-200"
    if root.exists():
        ensure_val_layout(root)
        return root

    archive = data_dir / "tiny-imagenet-200.zip"
    if archive.exists():
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(data_dir)
        ensure_val_layout(root)
        return root

    raise FileNotFoundError(
        f"Tiny-ImageNet data not found under {data_dir}. "
        "Expected tiny-imagenet-200/ or tiny-imagenet-200.zip."
    )


def get_tiny_imagenet(data_dir: str):
    import torchvision
    import torchvision.transforms as T

    data_root = maybe_unpack_zip(Path(data_dir))

    train_tf = T.Compose(
        [
            T.RandomCrop(64, padding=8),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
        ]
    )
    test_tf = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
        ]
    )

    train = torchvision.datasets.ImageFolder(root=str(data_root / "train"), transform=train_tf)
    val = torchvision.datasets.ImageFolder(root=str(data_root / "val"), transform=test_tf)
    return train, val


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == y).float().mean().item())


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_n = 0
    ce = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            total_loss += float(ce(logits, y).item())
            total_acc += float((logits.argmax(dim=-1) == y).sum().item())
            total_n += int(y.numel())
    model.train()
    return total_loss / max(total_n, 1), total_acc / max(total_n, 1)


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
    os.environ["HAMGNN_RT_METRIC_ALPHA_ROW"] = str(args.metric_alpha_row)
    os.environ["HAMGNN_RT_METRIC_ALPHA_COL"] = str(args.metric_alpha_col)
    os.environ["HAMGNN_RT_METRIC_LAMBDA_NOISE"] = str(args.metric_lambda_noise)
    os.environ["HAMGNN_RT_METRIC_TMIN"] = str(args.metric_tmin)
    os.environ["HAMGNN_RT_METRIC_TMAX"] = str(args.metric_tmax)
    os.environ["HAMGNN_RT_METRIC_EMA_BETA"] = str(args.metric_ema_beta)
    os.environ["HAMGNN_RT_HT_POWER"] = str(args.ht_power)
    os.environ["HAMGNN_RT_MCLIP_DELTA"] = str(args.mclip_delta)
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


def train_one(args: argparse.Namespace, seed: int, lr: float, method: str, outdir: Path) -> Dict[str, float]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_set, test_set = get_tiny_imagenet(args.data_dir)
    model = TinyMLPMixer(
        image_size=64,
        patch_size=args.patch_size,
        num_classes=200,
        embed_dim=args.embed_dim,
        depth=args.depth,
        token_dim=args.token_dim,
        channel_dim=args.channel_dim,
        dropout=args.dropout,
    ).to(device)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    method_dir = outdir / method / f"lr_{lr:g}" / f"seed_{seed}"
    diag_dir = method_dir / "diag"
    method_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)

    optimizer = make_optimizer(model, args, method, lr, diag_dir)
    ce = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    records: List[Dict[str, float]] = []
    start = time.time()
    step = 0
    auc_train_loss = 0.0
    best_test_acc = 0.0
    final_train_loss = float("nan")
    final_test_loss = float("nan")
    final_test_acc = float("nan")

    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            if args.max_steps and step >= args.max_steps:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            optimizer.step()

            train_loss = float(loss.detach().item())
            train_acc = accuracy(logits.detach(), y)
            auc_train_loss += train_loss
            final_train_loss = train_loss

            if step % args.log_every == 0 or (args.max_steps and step == args.max_steps - 1):
                test_loss, test_acc = evaluate(model, test_loader, device, max_batches=args.eval_batches)
                best_test_acc = max(best_test_acc, test_acc)
                final_test_loss, final_test_acc = test_loss, test_acc
                records.append(
                    {
                        "step": step,
                        "epoch": epoch,
                        "seed": seed,
                        "lr": lr,
                        "method": method,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "test_loss": test_loss,
                        "test_acc": test_acc,
                        "best_test_acc": best_test_acc,
                        "time_sec": time.time() - start,
                    }
                )
            step += 1

        scheduler.step()
        if args.max_steps and step >= args.max_steps:
            break

    if records:
        with (method_dir / "records.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    summary = {
        "method": method,
        "seed": seed,
        "lr": lr,
        "steps": step,
        "auc_train_loss": auc_train_loss / max(step, 1),
        "final_train_loss": final_train_loss,
        "final_test_loss": final_test_loss,
        "final_test_acc": final_test_acc,
        "best_test_acc": best_test_acc,
        "time_sec": time.time() - start,
        "device": str(device),
        "embed_dim": args.embed_dim,
        "depth": args.depth,
        "token_dim": args.token_dim,
        "channel_dim": args.channel_dim,
    }
    with (method_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--methods", default="adamw,muon_ns,rt_v43_stream,rt_v43_ns,rt_v6_fdt_metric")
    ap.add_argument("--lrs", default="0.0005,0.001")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--eval-batch-size", type=int, default=192)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--momentum", type=float, default=0.95)
    ap.add_argument("--ns-steps", type=int, default=5)
    ap.add_argument("--stream-k", type=int, default=0)
    ap.add_argument("--power-iters", type=int, default=1)
    ap.add_argument("--rt-period", type=int, default=1)
    ap.add_argument("--lr-switch", type=float, default=0.01)
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
    ap.add_argument("--metric-alpha-row", type=float, default=0.25)
    ap.add_argument("--metric-alpha-col", type=float, default=0.25)
    ap.add_argument("--metric-lambda-noise", type=float, default=1.0)
    ap.add_argument("--metric-tmin", type=float, default=0.5)
    ap.add_argument("--metric-tmax", type=float, default=2.0)
    ap.add_argument("--metric-ema-beta", type=float, default=0.9)
    ap.add_argument("--ht-power", type=float, default=0.5)
    ap.add_argument("--mclip-delta", type=float, default=0.5)
    ap.add_argument("--beta-grid", default="0.0,0.05,0.1,0.2,0.35,0.5")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--patch-size", type=int, default=4)
    ap.add_argument("--embed-dim", type=int, default=320)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--token-dim", type=int, default=160)
    ap.add_argument("--channel-dim", type=int, default=640)
    ap.add_argument("--dropout", type=float, default=0.0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    lrs = [float(x.strip()) for x in args.lrs.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    rows: List[Dict[str, float]] = []
    for method in methods:
        for lr in lrs:
            for seed in seeds:
                rows.append(train_one(args, seed, lr, method, outdir))

    with (outdir / "all_summaries.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
