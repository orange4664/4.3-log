from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List

import torch
from torch.utils.data import DataLoader, IterableDataset


@dataclass
class TrainConfig:
    tokenizer_name_or_path: str = ""
    seq_len: int = 256
    train_batch_size: int = 8
    eval_batch_size: int = 8
    grad_accum_steps: int = 4
    max_steps: int = 600
    eval_interval: int = 100
    eval_batches: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_steps: int = 50
    log_interval: int = 10
    seed: int = 666
    num_workers: int = 0
    optimizer: str = "adamw"
    out_dir: str = "./out"
    diag_dir: str = "./diag"
    train_text_file: str = ""
    val_text_file: str = ""
    autocast_dtype: str = "none"
    vocab_size: int = 50257
    n_embd: int = 512
    n_layer: int = 8
    n_head: int = 8


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TextLineDataset(IterableDataset):
    def __init__(self, path: str, repeat: bool) -> None:
        super().__init__()
        self.path = path
        self.repeat = repeat

    def __iter__(self) -> Iterator[Dict[str, str]]:
        while True:
            emitted = False
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    text = line.strip()
                    if text:
                        emitted = True
                        yield {"text": text}
            if not self.repeat:
                break
            if not emitted:
                raise RuntimeError(f"No non-empty text lines found in {self.path}")


class TokenChunkDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, seq_len: int, text_key: str = "text") -> None:
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.text_key = text_key
        self.eos_id = tokenizer.eos_token_id
        if self.eos_id is None:
            raise ValueError("Tokenizer must provide eos_token_id.")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        buffer: List[int] = []
        for row in self.dataset:
            text = row.get(self.text_key, "")
            if not text:
                continue
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            if not ids:
                continue
            buffer.extend(ids)
            buffer.append(self.eos_id)
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                del buffer[: self.seq_len]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": x, "labels": y}


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch], dim=0),
        "labels": torch.stack([x["labels"] for x in batch], dim=0),
    }


def build_eval_loader(dataset, tokenizer, seq_len: int, batch_size: int, eval_batches: int) -> DataLoader:
    iterable = TokenChunkDataset(dataset, tokenizer, seq_len)
    items: List[Dict[str, torch.Tensor]] = []
    target = batch_size * eval_batches
    for sample in iterable:
        items.append(sample)
        if len(items) >= target:
            break
    if not items:
        raise RuntimeError("No evaluation samples were produced.")
    return DataLoader(items, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)


def safe_exp(value: float) -> float:
    return float(math.exp(min(float(value), 80.0)))


def autocast_context(dtype_name: str):
    dtype_name = str(dtype_name).lower()
    if dtype_name in {"none", "off", "false", "0"}:
        return torch.autocast(device_type="cuda", enabled=False)
    if dtype_name in {"bf16", "bfloat16"}:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if dtype_name in {"fp16", "float16", "half"}:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"Unsupported autocast dtype: {dtype_name}")


def set_optimizer_env(cfg: TrainConfig) -> None:
    os.environ["HAMGNN_RT_OPTIMIZER"] = cfg.optimizer
    os.environ["HAMGNN_RT_WEIGHT_DECAY"] = str(cfg.weight_decay)
    os.environ["HAMGNN_RT_MOMENTUM"] = "0.95"
    os.environ["HAMGNN_MUON_NESTEROV"] = "true"
    os.environ["HAMGNN_MUON_NS_STEPS"] = "5"
    os.environ["HAMGNN_RT_STREAM_K"] = "0"
    os.environ["HAMGNN_RT_POWER_ITERS"] = "1"
    os.environ["HAMGNN_RT_PERIOD"] = "1"
    os.environ["HAMGNN_RT_LR_SWITCH"] = "0.05"
    os.environ["HAMGNN_RT_HARD_DELTA"] = "0.35"
    os.environ["HAMGNN_RT_SOFT_DELTA"] = "0.35"
    os.environ["HAMGNN_RT_LAMBDA_NOISE"] = "2.0"
    os.environ["HAMGNN_RT_LAMBDA_CONSISTENCY"] = "0.25"
    os.environ["HAMGNN_RT_LAMBDA_DIFF"] = "0.5"
    os.environ["HAMGNN_RT_LAMBDA_UNCERT"] = "0.5"
    os.environ["HAMGNN_RT_LAMBDA_TUBE"] = "0.05"
    os.environ["HAMGNN_RT_HARD_PHI_THRESHOLD"] = "0.0"
    os.environ["HAMGNN_RT_SOFT_PHI_THRESHOLD"] = "0.0"
    os.environ["HAMGNN_RT_METRIC_ALPHA_ROW"] = "0.25"
    os.environ["HAMGNN_RT_METRIC_ALPHA_COL"] = "0.25"
    os.environ["HAMGNN_RT_METRIC_TMIN"] = "0.5"
    os.environ["HAMGNN_RT_METRIC_TMAX"] = "2.0"
    os.environ["HAMGNN_RT_METRIC_LAMBDA_NOISE"] = "1.0"
    os.environ["HAMGNN_RT_LOG_INTERVAL"] = str(cfg.log_interval)
    os.environ["HAMGNN_RT_DIAG_DIR"] = cfg.diag_dir
    os.environ["HAMGNN_RT_RUN_NAME"] = cfg.optimizer


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--diag-dir", required=True)
    parser.add_argument("--train-text-file", required=True)
    parser.add_argument("--val-text-file", required=True)
    parser.add_argument("--tokenizer-name-or-path", required=True)
    parser.add_argument("--seq-len", type=int, default=TrainConfig.seq_len)
    parser.add_argument("--train-batch-size", type=int, default=TrainConfig.train_batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=TrainConfig.eval_batch_size)
    parser.add_argument("--grad-accum-steps", type=int, default=TrainConfig.grad_accum_steps)
    parser.add_argument("--max-steps", type=int, default=TrainConfig.max_steps)
    parser.add_argument("--eval-interval", type=int, default=TrainConfig.eval_interval)
    parser.add_argument("--eval-batches", type=int, default=TrainConfig.eval_batches)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--warmup-steps", type=int, default=TrainConfig.warmup_steps)
    parser.add_argument("--log-interval", type=int, default=TrainConfig.log_interval)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--autocast-dtype", default=TrainConfig.autocast_dtype)
    parser.add_argument("--vocab-size", type=int, default=TrainConfig.vocab_size)
    parser.add_argument("--n-embd", type=int, default=TrainConfig.n_embd)
    parser.add_argument("--n-layer", type=int, default=TrainConfig.n_layer)
    parser.add_argument("--n-head", type=int, default=TrainConfig.n_head)
    args = parser.parse_args()

    cfg = TrainConfig(
        optimizer=args.optimizer,
        out_dir=args.out_dir,
        diag_dir=args.diag_dir,
        train_text_file=args.train_text_file,
        val_text_file=args.val_text_file,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        seq_len=args.seq_len,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        log_interval=args.log_interval,
        seed=args.seed,
        num_workers=args.num_workers,
        autocast_dtype=args.autocast_dtype,
        vocab_size=args.vocab_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
    )

    out_dir = Path(cfg.out_dir)
    diag_dir = Path(cfg.diag_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)
    set_optimizer_env(cfg)

    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, get_cosine_schedule_with_warmup
    from hamgnn_rt.optimizers import build_optimizer_from_env

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer_name_or_path,
        use_fast=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_cfg = GPT2Config(
        vocab_size=max(cfg.vocab_size, tokenizer.vocab_size),
        n_positions=cfg.seq_len,
        n_ctx=cfg.seq_len,
        n_embd=cfg.n_embd,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        use_cache=False,
    )
    model = GPT2LMHeadModel(model_cfg)
    model.float()
    model.to("cuda")

    train_split = TextLineDataset(cfg.train_text_file, repeat=True)
    val_split = TextLineDataset(cfg.val_text_file, repeat=False)

    train_loader = DataLoader(
        TokenChunkDataset(train_split, tokenizer, cfg.seq_len),
        batch_size=cfg.train_batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate_batch,
    )
    eval_loader = build_eval_loader(val_split, tokenizer, cfg.seq_len, cfg.eval_batch_size, cfg.eval_batches)

    optimizer = build_optimizer_from_env(
        model.named_parameters(),
        lr=cfg.learning_rate,
        eps=1e-8,
        beta1=0.9,
        beta2=0.95,
        amsgrad=False,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=cfg.max_steps,
    )

    save_json(diag_dir / "run_config.json", asdict(cfg))
    metrics_path = diag_dir / "train_metrics.jsonl"

    step = 0
    best_val = float("inf")
    best_train = float("inf")
    train_loss_sum = 0.0
    val_points: List[tuple[int, float]] = []
    start = time.time()
    tokens_per_step = cfg.seq_len * cfg.train_batch_size * cfg.grad_accum_steps
    train_iter = iter(train_loader)
    model.train()

    def run_eval() -> float:
        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
                with autocast_context(cfg.autocast_dtype):
                    out = model(**batch)
                total += float(out.loss.item())
                count += 1
        model.train()
        return total / max(count, 1)

    while step < cfg.max_steps:
        step_start = time.time()
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(cfg.grad_accum_steps):
            batch = next(train_iter)
            batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
            with autocast_context(cfg.autocast_dtype):
                out = model(**batch)
                loss = out.loss / cfg.grad_accum_steps
            loss.backward()
            accum_loss += float(loss.item()) * cfg.grad_accum_steps
        optimizer.step()
        scheduler.step()
        step += 1
        elapsed_s = time.time() - start
        step_time_s = time.time() - step_start
        train_loss_sum += accum_loss
        best_train = min(best_train, accum_loss)
        tokens_seen = step * tokens_per_step

        rec = {
            "step": step,
            "train_loss": accum_loss,
            "train_ppl": safe_exp(accum_loss),
            "lr": float(scheduler.get_last_lr()[0]),
            "elapsed_s": elapsed_s,
            "step_time_s": step_time_s,
            "tokens_seen": tokens_seen,
            "tokens_per_s": tokens_seen / max(elapsed_s, 1e-9),
        }

        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            val_loss = run_eval()
            rec["val_loss"] = val_loss
            rec["val_ppl"] = safe_exp(val_loss)
            val_points.append((step, val_loss))
            best_val = min(best_val, val_loss)

        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if step % cfg.log_interval == 0:
            print(json.dumps(rec, ensure_ascii=False), flush=True)

    summary = {
        "optimizer": cfg.optimizer,
        "data_source": "local_text",
        "max_steps": cfg.max_steps,
        "completed_steps": step,
        "tokens_seen": step * tokens_per_step,
        "train_loss_last": accum_loss if step else None,
        "train_loss_best": best_train,
        "train_loss_mean": train_loss_sum / max(step, 1),
        "best_val_loss": best_val,
        "best_val_ppl": safe_exp(best_val) if math.isfinite(best_val) else None,
        "last_val_loss": val_points[-1][1] if val_points else None,
        "last_val_ppl": safe_exp(val_points[-1][1]) if val_points else None,
        "elapsed_s": time.time() - start,
        "tokens_per_s": (step * tokens_per_step) / max(time.time() - start, 1e-9),
        "n_embd": cfg.n_embd,
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
    }
    save_json(diag_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
