"""Wrapper entry point for running HamGNN with RT/FDT optimizers.

Usage:
  python -m hamgnn_rt.run_hamgnn_rt --config config.yaml

The wrapper reads optim_params.optimizer from YAML, exports HAMGNN_RT_* env vars,
monkey-patches HamGNN's Lightning Model.configure_optimizers, and then calls the
normal HamGNN CLI. It does not modify HamGNN source files.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict

import yaml


def _flatten_optim_params(d: Dict[str, Any]) -> Dict[str, str]:
    out = {}
    mapping = {
        "optimizer": "HAMGNN_RT_OPTIMIZER",
        "optimizer_type": "HAMGNN_RT_OPTIMIZER",
        "weight_decay": "HAMGNN_RT_WEIGHT_DECAY",
        "muon_momentum": "HAMGNN_RT_MOMENTUM",
        "rt_momentum": "HAMGNN_RT_MOMENTUM",
        "muon_ns_steps": "HAMGNN_MUON_NS_STEPS",
        "rt_ns_steps": "HAMGNN_RT_NS_STEPS",
        "muon_nesterov": "HAMGNN_MUON_NESTEROV",
        "rt_stream_k": "HAMGNN_RT_STREAM_K",
        "rt_power_iters": "HAMGNN_RT_POWER_ITERS",
        "rt_period": "HAMGNN_RT_PERIOD",
        "rt_lr_switch": "HAMGNN_RT_LR_SWITCH",
        "rt_hard_delta": "HAMGNN_RT_HARD_DELTA",
        "rt_soft_delta": "HAMGNN_RT_SOFT_DELTA",
        "rt_lambda_noise": "HAMGNN_RT_LAMBDA_NOISE",
        "rt_lambda_consistency": "HAMGNN_RT_LAMBDA_CONSISTENCY",
        "rt_lambda_c": "HAMGNN_RT_LAMBDA_C",
        "rt_lambda_diff": "HAMGNN_RT_LAMBDA_DIFF",
        "rt_lambda_uncert": "HAMGNN_RT_LAMBDA_UNCERT",
        "rt_lambda_tube": "HAMGNN_RT_LAMBDA_TUBE",
        "rt_beta_grid": "HAMGNN_RT_BETA_GRID",
        "rt_hard_phi_threshold": "HAMGNN_RT_HARD_PHI_THRESHOLD",
        "rt_soft_phi_threshold": "HAMGNN_RT_SOFT_PHI_THRESHOLD",
        "rt_soft_tau": "HAMGNN_RT_SOFT_TAU",
        "metric_alpha_row": "HAMGNN_RT_METRIC_ALPHA_ROW",
        "metric_alpha_col": "HAMGNN_RT_METRIC_ALPHA_COL",
        "metric_lambda_noise": "HAMGNN_RT_METRIC_LAMBDA_NOISE",
        "metric_tmin": "HAMGNN_RT_METRIC_TMIN",
        "metric_tmax": "HAMGNN_RT_METRIC_TMAX",
        "metric_ema_beta": "HAMGNN_RT_METRIC_EMA_BETA",
        "ht_power": "HAMGNN_RT_HT_POWER",
        "mclip_delta": "HAMGNN_RT_MCLIP_DELTA",
        "rt_log_interval": "HAMGNN_RT_LOG_INTERVAL",
    }
    for k, env in mapping.items():
        if k in d and d[k] is not None:
            out[env] = str(d[k])
    return out


def patch_hamgnn_optimizer() -> None:
    import torch.optim as optim
    from hamgnn.models import Model as model_module
    from hamgnn_rt.optimizers import build_optimizer_from_env

    def configure_optimizers(self):
        optimizer = build_optimizer_from_env(
            self.named_parameters(),
            lr=self.lr,
            eps=self.epsilon,
            beta1=getattr(self, "beta1", 0.9),
            beta2=getattr(self, "beta2", 0.999),
            amsgrad=getattr(self, "amsgrad", True),
        )
        scheduler = {
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.lr_decay,
                patience=self.lr_patience,
                threshold=1e-6,
                cooldown=self.lr_patience // 2,
                min_lr=1e-6,
            ),
            "monitor": self.lr_monitor,
            "interval": "epoch",
            "frequency": 1,
            "strict": True,
        }
        return [optimizer], [scheduler]

    model_module.Model.configure_optimizers = configure_optimizers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--diag-dir", default=None)
    parser.add_argument("--run-name", default=None)
    args, unknown = parser.parse_known_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    optim_params = cfg.get("optim_params", {}) or {}
    env_updates = _flatten_optim_params(optim_params)
    for k, v in env_updates.items():
        os.environ.setdefault(k, v)
    if args.diag_dir:
        os.environ["HAMGNN_RT_DIAG_DIR"] = args.diag_dir
    if args.run_name:
        os.environ["HAMGNN_RT_RUN_NAME"] = args.run_name
    os.environ.setdefault("HAMGNN_RT_OPTIMIZER", "adamw")

    patch_hamgnn_optimizer()

    # Call the original CLI with a clean argv that HamGNN understands.
    from hamgnn.main import HamGNN

    sys.argv = ["HamGNN2.0", "--config", args.config] + unknown
    HamGNN()


if __name__ == "__main__":
    main()
