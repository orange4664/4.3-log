#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


METHODS = {
    "adamw": {"optimizer": "adamw"},
    "muon_ns": {
        "optimizer": "muon_ns",
        "rt_stream_k": 0,
        "rt_power_iters": 1,
        "rt_log_interval": 5,
    },
    "rt_v43_stream": {
        "optimizer": "rt_v43_stream",
        "rt_stream_k": 0,
        "rt_power_iters": 1,
        "rt_period": 1,
        "rt_lr_switch": 0.02,
        "rt_hard_delta": 0.2,
        "rt_soft_delta": 0.5,
        "rt_hard_phi_threshold": 0.0,
        "rt_soft_phi_threshold": 0.0,
        "rt_log_interval": 5,
    },
    "rt_v6_fdt_metric": {
        "optimizer": "rt_v6_fdt_metric",
        "rt_stream_k": 0,
        "rt_power_iters": 1,
        "rt_period": 1,
        "rt_lr_switch": 0.02,
        "rt_hard_delta": 0.2,
        "rt_soft_delta": 0.5,
        "metric_alpha_row": 0.25,
        "metric_alpha_col": 0.25,
        "metric_tmin": 0.5,
        "metric_tmax": 2.0,
        "metric_lambda_noise": 1.0,
        "rt_log_interval": 5,
    },
}


def base_config(data_dir: str, train_root: str) -> dict:
    return {
        "dataset_params": {
            "batch_size": 1,
            "split_file": None,
            "test_ratio": 0.1,
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "graph_data_path": data_dir,
            "num_workers": 0,
        },
        "losses_metrics": {
            "losses": [
                {
                    "loss_weight": 1.0,
                    "metric": "mae",
                    "prediction": "hamiltonian",
                    "target": "hamiltonian",
                }
            ],
            "metrics": [
                {"metric": "mae", "prediction": "hamiltonian", "target": "hamiltonian"}
            ],
        },
        "optim_params": {
            "lr": 0.003,
            "lr_decay": 0.5,
            "lr_patience": 3,
            "gradient_clip_val": 0.0,
            "max_epochs": 8,
            "min_epochs": 1,
            "stop_patience": 8,
        },
        "output_nets": {
            "output_module": "HamGNN_out",
            "HamGNN_out": {
                "ham_only": True,
                "ham_type": "openmx",
                "nao_max": 14,
                "add_H0": True,
                "symmetrize": True,
                "calculate_band_energy": False,
                "num_k": 5,
                "band_num_control": 8,
                "k_path": None,
                "soc_switch": False,
                "nonlinearity_type": "gate",
                "spin_constrained": False,
                "collinear_spin": False,
                "minMagneticMoment": 0.5,
            },
        },
        "profiler_params": {
            "progress_bar_refresh_rat": 1,
            "train_dir": train_root,
        },
        "representation_nets": {
            "HamGNN_pre": {
                "legacy_edge_update": False,
                "cutoff": 8.0,
                "cutoff_func": "cos",
                "edge_sh_normalization": "component",
                "edge_sh_normalize": True,
                "irreps_edge_sh": "0e + 1o + 2e",
                "irreps_node_features": "16x0e+8x1o+4x1e+4x2e",
                "num_layers": 1,
                "num_radial": 16,
                "num_types": 32,
                "rbf_func": "bessel",
                "set_features": True,
                "radial_MLP": [16, 16],
                "use_corr_prod": False,
                "correlation": 1,
                "num_hidden_features": 8,
                "use_kan": False,
                "radius_scale": 1.01,
                "build_internal_graph": False,
                "use_gradient_checkpointing": False,
            },
        },
        "setup": {
            "GNN_Net": "HamGNNpre",
            "accelerator": None,
            "ignore_warnings": True,
            "checkpoint_path": None,
            "load_from_checkpoint": False,
            "resume": False,
            "num_gpus": 1,
            "precision": 32,
            "property": "hamiltonian",
            "stage": "fit",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--methods", default="adamw,muon_ns,rt_v43_stream,rt_v6_fdt_metric")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    Path(args.train_root).mkdir(parents=True, exist_ok=True)

    manifest = []
    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        cfg = base_config(args.data_dir, os.path.join(args.train_root, method))
        cfg["optim_params"].update(METHODS[method])
        path = outdir / f"{method}.yaml"
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        manifest.append((method, str(path)))

    manifest_path = outdir / "manifest.tsv"
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("method\tconfig\n")
        for method, path in manifest:
            f.write(f"{method}\t{path}\n")
    print(manifest_path)


if __name__ == "__main__":
    main()
