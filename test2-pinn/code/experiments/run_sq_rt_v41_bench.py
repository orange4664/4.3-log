#!/usr/bin/env python3
"""SQ-RT-Muon v4.1 fair benchmark.

Families:
  * muon: exact polar/msign endpoint.
  * htmuon_like: exact-SVD U Sigma^p V^T proxy, p-grid.
  * rt_fixed: v3 RT-Muon, fixed hyperparameter triples.
  * sq_rt: v4 Self-Quenched RT-Muon, same number of presets as HT and fixed RT.
  * optional ablations: sq_no_diffusion, sq_no_uncertainty, sq_no_tube_penalty,
    sq_no_quench, sq_no_noise, sq_no_consistency.

This benchmark is a toy/mechanism screen.  It is not an official HTMuon LLM/image
benchmark.  All exact-SVD matrix updates are Frobenius/RMS aligned inside the toy.

v4.1 differs from v4 only in the thermostat/grid/evaluation layer:
  * finer beta grid for near-Muon late-stage corrections;
  * stronger phi-threshold presets for self-quench;
  * per-learning-rate scorecards and useful-LR-window summaries.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rtmuon.toy import ToyConfig, simulate_one


def parse_list(s: str, typ):
    if s is None or str(s).strip() == "":
        return []
    return [typ(x.strip()) for x in str(s).split(",") if x.strip()]


def auto_workers(requested: int) -> int:
    if requested and requested > 0:
        return requested
    env = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("NSLOTS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return max(1, min(32, os.cpu_count() or 1))


def parse_rt_presets(s: str) -> List[Tuple[float, float, float]]:
    out: List[Tuple[float, float, float]] = []
    for item in (s or "").split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"Bad RT preset '{item}', expected delta:lambda_nu:lambda_c")
        out.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return out


def parse_sq_presets(s: str) -> List[Tuple[float, float, float, float, float, float, float]]:
    """Parse 'delta:lambda_nu:lambda_c:lambda_diff:lambda_uncert:lambda_tube:phi_threshold'."""
    out: List[Tuple[float, float, float, float, float, float, float]] = []
    for item in (s or "").split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 7:
            raise ValueError(
                f"Bad SQ preset '{item}', expected "
                "delta:lambda_nu:lambda_c:lambda_diff:lambda_uncert:lambda_tube:phi_threshold"
            )
        out.append(tuple(float(x) for x in parts))
    return out


def rt_label(delta: float, lnu: float, lc: float) -> str:
    return f"delta={delta:g};lnu={lnu:g};lc={lc:g}"


def sq_label(preset: Tuple[float, float, float, float, float, float, float]) -> str:
    d, n, c, ld, lu, lt, th = preset
    return f"{rt_label(d,n,c)};ld={ld:g};lu={lu:g};ltube={lt:g};thr={th:g}"


def base_entry(family: str, method: str, algo_config: str) -> Dict:
    return {
        "family": family,
        "method": method,
        "algo_config": algo_config,
        "ht_p": None,
        "rt_delta": 0.35,
        "rt_lambda_nu": 2.0,
        "rt_lambda_c": 0.25,
        "sq_lambda_diff": 0.0,
        "sq_lambda_uncert": 0.0,
        "sq_lambda_tube": 0.0,
        "sq_phi_threshold": 0.0,
    }


def make_method_configs(args) -> List[Dict]:
    entries: List[Dict] = []

    if args.include_muon:
        e = base_entry("muon", "muon", "muon")
        entries.append(e)

    ht_ps = parse_list(args.ht_p_grid, float)
    rt_fixed = parse_rt_presets(args.rt_fixed_grid)
    sq_presets = parse_sq_presets(args.sq_preset_grid)

    if args.mode == "budget_matched":
        counts = [len(ht_ps), len(rt_fixed), len(sq_presets)]
        if len(set(counts)) != 1:
            raise ValueError(
                "budget_matched requires equal config counts for ht_p_grid, rt_fixed_grid, and sq_preset_grid; "
                f"got {counts}"
            )
    elif args.mode in {"diagnostic_fullgrid", "fixed_defaults"}:
        pass
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    if args.mode == "fixed_defaults":
        ht_ps = [args.ht_default_p]
        rt_fixed = [(args.rt_default_delta, args.rt_default_lambda_nu, args.rt_default_lambda_c)]
        sq_presets = [tuple(parse_sq_presets(args.sq_default_preset)[0])]

    for hp in ht_ps:
        e = base_entry("htmuon_like", f"htmuon_p{hp:g}", f"p={hp:g}")
        e["ht_p"] = float(hp)
        entries.append(e)

    for d, n, c in rt_fixed:
        e = base_entry("rt_fixed", "rt_muon", rt_label(d, n, c))
        e.update(rt_delta=float(d), rt_lambda_nu=float(n), rt_lambda_c=float(c))
        entries.append(e)

    for preset in sq_presets:
        d, n, c, ld, lu, lt, th = preset
        e = base_entry("sq_rt", "sq_rt_muon", sq_label(preset))
        e.update(
            rt_delta=float(d), rt_lambda_nu=float(n), rt_lambda_c=float(c),
            sq_lambda_diff=float(ld), sq_lambda_uncert=float(lu), sq_lambda_tube=float(lt),
            sq_phi_threshold=float(th),
        )
        entries.append(e)

    if args.include_old_sfem:
        entries.append(base_entry("old_sfem_baseline", "sfem_eps4_ht", "fixed_eps4_ht"))

    if args.include_sq_ablations:
        ab_methods = ["sq_no_diffusion", "sq_no_uncertainty", "sq_no_tube_penalty", "sq_no_quench", "sq_no_noise", "sq_no_consistency"]
        for ab in ab_methods:
            for preset in sq_presets:
                d, n, c, ld, lu, lt, th = preset
                e = base_entry(ab, ab, sq_label(preset))
                e.update(
                    rt_delta=float(d), rt_lambda_nu=float(n), rt_lambda_c=float(c),
                    sq_lambda_diff=float(ld), sq_lambda_uncert=float(lu), sq_lambda_tube=float(lt),
                    sq_phi_threshold=float(th),
                )
                entries.append(e)

    if args.include_rt_ablations:
        for ab in ["rt_no_cv", "rt_no_noise", "rt_no_consistency", "rt_fixed_beta1"]:
            for d, n, c in rt_fixed:
                e = base_entry(ab, ab, rt_label(d, n, c))
                e.update(rt_delta=float(d), rt_lambda_nu=float(n), rt_lambda_c=float(c))
                entries.append(e)

    return entries


def make_tasks(args):
    settings = parse_list(args.settings, str)
    lrs = parse_list(args.lrs, float)
    beta_grid = tuple(parse_list(args.beta_grid, float))
    entries = make_method_configs(args)
    tasks = []
    for setting in settings:
        for lr in lrs:
            for seed in range(args.seeds):
                for ent in entries:
                    cfg = ToyConfig(
                        setting=setting, n=args.n, steps=args.steps, lr=lr,
                        momentum=args.momentum, noise=args.noise,
                        offdiag_curv=args.offdiag_curv, seed=seed,
                        kappa_mode=args.kappa_mode,
                        rt_delta=ent["rt_delta"], rt_lambda_nu=ent["rt_lambda_nu"],
                        rt_lambda_c=ent["rt_lambda_c"],
                        sq_lambda_diff=ent["sq_lambda_diff"],
                        sq_lambda_uncert=ent["sq_lambda_uncert"],
                        sq_lambda_tube=ent["sq_lambda_tube"],
                        sq_phi_threshold=ent["sq_phi_threshold"],
                        beta_grid=beta_grid,
                    )
                    tasks.append((ent, cfg))
    return tasks, entries


def run_task(task):
    ent, cfg = task
    rec = simulate_one(ent["method"], cfg)
    for k, v in ent.items():
        if k not in {"method"}:
            rec[k] = v
    return rec


def summarize(df: pd.DataFrame, outdir: Path, entries: List[Dict], args) -> None:
    metrics = [
        "auc_loss", "final_loss", "auc_weak_error", "mean_tail_update_mass",
        "mean_entropy", "mean_beta", "beta_zero_rate", "accept_rate",
        "mean_cv_gain", "mean_raw_cv_gain", "mean_phi", "mean_diffusion_cost",
        "mean_uncertainty_cost", "mean_tube_penalty", "mean_fro_ratio",
        "mean_abs_w_minus_1", "mean_min_w", "mean_max_w",
    ]
    group_cols = [
        "setting", "family", "method", "algo_config", "lr", "ht_p",
        "rt_delta", "rt_lambda_nu", "rt_lambda_c", "sq_lambda_diff",
        "sq_lambda_uncert", "sq_lambda_tube", "sq_phi_threshold",
    ]
    agg = df.groupby(group_cols, dropna=False)[metrics].agg(["mean", "std", "count"]).reset_index()
    agg.columns = ["_".join([str(x) for x in col if str(x)]) for col in agg.columns]
    agg.to_csv(outdir / "summary_by_config.csv", index=False)

    best_rows = []
    for metric in ["auc_loss", "final_loss"]:
        mcol = f"{metric}_mean"
        for (setting, family), g in agg.groupby(["setting", "family"]):
            row = g.sort_values(mcol).iloc[0].to_dict()
            row["selection_metric"] = metric
            best_rows.append(row)
    best = pd.DataFrame(best_rows)
    best.to_csv(outdir / "best_by_family_metric.csv", index=False)

    comp_rows = []
    comp_fams = ["muon", "htmuon_like", "rt_fixed", "sq_rt", "old_sfem_baseline"]
    for setting in sorted(df["setting"].unique()):
        for metric in ["auc_loss", "final_loss"]:
            mcol = f"{metric}_mean"
            sub = best[(best["setting"] == setting) & (best["selection_metric"] == metric)]
            row = {"setting": setting, "metric": metric}
            for fam in comp_fams:
                f = sub[sub["family"] == fam]
                if len(f):
                    r = f.iloc[0]
                    row[f"{fam}_best"] = float(r[mcol])
                    row[f"{fam}_config"] = str(r["algo_config"])
                    row[f"{fam}_lr"] = float(r["lr"])
                else:
                    row[f"{fam}_best"] = float("nan")
                    row[f"{fam}_config"] = ""
                    row[f"{fam}_lr"] = float("nan")
            row["sq_beats_ht"] = bool(row.get("sq_rt_best", float("inf")) < row.get("htmuon_like_best", float("inf")))
            row["sq_beats_rt_fixed"] = bool(row.get("sq_rt_best", float("inf")) < row.get("rt_fixed_best", float("inf")))
            row["sq_beats_muon"] = bool(row.get("sq_rt_best", float("inf")) < row.get("muon_best", float("inf")))
            comp_rows.append(row)
    pd.DataFrame(comp_rows).to_csv(outdir / "oracle_comparison.csv", index=False)

    # Muon-quality constrained AUC: among configs whose final loss is within eps of tuned Muon final,
    # choose the best AUC.  This targets "fast while not losing Muon final quality".
    eps_values = [float(x) for x in parse_list(args.quality_eps_grid, float)]
    q_rows = []
    for setting in sorted(agg["setting"].unique()):
        setting_agg = agg[agg["setting"] == setting]
        mu = setting_agg[setting_agg["family"] == "muon"]
        if len(mu) == 0:
            continue
        mu_best_final = float(mu["final_loss_mean"].min())
        for eps in eps_values:
            threshold = (1.0 + eps) * mu_best_final
            for fam, g in setting_agg.groupby("family"):
                feasible = g[g["final_loss_mean"] <= threshold]
                row = {
                    "setting": setting,
                    "family": fam,
                    "eps": eps,
                    "muon_best_final": mu_best_final,
                    "final_threshold": threshold,
                    "num_feasible_rows": int(len(feasible)),
                    "num_total_rows": int(len(g)),
                }
                if len(feasible):
                    r = feasible.sort_values("auc_loss_mean").iloc[0]
                    row.update(
                        constrained_auc=float(r["auc_loss_mean"]),
                        constrained_final=float(r["final_loss_mean"]),
                        constrained_lr=float(r["lr"]),
                        constrained_config=str(r["algo_config"]),
                    )
                else:
                    row.update(
                        constrained_auc=float("nan"),
                        constrained_final=float("nan"),
                        constrained_lr=float("nan"),
                        constrained_config="",
                    )
                q_rows.append(row)
    pd.DataFrame(q_rows).to_csv(outdir / "muon_quality_constrained_auc.csv", index=False)

    # Positive-p HT oracle: sometimes p=0 is just Muon.  Report p>0 HT separately.
    ht_pos_rows = []
    for setting in sorted(agg["setting"].unique()):
        ht = agg[(agg["setting"] == setting) & (agg["family"] == "htmuon_like") & (agg["ht_p"].fillna(0) > 0)]
        sq = best[(best["setting"] == setting) & (best["family"] == "sq_rt")]
        for metric in ["auc_loss", "final_loss"]:
            mcol = f"{metric}_mean"
            row = {"setting": setting, "metric": metric}
            if len(ht):
                h = ht.sort_values(mcol).iloc[0]
                row["best_positive_ht"] = float(h[mcol])
                row["best_positive_ht_config"] = str(h["algo_config"])
                row["best_positive_ht_lr"] = float(h["lr"])
            else:
                row["best_positive_ht"] = float("nan")
                row["best_positive_ht_config"] = ""
                row["best_positive_ht_lr"] = float("nan")
            s = sq[sq["selection_metric"] == metric]
            if len(s):
                r = s.iloc[0]
                row["sq_best"] = float(r[mcol])
                row["sq_config"] = str(r["algo_config"])
                row["sq_lr"] = float(r["lr"])
            else:
                row["sq_best"] = float("nan")
                row["sq_config"] = ""
                row["sq_lr"] = float("nan")
            row["sq_beats_positive_ht"] = bool(row["sq_best"] < row["best_positive_ht"])
            ht_pos_rows.append(row)
    pd.DataFrame(ht_pos_rows).to_csv(outdir / "positive_ht_comparison.csv", index=False)


    # Per-learning-rate family best scorecard.  This makes the LR-regime effect explicit:
    # tiny LR often favors exact Muon final loss, while medium/large LR favors RT/SQ-RT speed and stability.
    per_lr_rows = []
    for setting in sorted(agg["setting"].unique()):
        for lr in sorted(agg[agg["setting"] == setting]["lr"].unique()):
            sub_lr = agg[(agg["setting"] == setting) & (agg["lr"] == lr)]
            for metric in ["auc_loss", "final_loss"]:
                mcol = f"{metric}_mean"
                row = {"setting": setting, "lr": float(lr), "metric": metric}
                for fam, g in sub_lr.groupby("family"):
                    r = g.sort_values(mcol).iloc[0]
                    row[f"{fam}_best"] = float(r[mcol])
                    row[f"{fam}_config"] = str(r["algo_config"])
                sqv = row.get("sq_rt_best", float("nan"))
                muv = row.get("muon_best", float("nan"))
                htv = row.get("htmuon_like_best", float("nan"))
                rtv = row.get("rt_fixed_best", float("nan"))
                row["sq_beats_muon"] = bool(pd.notna(sqv) and pd.notna(muv) and sqv < muv)
                row["sq_beats_ht"] = bool(pd.notna(sqv) and pd.notna(htv) and sqv < htv)
                row["sq_beats_rt_fixed"] = bool(pd.notna(sqv) and pd.notna(rtv) and sqv < rtv)
                per_lr_rows.append(row)
    pd.DataFrame(per_lr_rows).to_csv(outdir / "lr_regime_scorecard.csv", index=False)

    # Useful LR window: number/range of learning rates whose final loss is within eps of tuned-Muon final.
    # It reports both the feasibility window and the best AUC inside that window.
    lrwin_rows = []
    eps_lr_values = [float(x) for x in parse_list(args.lr_window_eps_grid, float)]
    for setting in sorted(agg["setting"].unique()):
        setting_agg = agg[agg["setting"] == setting]
        mu = setting_agg[setting_agg["family"] == "muon"]
        if len(mu) == 0:
            continue
        mu_best_final = float(mu["final_loss_mean"].min())
        for eps in eps_lr_values:
            threshold = (1.0 + eps) * mu_best_final
            for fam, g in setting_agg.groupby("family"):
                # For each lr, take this family's best final and best AUC over algorithm configs.
                lr_rows = []
                for lr, glr in g.groupby("lr"):
                    best_final = float(glr["final_loss_mean"].min())
                    best_auc = float(glr["auc_loss_mean"].min())
                    # best AUC among configs that satisfy the final threshold at this lr
                    feasible_configs = glr[glr["final_loss_mean"] <= threshold]
                    auc_if_feasible = float(feasible_configs["auc_loss_mean"].min()) if len(feasible_configs) else float("nan")
                    lr_rows.append((float(lr), best_final, best_auc, auc_if_feasible, len(feasible_configs)))
                feasible = [x for x in lr_rows if x[1] <= threshold]
                feasible_auc = [x for x in lr_rows if x[4] > 0]
                row = {
                    "setting": setting,
                    "family": fam,
                    "eps": eps,
                    "muon_best_final": mu_best_final,
                    "final_threshold": threshold,
                    "num_feasible_lrs_by_best_final": len(feasible),
                    "num_total_lrs": len(lr_rows),
                    "min_feasible_lr": min([x[0] for x in feasible], default=float("nan")),
                    "max_feasible_lr": max([x[0] for x in feasible], default=float("nan")),
                    "best_auc_any_lr": min([x[2] for x in lr_rows], default=float("nan")),
                    "best_auc_with_final_feasible_config": min([x[3] for x in feasible_auc], default=float("nan")),
                }
                lrwin_rows.append(row)
    pd.DataFrame(lrwin_rows).to_csv(outdir / "useful_lr_window.csv", index=False)

    # Search budget accounting, excluding seeds.
    lrs = parse_list(args.lrs, float)
    settings = parse_list(args.settings, str)
    budget_rows = []
    for fam, g in pd.DataFrame(entries).groupby("family"):
        n_algo = g[["method", "algo_config"]].drop_duplicates().shape[0]
        budget_rows.append({
            "family": fam,
            "num_algorithm_configs": int(n_algo),
            "num_lrs": len(lrs),
            "num_settings": len(settings),
            "num_seeds": args.seeds,
            "configs_per_setting_excluding_seeds": int(n_algo * len(lrs)),
            "total_runs": int(n_algo * len(lrs) * len(settings) * args.seeds),
        })
    pd.DataFrame(budget_rows).to_csv(outdir / "search_budget.csv", index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, default="results/sq_rt_v4")
    p.add_argument("--mode", choices=["budget_matched", "diagnostic_fullgrid", "fixed_defaults"], default="budget_matched")
    p.add_argument("--settings", type=str, default="fullrank,lowrank,mixed,spiked_noise")
    p.add_argument("--lrs", type=str, default="0.01,0.02,0.05,0.1,0.2,0.3")
    p.add_argument("--quality-eps-grid", type=str, default="0,0.005,0.01,0.02")
    p.add_argument("--lr-window-eps-grid", type=str, default="0,0.005,0.01,0.02", help="Eps values for useful LR window based on tuned-Muon final loss.")
    p.add_argument("--ht-p-grid", type=str, default="0,0.03125,0.0625,0.1,0.125,0.1875,0.25,0.375,0.5")
    p.add_argument("--ht-default-p", type=float, default=0.125)
    p.add_argument(
        "--rt-fixed-grid", type=str,
        default="0.2:1.0:0.0,0.2:2.0:0.25,0.35:1.0:0.25,0.35:2.0:0.25,0.35:4.0:0.5,0.5:2.0:0.25,0.5:4.0:0.5,0.2:1.0:0.25,0.5:1.0:0.25",
        help="v3 RT triples delta:lambda_nu:lambda_c. Count must match ht-p-grid in budget_matched mode.",
    )
    p.add_argument(
        "--sq-preset-grid", type=str,
        default=(
            # 9 configs to match the 9-point HT p-grid.
            # Aggressive entries preserve v3/v4 AUC gains; conservative/high-threshold entries target exact-Muon final quality.
            "0.5:4.0:0.5:0.25:0.25:0.02:0.0,"
            "0.5:4.0:0.5:0.25:0.25:0.02:0.001,"
            "0.5:4.0:0.5:0.5:1.0:0.10:0.005,"
            "0.5:4.0:0.5:1.0:2.0:0.20:0.01,"
            "0.5:4.0:0.5:1.0:2.0:0.20:0.02,"
            "0.35:2.0:0.25:0.5:0.5:0.05:0.005,"
            "0.35:2.0:0.25:1.0:1.0:0.10:0.02,"
            "0.2:1.0:0.0:0.5:0.5:0.05:0.005,"
            "0.2:1.0:0.0:1.0:1.0:0.10:0.02"
        ),
        help="SQ presets delta:lambda_nu:lambda_c:lambda_diff:lambda_uncert:lambda_tube:phi_threshold.",
    )
    p.add_argument("--sq-default-preset", type=str, default="0.35:2.0:0.25:0.5:0.5:0.05:0.0")
    p.add_argument("--rt-default-delta", type=float, default=0.35)
    p.add_argument("--rt-default-lambda-nu", type=float, default=2.0)
    p.add_argument("--rt-default-lambda-c", type=float, default=0.25)
    p.add_argument("--beta-grid", type=str, default="0,0.02,0.05,0.1,0.25,0.5,1,2,4")
    p.add_argument("--include-muon", action="store_true", default=True)
    p.add_argument("--no-muon", dest="include_muon", action="store_false")
    p.add_argument("--include-old-sfem", action="store_true")
    p.add_argument("--include-sq-ablations", action="store_true")
    p.add_argument("--include-rt-ablations", action="store_true")
    p.add_argument("--seeds", type=int, default=16)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--noise", type=float, default=0.03)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--offdiag-curv", type=float, default=0.05)
    p.add_argument("--kappa-mode", choices=["true", "ones"], default="true")
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    workers = auto_workers(args.workers)
    tasks, entries = make_tasks(args)

    with open(outdir / "run_args.json", "w") as f:
        d = vars(args).copy()
        d["workers_resolved"] = workers
        d["num_tasks"] = len(tasks)
        d["num_method_configs"] = len(entries)
        json.dump(d, f, indent=2, ensure_ascii=False)
    pd.DataFrame(entries).to_csv(outdir / "method_configs.csv", index=False)

    print(f"[run_sq_rt_bench] mode={args.mode} tasks={len(tasks)} workers={workers} outdir={outdir}", flush=True)
    records: List[Dict] = []
    if workers == 1:
        for i, t in enumerate(tasks, 1):
            records.append(run_task(t))
            if i % max(1, len(tasks)//10) == 0:
                print(f"  completed {i}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(run_task, t) for t in tasks]
            for i, fut in enumerate(as_completed(futs), 1):
                records.append(fut.result())
                if i % max(1, len(tasks)//10) == 0:
                    print(f"  completed {i}/{len(tasks)}", flush=True)

    df = pd.DataFrame(records)
    df.to_csv(outdir / "records.csv", index=False)
    summarize(df, outdir, entries, args)
    print(f"[run_sq_rt_bench] done: {outdir}", flush=True)


if __name__ == "__main__":
    main()
