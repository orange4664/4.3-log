#!/usr/bin/env python3
"""SQ-RT-Muon v4.3 fair benchmark.

v4.3 does not change the low-level SQ-RT update.  It adds a two-thermostat
selector family:

    low learning-rate / final-quality regime  -> v4.1 hard-quench SQ-RT
    medium-large learning-rate / speed regime -> v4.2 soft-quench SQ-RT

This is intentionally a selector benchmark, not a new penalty term.  It tests
whether the empirical v1--v4 observation (small-lr Muon/hard-quench is best for
exact final loss; larger-lr soft/RT dominates AUC and robustness) can be turned
into a clean optimizer family with matched tuning budget.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from rtmuon.toy import ToyConfig, simulate_one


def parse_list(s: str, typ):
    return [typ(x.strip()) for x in str(s or '').split(',') if x.strip()]


def resolve_workers(n: int) -> int:
    if n and n > 0:
        return int(n)
    for k in ('SLURM_CPUS_PER_TASK', 'NSLOTS'):
        v = os.environ.get(k)
        if v:
            try:
                return max(1, int(v))
            except ValueError:
                pass
    return max(1, min(32, os.cpu_count() or 1))


def parse_rt(s: str) -> List[Tuple[float, float, float]]:
    out = []
    for item in str(s or '').split(','):
        item = item.strip()
        if not item:
            continue
        p = item.split(':')
        if len(p) != 3:
            raise ValueError(f'bad rt preset {item!r}; expected delta:lambda_nu:lambda_c')
        out.append(tuple(float(x) for x in p))
    return out


def parse_sq(s: str) -> List[Tuple[float, float, float, float, float, float, float, float]]:
    out = []
    for item in str(s or '').split(','):
        item = item.strip()
        if not item:
            continue
        p = item.split(':')
        if len(p) != 8:
            raise ValueError(f'bad sq preset {item!r}; expected d:lnu:lc:ld:lu:lt:thr:tau')
        out.append(tuple(float(x) for x in p))
    return out


def repeat_to_len(vals, n):
    vals = list(vals)
    if not vals:
        vals = [0.05]
    return [vals[i % len(vals)] for i in range(n)]


def rt_label(d, n, c):
    return f'delta={d:g};lnu={n:g};lc={c:g}'


def sq_label(x, mode, diff):
    d, n, c, ld, lu, lt, thr, tau = x
    return f'mode={mode};diff={diff};{rt_label(d,n,c)};ld={ld:g};lu={lu:g};lt={lt:g};thr={thr:g};tau={tau:g}'


def entry(family, method, algo_config):
    return dict(
        family=family, method=method, method_label=method, algo_config=algo_config, ht_p=None,
        rt_delta=0.35, rt_lambda_nu=2.0, rt_lambda_c=0.25,
        sq_lambda_diff=0.0, sq_lambda_uncert=0.0, sq_lambda_tube=0.0,
        sq_phi_threshold=0.0, sq_soft_tau=0.001,
        sq_quench_mode='hard', sq_diffusion_mode='mean',
        selector_switch=float('nan'), selector_mode='none', actual_method=method,
    )


def apply_sq_params(e, x, qmode, dmode):
    d, n, c, ld, lu, lt, thr, tau = x
    e.update(
        rt_delta=d, rt_lambda_nu=n, rt_lambda_c=c,
        sq_lambda_diff=ld, sq_lambda_uncert=lu, sq_lambda_tube=lt,
        sq_phi_threshold=thr, sq_soft_tau=tau,
        sq_quench_mode=qmode, sq_diffusion_mode=dmode,
    )
    return e


def configs(args):
    E = []
    if args.include_muon:
        E.append(entry('muon', 'muon', 'muon'))

    ht = parse_list(args.ht_p_grid, float)
    rt = parse_rt(args.rt_fixed_grid)
    hard = parse_sq(args.sq_hard_grid)
    soft = parse_sq(args.sq_soft_grid)
    switches = repeat_to_len(parse_list(args.selector_switch_grid, float), max(len(hard), len(soft)))

    if args.mode == 'fixed_defaults':
        ht = [args.ht_default_p]
        rt = [(args.rt_default_delta, args.rt_default_lambda_nu, args.rt_default_lambda_c)]
        hard = [parse_sq(args.sq_default_hard)[0]]
        soft = [parse_sq(args.sq_default_soft)[0]]
        switches = [args.selector_default_switch]
    elif args.mode == 'budget_matched':
        counts = [len(ht), len(rt), len(hard), len(soft)]
        if len(set(counts)) != 1:
            raise ValueError(f'budget_matched needs equal non-selector counts, got {counts}')

    # Baselines.
    for p in ht:
        e = entry('htmuon_like', f'htmuon_p{p:g}', f'p={p:g}')
        e['ht_p'] = float(p)
        E.append(e)
    for d, n, c in rt:
        e = entry('rt_fixed', 'rt_muon', rt_label(d, n, c))
        e.update(rt_delta=d, rt_lambda_nu=n, rt_lambda_c=c)
        E.append(e)
    for x in hard:
        e = entry('sq_hard_v41', 'sq_rt_muon', sq_label(x, 'hard', 'mean'))
        apply_sq_params(e, x, 'hard', 'mean')
        E.append(e)
    for x in soft:
        e = entry('sq_soft_v42', 'sq_rt_soft', sq_label(x, 'soft', 'normalized'))
        apply_sq_params(e, x, 'soft', 'normalized')
        E.append(e)

    # v4.3 selector: one selector config pairs hard[i] with soft[i].
    if args.include_selector:
        if len(hard) != len(soft):
            raise ValueError('selector requires hard and soft grids to have equal length')
        for i, (xh, xs) in enumerate(zip(hard, soft)):
            sw = float(switches[i])
            e = entry(
                'sq_selector_v43', 'sq_selector_v43',
                f'lr_switch={sw:g};low=[{sq_label(xh,"hard","mean")}];high=[{sq_label(xs,"soft","normalized")}]'
            )
            e['selector_switch'] = sw
            # Store hard/soft tuples as strings for method_configs.csv; make_tasks will unpack from lists by index.
            e['selector_index'] = i
            e['hard_preset'] = ':'.join(f'{v:g}' for v in xh)
            e['soft_preset'] = ':'.join(f'{v:g}' for v in xs)
            E.append(e)

    if args.include_sq_ablations:
        for ab in ['sq_no_quench', 'sq_no_diffusion', 'sq_no_uncertainty', 'sq_no_tube_penalty', 'sq_no_noise', 'sq_no_consistency']:
            for x in soft:
                e = entry(ab, ab, sq_label(x, 'soft', 'normalized'))
                apply_sq_params(e, x, 'soft', 'normalized')
                E.append(e)
    return E, hard, soft


def selected_selector_entry(e, hard_grid, soft_grid, lr):
    idx = int(e['selector_index'])
    sw = float(e['selector_switch'])
    if float(lr) < sw:
        x = hard_grid[idx]
        ee = dict(e)
        ee['actual_method'] = 'sq_rt_muon'
        ee['method_for_run'] = 'sq_rt_muon'
        ee['method_label'] = 'sq_selector_v43'
        ee['selector_mode'] = 'hard_low_lr'
        apply_sq_params(ee, x, 'hard', 'mean')
        return ee
    x = soft_grid[idx]
    ee = dict(e)
    ee['actual_method'] = 'sq_rt_soft'
    ee['method_for_run'] = 'sq_rt_soft'
    ee['method_label'] = 'sq_selector_v43'
    ee['selector_mode'] = 'soft_high_lr'
    apply_sq_params(ee, x, 'soft', 'normalized')
    return ee


def make_tasks(args):
    beta = tuple(parse_list(args.beta_grid, float))
    E, hard_grid, soft_grid = configs(args)
    T = []
    for setting in parse_list(args.settings, str):
        for lr in parse_list(args.lrs, float):
            for seed in range(args.seeds):
                for e0 in E:
                    e = selected_selector_entry(e0, hard_grid, soft_grid, lr) if e0['family'] == 'sq_selector_v43' else dict(e0, method_for_run=e0['method'], actual_method=e0['method'], selector_mode=e0.get('selector_mode', 'none'))
                    cfg = ToyConfig(
                        setting=setting, n=args.n, steps=args.steps, lr=lr,
                        momentum=args.momentum, noise=args.noise, offdiag_curv=args.offdiag_curv,
                        seed=seed, kappa_mode=args.kappa_mode,
                        rt_delta=e['rt_delta'], rt_lambda_nu=e['rt_lambda_nu'], rt_lambda_c=e['rt_lambda_c'],
                        sq_lambda_diff=e['sq_lambda_diff'], sq_lambda_uncert=e['sq_lambda_uncert'], sq_lambda_tube=e['sq_lambda_tube'],
                        sq_phi_threshold=e['sq_phi_threshold'], sq_soft_tau=e['sq_soft_tau'],
                        sq_quench_mode=e['sq_quench_mode'], sq_diffusion_mode=e['sq_diffusion_mode'],
                        beta_grid=beta,
                    )
                    T.append((e, cfg))
    return T, E


def run_task(task):
    e, cfg = task
    actual = e.get('method_for_run', e['method'])
    rec = simulate_one(actual, cfg)
    rec['actual_method'] = actual
    rec['method'] = e.get('method_label', e['method'])
    for k, v in e.items():
        if k not in {'method_for_run'}:
            rec[k] = v
    return rec


def flat_cols(df):
    df = df.copy()
    df.columns = ['_'.join([str(x) for x in c if str(x)]) if isinstance(c, tuple) else str(c) for c in df.columns]
    return df


def summarize(df, out, E, args):
    metrics = [
        'auc_loss', 'final_loss', 'auc_weak_error', 'mean_tail_update_mass',
        'mean_entropy', 'mean_beta', 'mean_beta_candidate', 'mean_soft_shrink',
        'beta_zero_rate', 'accept_rate', 'mean_cv_gain', 'mean_raw_cv_gain',
        'mean_phi', 'mean_diffusion_cost', 'mean_uncertainty_cost',
        'mean_tube_penalty', 'mean_fro_ratio', 'mean_abs_w_minus_1',
        'mean_min_w', 'mean_max_w'
    ]
    groups = [
        'setting', 'family', 'method', 'algo_config', 'lr', 'ht_p',
        'rt_delta', 'rt_lambda_nu', 'rt_lambda_c',
        'sq_lambda_diff', 'sq_lambda_uncert', 'sq_lambda_tube',
        'sq_phi_threshold', 'sq_soft_tau', 'sq_quench_mode', 'sq_diffusion_mode',
        'selector_switch'
    ]
    agg = flat_cols(df.groupby(groups, dropna=False)[metrics].agg(['mean', 'std', 'count']).reset_index())
    agg.to_csv(out / 'summary_by_config.csv', index=False)

    # Best config per family and metric.
    best_rows = []
    for metric in ['auc_loss', 'final_loss']:
        mc = f'{metric}_mean'
        for (setting, fam), g in agg.groupby(['setting', 'family']):
            r = g.sort_values(mc).iloc[0].to_dict()
            r['selection_metric'] = metric
            best_rows.append(r)
    best = pd.DataFrame(best_rows)
    best.to_csv(out / 'best_by_family_metric.csv', index=False)

    fams = ['muon', 'htmuon_like', 'rt_fixed', 'sq_hard_v41', 'sq_soft_v42', 'sq_selector_v43']
    rows = []
    for setting in sorted(df.setting.unique()):
        for metric in ['auc_loss', 'final_loss']:
            mc = f'{metric}_mean'
            sub = best[(best.setting == setting) & (best.selection_metric == metric)]
            row = {'setting': setting, 'metric': metric}
            for fam in fams:
                ss = sub[sub.family == fam]
                if len(ss):
                    r = ss.iloc[0]
                    row[f'{fam}_best'] = float(r[mc])
                    row[f'{fam}_config'] = str(r.algo_config)
                    row[f'{fam}_lr'] = float(r.lr)
                else:
                    row[f'{fam}_best'] = float('nan')
                    row[f'{fam}_config'] = ''
                    row[f'{fam}_lr'] = float('nan')
            sel = row.get('sq_selector_v43_best', float('nan'))
            row['selector_beats_muon'] = sel < row.get('muon_best', float('nan'))
            row['selector_beats_positive_or_all_ht'] = sel < row.get('htmuon_like_best', float('nan'))
            row['selector_beats_hard'] = sel < row.get('sq_hard_v41_best', float('nan'))
            row['selector_beats_soft'] = sel < row.get('sq_soft_v42_best', float('nan'))
            row['selector_beats_rt_fixed'] = sel < row.get('rt_fixed_best', float('nan'))
            rows.append(row)
    pd.DataFrame(rows).to_csv(out / 'oracle_comparison.csv', index=False)

    # Positive-p HT comparison.
    pos = []
    for setting in sorted(df.setting.unique()):
        for metric in ['auc_loss', 'final_loss']:
            mc = f'{metric}_mean'
            sub = best[(best.setting == setting) & (best.selection_metric == metric)]
            ht = sub[(sub.family == 'htmuon_like') & (sub.ht_p.fillna(0) > 0)].sort_values(mc)
            se = sub[sub.family == 'sq_selector_v43']
            row = {'setting': setting, 'metric': metric}
            if len(ht):
                row.update(best_positive_ht=float(ht.iloc[0][mc]), best_positive_ht_config=str(ht.iloc[0].algo_config), best_positive_ht_lr=float(ht.iloc[0].lr))
            else:
                row.update(best_positive_ht=float('nan'), best_positive_ht_config='', best_positive_ht_lr=float('nan'))
            if len(se):
                row.update(selector_best=float(se.iloc[0][mc]), selector_config=str(se.iloc[0].algo_config), selector_lr=float(se.iloc[0].lr))
            else:
                row.update(selector_best=float('nan'), selector_config='', selector_lr=float('nan'))
            row['selector_beats_positive_ht'] = row['selector_best'] < row['best_positive_ht']
            pos.append(row)
    pd.DataFrame(pos).to_csv(out / 'positive_ht_comparison.csv', index=False)

    # Muon-quality constrained AUC.
    cons = []
    for setting in sorted(agg.setting.unique()):
        sa = agg[agg.setting == setting]
        mu = sa[sa.family == 'muon']
        if not len(mu):
            continue
        mf = float(mu.final_loss_mean.min())
        for eps in parse_list(args.quality_eps_grid, float):
            thr = (1 + eps) * mf
            for fam, g in sa.groupby('family'):
                fg = g[g.final_loss_mean <= thr]
                row = {'setting': setting, 'family': fam, 'eps': eps, 'muon_best_final': mf, 'threshold': thr, 'feasible': bool(len(fg))}
                if len(fg):
                    r = fg.sort_values('auc_loss_mean').iloc[0]
                    row.update(best_auc_under_muon_quality=float(r.auc_loss_mean), config=str(r.algo_config), lr=float(r.lr))
                else:
                    row.update(best_auc_under_muon_quality=float('nan'), config='', lr=float('nan'))
                cons.append(row)
    pd.DataFrame(cons).to_csv(out / 'muon_quality_constrained_auc.csv', index=False)

    # Per-lr scorecard.
    lrrows = []
    for setting in sorted(agg.setting.unique()):
        for lr in sorted(agg[agg.setting == setting].lr.unique()):
            sl = agg[(agg.setting == setting) & (agg.lr == lr)]
            for metric in ['auc_loss', 'final_loss']:
                mc = f'{metric}_mean'
                row = {'setting': setting, 'lr': float(lr), 'metric': metric}
                for fam, g in sl.groupby('family'):
                    r = g.sort_values(mc).iloc[0]
                    row[f'{fam}_best'] = float(r[mc])
                    row[f'{fam}_config'] = str(r.algo_config)
                sel = row.get('sq_selector_v43_best', float('nan'))
                row['selector_beats_muon'] = sel < row.get('muon_best', float('nan'))
                row['selector_beats_ht'] = sel < row.get('htmuon_like_best', float('nan'))
                row['selector_beats_hard'] = sel < row.get('sq_hard_v41_best', float('nan'))
                row['selector_beats_soft'] = sel < row.get('sq_soft_v42_best', float('nan'))
                row['selector_beats_rt_fixed'] = sel < row.get('rt_fixed_best', float('nan'))
                lrrows.append(row)
    lrdf = pd.DataFrame(lrrows)
    lrdf.to_csv(out / 'lr_regime_scorecard.csv', index=False)
    if len(lrdf):
        wc = []
        for (metric, regime), g in [(('auc_loss','all'), lrdf[lrdf.metric == 'auc_loss']), (('final_loss','all'), lrdf[lrdf.metric == 'final_loss']), (('auc_loss','lr_ge_0.05'), lrdf[(lrdf.metric == 'auc_loss') & (lrdf.lr >= 0.05)]), (('final_loss','lr_ge_0.05'), lrdf[(lrdf.metric == 'final_loss') & (lrdf.lr >= 0.05)])]:
            if isinstance(metric, tuple):
                metric, regime = metric
            wc.append(dict(
                metric=metric, regime=regime, n=len(g),
                selector_beats_muon=int(g.selector_beats_muon.sum()) if len(g) else 0,
                selector_beats_ht=int(g.selector_beats_ht.sum()) if len(g) else 0,
                selector_beats_hard=int(g.selector_beats_hard.sum()) if len(g) else 0,
                selector_beats_soft=int(g.selector_beats_soft.sum()) if len(g) else 0,
                selector_beats_rt_fixed=int(g.selector_beats_rt_fixed.sum()) if len(g) else 0,
            ))
        pd.DataFrame(wc).to_csv(out / 'lr_regime_win_counts.csv', index=False)

    # Selector actual-mode diagnostics.
    sdf = df[df.family == 'sq_selector_v43'].copy()
    if len(sdf):
        diag = flat_cols(sdf.groupby(['setting', 'lr', 'algo_config', 'selector_mode'])[
            ['auc_loss', 'final_loss', 'mean_beta', 'beta_zero_rate', 'accept_rate', 'mean_abs_w_minus_1']
        ].agg(['mean', 'count']).reset_index())
        diag.to_csv(out / 'selector_mode_diagnostics.csv', index=False)

    # Search budget from declared config list, not per-lr expanded active mode.
    lrs = parse_list(args.lrs, float)
    settings = parse_list(args.settings, str)
    br = []
    edf = pd.DataFrame(E)
    for fam, g in edf.groupby('family'):
        ncfg = g[['method', 'algo_config']].drop_duplicates().shape[0]
        br.append(dict(family=fam, num_algorithm_configs=int(ncfg), num_lrs=len(lrs), num_settings=len(settings), num_seeds=args.seeds, total_runs=int(ncfg * len(lrs) * len(settings) * args.seeds)))
    pd.DataFrame(br).to_csv(out / 'search_budget.csv', index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', default='results/sq_rt_v43_selector')
    p.add_argument('--mode', choices=['budget_matched', 'fixed_defaults', 'smallfast'], default='budget_matched')
    p.add_argument('--settings', default='fullrank,lowrank,mixed,spiked_noise')
    p.add_argument('--lrs', default='0.01,0.02,0.05,0.1,0.2,0.3')
    p.add_argument('--quality-eps-grid', default='0,0.005,0.01,0.02')
    p.add_argument('--ht-p-grid', default='0,0.03125,0.0625,0.1,0.125,0.1875,0.25,0.375,0.5')
    p.add_argument('--ht-default-p', type=float, default=0.125)
    p.add_argument('--rt-fixed-grid', default='0.2:1.0:0.0,0.2:2.0:0.25,0.35:1.0:0.25,0.35:2.0:0.25,0.35:4.0:0.5,0.5:2.0:0.25,0.5:4.0:0.5,0.2:1.0:0.25,0.5:1.0:0.25')
    p.add_argument('--sq-hard-grid', default='0.5:4.0:0.5:0.25:0.25:0.02:0.0:0.001,0.5:4.0:0.5:0.25:0.25:0.02:0.001:0.001,0.5:4.0:0.5:0.5:1.0:0.10:0.005:0.001,0.5:4.0:0.5:1.0:2.0:0.20:0.01:0.001,0.5:4.0:0.5:1.0:2.0:0.20:0.02:0.001,0.35:2.0:0.25:0.5:0.5:0.05:0.005:0.001,0.35:2.0:0.25:1.0:1.0:0.10:0.02:0.001,0.2:1.0:0.0:0.5:0.5:0.05:0.005:0.001,0.2:1.0:0.0:1.0:1.0:0.10:0.02:0.001')
    p.add_argument('--sq-soft-grid', default='0.5:4.0:0.0:0.00:0.25:0.02:0.0000:0.001,0.5:4.0:0.0:0.02:0.25:0.02:0.0005:0.001,0.5:4.0:0.0:0.05:0.50:0.05:0.0010:0.001,0.5:4.0:0.0:0.10:0.50:0.05:0.0020:0.001,0.35:2.0:0.0:0.02:0.25:0.02:0.0005:0.001,0.35:2.0:0.0:0.05:0.50:0.05:0.0010:0.001,0.35:2.0:0.0:0.10:1.00:0.10:0.0020:0.002,0.2:1.0:0.0:0.02:0.25:0.02:0.0005:0.001,0.2:1.0:0.0:0.05:0.50:0.05:0.0010:0.002')
    p.add_argument('--selector-switch-grid', default='0.05')
    p.add_argument('--selector-default-switch', type=float, default=0.05)
    p.add_argument('--sq-default-hard', default='0.35:2.0:0.25:0.5:0.5:0.05:0.0:0.001')
    p.add_argument('--sq-default-soft', default='0.35:2.0:0.0:0.05:0.5:0.05:0.001:0.001')
    p.add_argument('--rt-default-delta', type=float, default=0.35)
    p.add_argument('--rt-default-lambda-nu', type=float, default=2.0)
    p.add_argument('--rt-default-lambda-c', type=float, default=0.25)
    p.add_argument('--beta-grid', default='0,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.25,0.5,1,2,4')
    p.add_argument('--include-muon', action='store_true', default=True)
    p.add_argument('--no-muon', dest='include_muon', action='store_false')
    p.add_argument('--include-selector', action='store_true', default=True)
    p.add_argument('--no-selector', dest='include_selector', action='store_false')
    p.add_argument('--include-sq-ablations', action='store_true')
    p.add_argument('--seeds', type=int, default=16)
    p.add_argument('--n', type=int, default=32)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--noise', type=float, default=0.03)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--offdiag-curv', type=float, default=0.05)
    p.add_argument('--kappa-mode', choices=['true', 'ones'], default='true')
    p.add_argument('--workers', type=int, default=0)
    args = p.parse_args()

    if args.mode == 'smallfast':
        args.seeds = min(args.seeds, 2)
        args.steps = min(args.steps, 40)
        args.settings = 'fullrank,mixed'
        args.lrs = '0.02,0.1'
        args.ht_p_grid = ','.join(args.ht_p_grid.split(',')[:2])
        args.rt_fixed_grid = ','.join(args.rt_fixed_grid.split(',')[:2])
        args.sq_hard_grid = ','.join(args.sq_hard_grid.split(',')[:2])
        args.sq_soft_grid = ','.join(args.sq_soft_grid.split(',')[:2])
        args.mode = 'budget_matched'

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    W = resolve_workers(args.workers)
    tasks, E = make_tasks(args)
    with open(out / 'run_args.json', 'w') as f:
        d = vars(args).copy()
        d.update(workers_resolved=W, num_tasks=len(tasks), num_method_configs=len(E))
        json.dump(d, f, indent=2, ensure_ascii=False)
    pd.DataFrame(E).to_csv(out / 'method_configs.csv', index=False)
    print(f'[v4.3 selector] tasks={len(tasks)} workers={W} outdir={out}', flush=True)

    recs = []
    if W == 1:
        for t in tasks:
            recs.append(run_task(t))
    else:
        with ProcessPoolExecutor(max_workers=W) as ex:
            futs = [ex.submit(run_task, t) for t in tasks]
            for i, fut in enumerate(as_completed(futs), 1):
                recs.append(fut.result())
                if i % max(1, len(tasks) // 10) == 0:
                    print(f'  completed {i}/{len(tasks)}', flush=True)
    df = pd.DataFrame(recs)
    df.to_csv(out / 'records.csv', index=False)
    summarize(df, out, E, args)
    print('[v4.3 selector] done', flush=True)


if __name__ == '__main__':
    main()
