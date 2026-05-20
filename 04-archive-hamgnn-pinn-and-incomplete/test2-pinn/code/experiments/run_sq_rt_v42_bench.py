#!/usr/bin/env python3
"""SQ-RT-Muon v4.2 fair benchmark.

Default families: Muon, HTMuon-like p-grid, v3 RT fixed, v4.1 hard SQ,
and v4.2 soft SQ. All non-Muon families use matched config counts.
"""
from __future__ import annotations
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from rtmuon.toy import ToyConfig, simulate_one


def parse_list(s: str, typ):
    return [typ(x.strip()) for x in str(s or '').split(',') if x.strip()]


def workers(n: int) -> int:
    if n and n > 0: return n
    for k in ('SLURM_CPUS_PER_TASK','NSLOTS'):
        v=os.environ.get(k)
        if v:
            try: return max(1,int(v))
            except ValueError: pass
    return max(1,min(32,os.cpu_count() or 1))


def parse_rt(s: str) -> List[Tuple[float,float,float]]:
    out=[]
    for item in str(s or '').split(','):
        if not item.strip(): continue
        p=item.split(':')
        if len(p)!=3: raise ValueError(f'bad rt preset {item}')
        out.append(tuple(float(x) for x in p))
    return out


def parse_sq(s: str) -> List[Tuple[float,float,float,float,float,float,float,float]]:
    out=[]
    for item in str(s or '').split(','):
        if not item.strip(): continue
        p=item.split(':')
        if len(p)!=8: raise ValueError(f'bad sq preset {item}; expected d:lnu:lc:ld:lu:lt:thr:tau')
        out.append(tuple(float(x) for x in p))
    return out


def rt_label(d,n,c): return f'delta={d:g};lnu={n:g};lc={c:g}'
def sq_label(x,mode,diff):
    d,n,c,ld,lu,lt,thr,tau=x
    return f'mode={mode};diff={diff};{rt_label(d,n,c)};ld={ld:g};lu={lu:g};lt={lt:g};thr={thr:g};tau={tau:g}'


def entry(family,method,algo_config):
    return dict(family=family, method=method, algo_config=algo_config, ht_p=None,
                rt_delta=0.35, rt_lambda_nu=2.0, rt_lambda_c=0.25,
                sq_lambda_diff=0.0, sq_lambda_uncert=0.0, sq_lambda_tube=0.0,
                sq_phi_threshold=0.0, sq_soft_tau=0.001,
                sq_quench_mode='hard', sq_diffusion_mode='mean')


def configs(args):
    E=[]
    if args.include_muon: E.append(entry('muon','muon','muon'))
    ht=parse_list(args.ht_p_grid,float); rt=parse_rt(args.rt_fixed_grid); hard=parse_sq(args.sq_hard_grid); soft=parse_sq(args.sq_soft_grid)
    if args.mode=='fixed_defaults':
        ht=[args.ht_default_p]; rt=[(args.rt_default_delta,args.rt_default_lambda_nu,args.rt_default_lambda_c)]
        hard=[parse_sq(args.sq_default_hard)[0]]; soft=[parse_sq(args.sq_default_soft)[0]]
    elif args.mode=='budget_matched':
        counts=[len(ht),len(rt),len(hard),len(soft)]
        if len(set(counts))!=1: raise ValueError(f'budget_matched needs equal counts, got {counts}')
    for p in ht:
        e=entry('htmuon_like',f'htmuon_p{p:g}',f'p={p:g}'); e['ht_p']=float(p); E.append(e)
    for d,n,c in rt:
        e=entry('rt_fixed','rt_muon',rt_label(d,n,c)); e.update(rt_delta=d,rt_lambda_nu=n,rt_lambda_c=c); E.append(e)
    for x in hard:
        d,n,c,ld,lu,lt,thr,tau=x
        e=entry('sq_hard_v41','sq_rt_muon',sq_label(x,'hard','mean'))
        e.update(rt_delta=d,rt_lambda_nu=n,rt_lambda_c=c,sq_lambda_diff=ld,sq_lambda_uncert=lu,sq_lambda_tube=lt,sq_phi_threshold=thr,sq_soft_tau=tau,sq_quench_mode='hard',sq_diffusion_mode='mean')
        E.append(e)
    for x in soft:
        d,n,c,ld,lu,lt,thr,tau=x
        e=entry('sq_soft_v42','sq_rt_soft',sq_label(x,'soft','normalized'))
        e.update(rt_delta=d,rt_lambda_nu=n,rt_lambda_c=c,sq_lambda_diff=ld,sq_lambda_uncert=lu,sq_lambda_tube=lt,sq_phi_threshold=thr,sq_soft_tau=tau,sq_quench_mode='soft',sq_diffusion_mode='normalized')
        E.append(e)
    if args.include_sq_ablations:
        for ab in ['sq_no_diffusion','sq_no_uncertainty','sq_no_tube_penalty','sq_no_quench','sq_no_noise','sq_no_consistency']:
            for x in soft:
                d,n,c,ld,lu,lt,thr,tau=x
                e=entry(ab,ab,sq_label(x,'soft','normalized'))
                e.update(rt_delta=d,rt_lambda_nu=n,rt_lambda_c=c,sq_lambda_diff=ld,sq_lambda_uncert=lu,sq_lambda_tube=lt,sq_phi_threshold=thr,sq_soft_tau=tau,sq_quench_mode='soft',sq_diffusion_mode='normalized')
                E.append(e)
    return E


def make_tasks(args):
    beta=tuple(parse_list(args.beta_grid,float)); E=configs(args); T=[]
    for setting in parse_list(args.settings,str):
        for lr in parse_list(args.lrs,float):
            for seed in range(args.seeds):
                for e in E:
                    cfg=ToyConfig(setting=setting,n=args.n,steps=args.steps,lr=lr,momentum=args.momentum,noise=args.noise,offdiag_curv=args.offdiag_curv,seed=seed,kappa_mode=args.kappa_mode,rt_delta=e['rt_delta'],rt_lambda_nu=e['rt_lambda_nu'],rt_lambda_c=e['rt_lambda_c'],sq_lambda_diff=e['sq_lambda_diff'],sq_lambda_uncert=e['sq_lambda_uncert'],sq_lambda_tube=e['sq_lambda_tube'],sq_phi_threshold=e['sq_phi_threshold'],sq_soft_tau=e['sq_soft_tau'],sq_quench_mode=e['sq_quench_mode'],sq_diffusion_mode=e['sq_diffusion_mode'],beta_grid=beta)
                    T.append((e,cfg))
    return T,E


def run_task(t):
    e,cfg=t; rec=simulate_one(e['method'],cfg)
    for k,v in e.items():
        if k!='method': rec[k]=v
    return rec


def flat_cols(df):
    df=df.copy(); df.columns=['_'.join([str(x) for x in c if str(x)]) if isinstance(c,tuple) else str(c) for c in df.columns]; return df


def summarize(df,out,E,args):
    metrics=['auc_loss','final_loss','auc_weak_error','mean_tail_update_mass','mean_entropy','mean_beta','mean_beta_candidate','mean_soft_shrink','beta_zero_rate','accept_rate','mean_cv_gain','mean_raw_cv_gain','mean_phi','mean_diffusion_cost','mean_uncertainty_cost','mean_tube_penalty','mean_fro_ratio','mean_abs_w_minus_1','mean_min_w','mean_max_w']
    groups=['setting','family','method','algo_config','lr','ht_p','rt_delta','rt_lambda_nu','rt_lambda_c','sq_lambda_diff','sq_lambda_uncert','sq_lambda_tube','sq_phi_threshold','sq_soft_tau','sq_quench_mode','sq_diffusion_mode']
    agg=flat_cols(df.groupby(groups,dropna=False)[metrics].agg(['mean','std','count']).reset_index())
    agg.to_csv(out/'summary_by_config.csv',index=False)
    B=[]
    for metric in ['auc_loss','final_loss']:
        mc=f'{metric}_mean'
        for (setting,fam),g in agg.groupby(['setting','family']):
            r=g.sort_values(mc).iloc[0].to_dict(); r['selection_metric']=metric; B.append(r)
    best=pd.DataFrame(B); best.to_csv(out/'best_by_family_metric.csv',index=False)
    fams=['muon','htmuon_like','rt_fixed','sq_hard_v41','sq_soft_v42']
    rows=[]
    for setting in sorted(df.setting.unique()):
        for metric in ['auc_loss','final_loss']:
            mc=f'{metric}_mean'; sub=best[(best.setting==setting)&(best.selection_metric==metric)]
            row={'setting':setting,'metric':metric}
            for fam in fams:
                ss=sub[sub.family==fam]
                if len(ss):
                    r=ss.iloc[0]; row[f'{fam}_best']=float(r[mc]); row[f'{fam}_config']=str(r.algo_config); row[f'{fam}_lr']=float(r.lr)
                else:
                    row[f'{fam}_best']=float('nan'); row[f'{fam}_config']=''; row[f'{fam}_lr']=float('nan')
            row['sq_soft_beats_muon']=row.get('sq_soft_v42_best',float('nan'))<row.get('muon_best',float('nan'))
            row['sq_soft_beats_ht']=row.get('sq_soft_v42_best',float('nan'))<row.get('htmuon_like_best',float('nan'))
            row['sq_soft_beats_sq_hard']=row.get('sq_soft_v42_best',float('nan'))<row.get('sq_hard_v41_best',float('nan'))
            rows.append(row)
    pd.DataFrame(rows).to_csv(out/'oracle_comparison.csv',index=False)
    # positive-p HT comparison
    pos=[]
    for setting in sorted(df.setting.unique()):
        for metric in ['auc_loss','final_loss']:
            mc=f'{metric}_mean'; sub=best[(best.setting==setting)&(best.selection_metric==metric)]
            ht=sub[(sub.family=='htmuon_like')&(sub.ht_p.fillna(0)>0)].sort_values(mc)
            sq=sub[sub.family=='sq_soft_v42']
            row={'setting':setting,'metric':metric}
            if len(ht): row.update(best_positive_ht=float(ht.iloc[0][mc]), best_positive_ht_config=str(ht.iloc[0].algo_config), best_positive_ht_lr=float(ht.iloc[0].lr))
            else: row.update(best_positive_ht=float('nan'), best_positive_ht_config='', best_positive_ht_lr=float('nan'))
            if len(sq): row.update(sq_soft_best=float(sq.iloc[0][mc]), sq_soft_config=str(sq.iloc[0].algo_config), sq_soft_lr=float(sq.iloc[0].lr))
            else: row.update(sq_soft_best=float('nan'), sq_soft_config='', sq_soft_lr=float('nan'))
            row['sq_soft_beats_positive_ht']=row['sq_soft_best']<row['best_positive_ht']; pos.append(row)
    pd.DataFrame(pos).to_csv(out/'positive_ht_comparison.csv',index=False)
    # muon-quality constrained AUC
    cons=[]
    for setting in sorted(agg.setting.unique()):
        sa=agg[agg.setting==setting]; mu=sa[sa.family=='muon']
        if not len(mu): continue
        mf=float(mu.final_loss_mean.min())
        for eps in parse_list(args.quality_eps_grid,float):
            thr=(1+eps)*mf
            for fam,g in sa.groupby('family'):
                fg=g[g.final_loss_mean<=thr]
                row={'setting':setting,'family':fam,'eps':eps,'muon_best_final':mf,'threshold':thr,'feasible':bool(len(fg))}
                if len(fg):
                    r=fg.sort_values('auc_loss_mean').iloc[0]; row.update(best_auc_under_muon_quality=float(r.auc_loss_mean),config=str(r.algo_config),lr=float(r.lr))
                else: row.update(best_auc_under_muon_quality=float('nan'),config='',lr=float('nan'))
                cons.append(row)
    pd.DataFrame(cons).to_csv(out/'muon_quality_constrained_auc.csv',index=False)
    # per-lr scorecard
    lrrows=[]
    for setting in sorted(agg.setting.unique()):
        for lr in sorted(agg[agg.setting==setting].lr.unique()):
            sl=agg[(agg.setting==setting)&(agg.lr==lr)]
            for metric in ['auc_loss','final_loss']:
                mc=f'{metric}_mean'; row={'setting':setting,'lr':float(lr),'metric':metric}
                for fam,g in sl.groupby('family'):
                    r=g.sort_values(mc).iloc[0]; row[f'{fam}_best']=float(r[mc]); row[f'{fam}_config']=str(r.algo_config)
                sq=row.get('sq_soft_v42_best',float('nan'))
                row['sq_soft_beats_muon']=sq<row.get('muon_best',float('nan'))
                row['sq_soft_beats_ht']=sq<row.get('htmuon_like_best',float('nan'))
                row['sq_soft_beats_sq_hard']=sq<row.get('sq_hard_v41_best',float('nan'))
                row['sq_soft_beats_rt_fixed']=sq<row.get('rt_fixed_best',float('nan'))
                lrrows.append(row)
    pd.DataFrame(lrrows).to_csv(out/'lr_regime_scorecard.csv',index=False)
    # budget
    lrs=parse_list(args.lrs,float); settings=parse_list(args.settings,str)
    br=[]
    for fam,g in pd.DataFrame(E).groupby('family'):
        ncfg=g[['method','algo_config']].drop_duplicates().shape[0]
        br.append(dict(family=fam,num_algorithm_configs=int(ncfg),num_lrs=len(lrs),num_settings=len(settings),num_seeds=args.seeds,total_runs=int(ncfg*len(lrs)*len(settings)*args.seeds)))
    pd.DataFrame(br).to_csv(out/'search_budget.csv',index=False)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--outdir',default='results/sq_rt_v42'); p.add_argument('--mode',choices=['budget_matched','fixed_defaults','smallfast'],default='budget_matched')
    p.add_argument('--settings',default='fullrank,lowrank,mixed,spiked_noise'); p.add_argument('--lrs',default='0.01,0.02,0.05,0.1,0.2,0.3')
    p.add_argument('--quality-eps-grid',default='0,0.005,0.01,0.02')
    p.add_argument('--ht-p-grid',default='0,0.03125,0.0625,0.1,0.125,0.1875,0.25,0.375,0.5'); p.add_argument('--ht-default-p',type=float,default=0.125)
    p.add_argument('--rt-fixed-grid',default='0.2:1.0:0.0,0.2:2.0:0.25,0.35:1.0:0.25,0.35:2.0:0.25,0.35:4.0:0.5,0.5:2.0:0.25,0.5:4.0:0.5,0.2:1.0:0.25,0.5:1.0:0.25')
    p.add_argument('--sq-hard-grid',default='0.5:4.0:0.5:0.25:0.25:0.02:0.0:0.001,0.5:4.0:0.5:0.25:0.25:0.02:0.001:0.001,0.5:4.0:0.5:0.5:1.0:0.10:0.005:0.001,0.5:4.0:0.5:1.0:2.0:0.20:0.01:0.001,0.5:4.0:0.5:1.0:2.0:0.20:0.02:0.001,0.35:2.0:0.25:0.5:0.5:0.05:0.005:0.001,0.35:2.0:0.25:1.0:1.0:0.10:0.02:0.001,0.2:1.0:0.0:0.5:0.5:0.05:0.005:0.001,0.2:1.0:0.0:1.0:1.0:0.10:0.02:0.001')
    p.add_argument('--sq-soft-grid',default='0.5:4.0:0.0:0.00:0.25:0.02:0.0000:0.001,0.5:4.0:0.0:0.02:0.25:0.02:0.0005:0.001,0.5:4.0:0.0:0.05:0.50:0.05:0.0010:0.001,0.5:4.0:0.0:0.10:0.50:0.05:0.0020:0.001,0.35:2.0:0.0:0.02:0.25:0.02:0.0005:0.001,0.35:2.0:0.0:0.05:0.50:0.05:0.0010:0.001,0.35:2.0:0.0:0.10:1.00:0.10:0.0020:0.002,0.2:1.0:0.0:0.02:0.25:0.02:0.0005:0.001,0.2:1.0:0.0:0.05:0.50:0.05:0.0010:0.002')
    p.add_argument('--sq-default-hard',default='0.35:2.0:0.25:0.5:0.5:0.05:0.0:0.001'); p.add_argument('--sq-default-soft',default='0.35:2.0:0.0:0.05:0.5:0.05:0.001:0.001')
    p.add_argument('--rt-default-delta',type=float,default=0.35); p.add_argument('--rt-default-lambda-nu',type=float,default=2.0); p.add_argument('--rt-default-lambda-c',type=float,default=0.25)
    p.add_argument('--beta-grid',default='0,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.25,0.5,1,2,4')
    p.add_argument('--include-muon',action='store_true',default=True); p.add_argument('--no-muon',dest='include_muon',action='store_false'); p.add_argument('--include-sq-ablations',action='store_true')
    p.add_argument('--seeds',type=int,default=16); p.add_argument('--n',type=int,default=32); p.add_argument('--steps',type=int,default=200); p.add_argument('--noise',type=float,default=0.03); p.add_argument('--momentum',type=float,default=0.9); p.add_argument('--offdiag-curv',type=float,default=0.05); p.add_argument('--kappa-mode',choices=['true','ones'],default='true'); p.add_argument('--workers',type=int,default=0)
    args=p.parse_args()
    if args.mode=='smallfast':
        args.seeds=min(args.seeds,2); args.steps=min(args.steps,40); args.settings='fullrank,mixed'; args.lrs='0.02,0.1'
        args.ht_p_grid=','.join(args.ht_p_grid.split(',')[:2]); args.rt_fixed_grid=','.join(args.rt_fixed_grid.split(',')[:2]); args.sq_hard_grid=','.join(args.sq_hard_grid.split(',')[:2]); args.sq_soft_grid=','.join(args.sq_soft_grid.split(',')[:2]); args.mode='budget_matched'
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True); W=workers(args.workers); T,E=make_tasks(args)
    with open(out/'run_args.json','w') as f:
        d=vars(args).copy(); d.update(workers_resolved=W,num_tasks=len(T),num_method_configs=len(E)); json.dump(d,f,indent=2,ensure_ascii=False)
    pd.DataFrame(E).to_csv(out/'method_configs.csv',index=False)
    print(f'[v4.2] tasks={len(T)} workers={W} outdir={out}',flush=True)
    rec=[]
    if W==1:
        for t in T: rec.append(run_task(t))
    else:
        with ProcessPoolExecutor(max_workers=W) as ex:
            futs=[ex.submit(run_task,t) for t in T]
            for i,f in enumerate(as_completed(futs),1):
                rec.append(f.result())
                if i%max(1,len(T)//10)==0: print(f'  completed {i}/{len(T)}',flush=True)
    df=pd.DataFrame(rec); df.to_csv(out/'records.csv',index=False); summarize(df,out,E,args); print('[v4.2] done',flush=True)

if __name__=='__main__': main()
