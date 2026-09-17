"""Strong simple controllers under the manuscript's original budget."""
from __future__ import annotations
import json
import os
from pathlib import Path
import platform
import sys
import time
import numpy as np
import pandas as pd

from experiments.paths import DATA, RESULTS, CONFIGS
OUT = RESULTS / "controllers"


def bootstrap(values, seed):
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(10000, len(values)))
    return np.quantile(np.asarray(values)[samples].mean(axis=1), [.025, .975])


EPS=1e-8;TOL=1e-12
BETAS=(.2,.4,.6,.8)
METHODS=('fixed','halving','coarse_0p05','bisection8','bisection12','bisection28')
REPEATS=3

def tv_search(x):
    a=np.asarray(x,np.float64)
    gx=np.zeros_like(a);gy=np.zeros_like(a)
    gx[:,:-1]=np.diff(a,axis=1);gy[:-1]=np.diff(a,axis=0)
    g=np.stack((gx,gy),axis=-1)
    return float(np.sqrt(np.sum(g*g,axis=(2,3))+EPS**2).mean())

def tv_audit(x):
    # Independent accumulation: no call to the search gradient implementation.
    a=np.asarray(x,np.float64)
    energy=np.full(a.shape[:2],EPS**2,dtype=np.float64)
    energy[:,:-1]+=np.sum((a[:,1:]-a[:,:-1])**2,axis=2)
    energy[:-1]+=np.sum((a[1:]-a[:-1])**2,axis=2)
    return float(np.sqrt(energy).mean())

def psnr(x,u):
    return float(-10*np.log10(np.mean((np.asarray(x,np.float64)-np.asarray(u,np.float64))**2)))

def control(f,v,beta,method,fixed_eta=None):
    initial=tv_search(f);budget=beta*(initial-EPS);count=1
    delta=v-f
    def feasible(eta):
        nonlocal count
        u=f+eta*delta;val=tv_search(u);count+=1
        return max(0.,initial-val)<=budget+TOL,u
    if method=='fixed':
        eta=float(fixed_eta);u=f+eta*delta;active=None
    else:
        proposal_tv=tv_search(v);count+=1
        active=max(0.,initial-proposal_tv)>budget+TOL
        if not active:eta=1.;u=v
        elif method=='halving':
            eta=.5
            while True:
                ok,u=feasible(eta)
                if ok:break
                eta*=.5
        elif method=='coarse_0p05':
            for index in range(19,-1,-1):
                eta=index/20
                ok,u=feasible(eta)
                if ok:break
        else:
            iterations=int(method.removeprefix('bisection'))
            lo,hi=0.,1.;u=f
            for _ in range(iterations):
                mid=(lo+hi)/2;ok,candidate=feasible(mid)
                if ok:lo=mid;u=candidate
                else:hi=mid
            eta=lo
    audit_initial=tv_audit(f);output_tv=tv_audit(u)
    audit_budget=beta*(audit_initial-EPS)
    residual=max(0.,audit_initial-output_tv)-audit_budget
    if method!='fixed':assert residual<=TOL,(method,beta,eta,residual)
    return u,dict(eta=eta,budget=audit_budget,residual=residual,violation=residual>TOL,
                  search_tv_evaluations=count,audit_tv_evaluations=2,
                  proposal_active=active,output_tv=output_tv)

def load_case(cache,image):
    return tuple(np.asarray(cache[f'{image}_{key}'],np.float32) for key in ('clean','noisy','proposal'))

def select_fixed(cases,tag):
    etas=np.arange(201)/200;quality=[];removed=[]
    for image,(x,f,v) in cases:
        initial=tv_audit(f);loss=[];scores=[]
        for eta in etas:
            u=f+float(eta)*(v-f)
            loss.append(max(0.,initial-tv_audit(u))/(initial-EPS))
            scores.append(psnr(x,u))
        removed.append(loss);quality.append(scores)
        print('validation grid',tag,image,flush=True)
    loss=np.asarray(removed);quality=np.asarray(quality)
    cfg={};candidates=[]
    for beta in BETAS:
        # Convert the absolute numerical tolerance to each observation's scale.
        valid=np.ones(len(etas),bool)
        for i,(_,(_,f,_)) in enumerate(cases):
            valid &= loss[i] <= beta+TOL/(tv_audit(f)-EPS)
        eligible=np.flatnonzero(valid)
        j=int(eligible[np.argmax(quality[:,eligible].mean(axis=0))])
        cfg[str(beta)]=dict(eta=float(etas[j]),validation_psnr=float(quality[:,j].mean()),
                            validation_max_removal_fraction=float(loss[:,j].max()))
    for j,eta in enumerate(etas):
        candidates.append(dict(dataset=tag,eta=eta,mean_validation_psnr=quality[:,j].mean(),
                               max_validation_removal_fraction=loss[:,j].max()))
    return cfg,candidates

def summarize(frame):
    rows=[];comparisons=[]
    for keys,g in frame.groupby(['dataset','beta','method'],sort=True):
        rows.append(dict(zip(['dataset','beta','method'],keys))|dict(n=len(g),
          psnr=g.psnr.mean(),psnr_vs_proposal=g.psnr_vs_proposal.mean(),median_eta=g.eta.median(),
          violations=int(g.violation.sum()),violation_percent=100*g.violation.mean(),
          max_budget_residual=g.residual.max(),mean_time_ms=1000*g.time_seconds.mean(),
          median_time_ms=1000*g.time_seconds.median(),mean_search_tv_evaluations=g.search_tv_evaluations.mean(),
          mean_audit_tv_evaluations=g.audit_tv_evaluations.mean()))
    for (tag,beta),g in frame.groupby(['dataset','beta']):
        target=g[g.method=='bisection28']
        for method in METHODS:
            if method=='bisection28':continue
            other=g[g.method==method]
            p=target.merge(other,on='image',suffixes=('_28','_other'),validate='one_to_one')
            delta=(p.psnr_28-p.psnr_other).to_numpy();lo,hi=bootstrap(delta,20260912)
            comparisons.append(dict(dataset=tag,beta=beta,comparison=method,n=len(p),
              psnr_28_minus_other=delta.mean(),ci_low=lo,ci_high=hi,
              max_absolute_psnr_difference=np.max(np.abs(delta)),
              mean_eta_gap=(p.eta_28-p.eta_other).mean(),max_eta_gap=(p.eta_28-p.eta_other).max(),
              mean_runtime_ratio_28_to_other=p.time_seconds_28.mean()/p.time_seconds_other.mean(),
              percent_abs_psnr_difference_le_0p01=100*np.mean(np.abs(delta)<=.01)))
    pd.DataFrame(rows).to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'paired_comparisons.csv',index=False)

def main():
    allrows=[];selections={};allcandidates=[];repeated=[];checks=[]
    rng=np.random.default_rng(20260912);started=time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    for tag in ['sigma25','polyu','sigma15','sigma50']:
        mn='icassp_polyu_budget_manifest.json' if tag=='polyu' else 'icassp_budget_manifest.json'
        manifest=json.loads((CONFIGS/mn).read_text())
        cn='icassp_polyu_drunet.npz' if tag=='polyu' else f'icassp_drunet_{tag}.npz'
        with np.load(DATA/'cache'/cn,allow_pickle=False) as cache:
            validation=[(name,load_case(cache,name)) for name in manifest['validation_images']]
            cfg,candidates=select_fixed(validation,tag);selections[tag]=cfg;allcandidates+=candidates
            (OUT/'selected_fixed.json').write_text(json.dumps(selections,indent=2)+'\n')
            pd.DataFrame(allcandidates).to_csv(OUT/'validation_candidates.csv',index=False)
            print('FROZEN',tag,json.dumps(cfg),flush=True)
            _,warm_f,warm_v=validation[0][1]
            for method in METHODS:control(warm_f,warm_v,.6,method,cfg['0.6']['eta'])
            del validation
            for image in manifest['test_images']:
                x,f,v=load_case(cache,image);proposal_psnr=psnr(x,v);input_psnr=psnr(x,f)
                initial=tv_audit(f);proposal_tv=tv_audit(v)
                checkdiff=max(abs(tv_search(f)-initial),abs(tv_search(v)-proposal_tv))
                checks.append(dict(dataset=tag,image=image,height=f.shape[0],width=f.shape[1],
                                   search_audit_tv_difference=checkdiff))
                assert checkdiff<1e-12
                for beta in BETAS:
                    fixed=cfg[str(beta)]['eta'];last={};timings={m:[] for m in METHODS}
                    for rep in range(REPEATS):
                        for method in rng.permutation(METHODS):
                            tick=time.perf_counter();u,metrics=control(f,v,beta,str(method),fixed)
                            elapsed=time.perf_counter()-tick
                            if method in last:assert last[method][1]['eta']==metrics['eta']
                            last[method]=(u,metrics);timings[method].append(elapsed)
                            repeated.append(dict(dataset=tag,image=image,beta=beta,method=method,repeat=rep,seconds=elapsed))
                    for method in METHODS:
                        u,metrics=last[method];quality=psnr(x,u)
                        allrows.append(dict(dataset=tag,image=image,beta=beta,method=method,psnr=quality,
                            psnr_vs_proposal=quality-proposal_psnr,proposal_psnr=proposal_psnr,input_psnr=input_psnr,
                            time_seconds=float(np.median(timings[method])),time_min=min(timings[method]),time_max=max(timings[method]))|metrics)
                    for method,u,quality in [('proposal',v,proposal_psnr),('noisy',f,input_psnr)]:
                        output_tv=proposal_tv if method=='proposal' else initial
                        residual=max(0.,initial-output_tv)-beta*(initial-EPS)
                        allrows.append(dict(dataset=tag,image=image,beta=beta,method=method,psnr=quality,
                            psnr_vs_proposal=quality-proposal_psnr,proposal_psnr=proposal_psnr,input_psnr=input_psnr,
                            eta=float(method=='proposal'),budget=beta*(initial-EPS),residual=residual,
                            violation=residual>TOL,time_seconds=np.nan,search_tv_evaluations=np.nan,audit_tv_evaluations=np.nan))
                pd.DataFrame(allrows).to_csv(OUT/'per_image.csv',index=False)
                pd.DataFrame(repeated).to_csv(OUT/'timing_repeats.csv',index=False)
                pd.DataFrame(checks).to_csv(OUT/'evaluator_checks.csv',index=False)
                summarize(pd.DataFrame(allrows))
                print('tested',tag,image,f'total {time.perf_counter()-started:.1f}s',flush=True)
    hardware=dict(platform=platform.platform(),numpy=np.__version__,python=platform.python_version(),
                  cpu_count=os.cpu_count(),threads={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']},
                  repeats=REPEATS,timing='median of three randomized-order calls per image/budget/controller; includes independent audit; excludes inference and scoring')
    (OUT/'runtime_protocol.json').write_text(json.dumps(hardware,indent=2)+'\n')
    print(pd.read_csv(OUT/'summary.csv').query("dataset in ['sigma25','polyu']").to_string(index=False),flush=True)

if __name__=='__main__':main()
