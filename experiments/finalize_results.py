from pathlib import Path
import json
import numpy as np
import pandas as pd
from experiments.run_controller_comparison import OUT,METHODS,summarize,bootstrap

f=pd.read_csv(OUT/'per_image.csv')
summarize(f)
s=pd.read_csv(OUT/'summary.csv')
extra=[]
for row in s.itertuples():
    group=f[(f.dataset==row.dataset)&(f.beta==row.beta)&(f.method==row.method)]
    proposal=f[(f.dataset==row.dataset)&(f.beta==row.beta)&(f.method=='proposal')][['image','violation']].rename(columns={'violation':'proposal_infeasible'})
    paired=group.merge(proposal,on='image',validate='one_to_one')
    extra.append(dict(dataset=row.dataset,beta=row.beta,method=row.method,
        passthrough_percent=100*np.mean(group.eta==1),proposal_active_percent=100*paired.proposal_infeasible.mean(),
        active_mean_time_ms=1000*paired.loc[paired.proposal_infeasible,'time_seconds'].mean(),
        inactive_mean_time_ms=1000*paired.loc[~paired.proposal_infeasible,'time_seconds'].mean()))
s=s.merge(pd.DataFrame(extra),on=['dataset','beta','method'],validate='one_to_one')
s.to_csv(OUT/'summary.csv',index=False)

precision=[]
for tag,g in f.groupby('dataset'):
    reference=g[g.method=='bisection28']
    for method in METHODS:
        if method=='bisection28':continue
        a=g[g.method==method].merge(reference,on=['image','beta'],suffixes=('_other','_28'),validate='one_to_one')
        # Budgets are repeated conditions, not independent bootstrap units.
        precision.append(dict(dataset=tag,method=method,n_conditions=len(a),n_images=a.image.nunique(),
            mean_signed_psnr_28_minus_other=(a.psnr_28-a.psnr_other).mean(),
            mean_absolute_psnr_difference=(a.psnr_28-a.psnr_other).abs().mean(),
            max_absolute_psnr_difference=(a.psnr_28-a.psnr_other).abs().max(),
            fraction_abs_difference_le_0p01=np.mean((a.psnr_28-a.psnr_other).abs()<=.01),
            max_eta_gap=(a.eta_28-a.eta_other).max(),
            runtime_ratio_28_to_other=a.time_seconds_28.mean()/a.time_seconds_other.mean()))
pd.DataFrame(precision).to_csv(OUT/'precision_summary.csv',index=False)

additional=[]
for (tag,beta),g in f.groupby(['dataset','beta']):
    for method in ('halving','coarse_0p05','fixed'):
        a=g[g.method=='bisection12'].merge(g[g.method==method],on='image',suffixes=('_12','_other'),validate='one_to_one')
        d=(a.psnr_12-a.psnr_other).to_numpy();lo,hi=bootstrap(d,20260912)
        additional.append(dict(dataset=tag,beta=beta,comparison=method,n_images=len(a),psnr_gain_bisection12=d.mean(),
                               ci_low=lo,ci_high=hi,runtime_ratio_12_to_other=a.time_seconds_12.mean()/a.time_seconds_other.mean()))
pd.DataFrame(additional).to_csv(OUT/'bisection12_comparisons.csv',index=False)

controller=f[f.method.isin(METHODS)]
adaptive=controller[controller.method!='fixed']
assert len(controller)==69*4*6
assert not adaptive.violation.any()
for method,k in [('bisection8',8),('bisection12',12)]:
    a=f[f.method=='bisection28'].merge(f[f.method==method],on=['dataset','image','beta'],suffixes=('_28','_short'),validate='one_to_one')
    assert (a.eta_28-a.eta_short>=-1e-14).all()
    assert (a.eta_28-a.eta_short<=2**(-k)+1e-12).all()
for row in pd.DataFrame(precision).query("method=='coarse_0p05'").itertuples():
    assert row.max_eta_gap<=.05+1e-6
fixed_known={'sigma25':{.2:.210,.4:.420,.6:.640,.8:.865},'polyu':{.2:.235,.4:.480,.6:.750,.8:.910}}
selected=json.loads((OUT/'selected_fixed.json').read_text())
for tag,bybeta in fixed_known.items():
    for beta,eta in bybeta.items():assert selected[tag][str(beta)]['eta']==eta
paper_examples={('sigma25',.6):(28.35,27.56),('polyu',.2):(37.13,36.60)}
for (tag,beta),(adaptive_psnr,fixed_psnr) in paper_examples.items():
    g=s[(s.dataset==tag)&(s.beta==beta)].set_index('method')
    assert abs(g.loc['bisection28','psnr']-adaptive_psnr)<.015
    assert abs(g.loc['fixed','psnr']-fixed_psnr)<.015
checks=pd.read_csv(OUT/'evaluator_checks.csv')
result=dict(test_conditions=int(len(controller)/len(METHODS)),controller_outputs=len(controller),
    adaptive_outputs=len(adaptive),adaptive_violations=int(adaptive.violation.sum()),
    fixed_violations=int(controller[controller.method=='fixed'].violation.sum()),
    max_adaptive_residual=float(adaptive.residual.max()),
    search_audit_max_difference=float(checks.search_audit_tv_difference.max()),
    coefficient_accuracy_bounds_passed=True,paper_fixed_parameters_reproduced=True,
    paper_psnr_examples_reproduced=True,shapes=checks.groupby('dataset')[['height','width']].first().to_dict(orient='index'))
(OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
print(pd.DataFrame(precision).query("method in ['bisection8','bisection12']").to_string(index=False))
