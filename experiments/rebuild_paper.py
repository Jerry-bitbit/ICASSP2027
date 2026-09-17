"""Generate final-size figures/tables directly from verified existing rows/arrays."""
from pathlib import Path
import argparse
import json,sys
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from experiments.paths import ROOT, OUTPUT
from experiments.verify_results import verify
parser = argparse.ArgumentParser(description="Rebuild Tables 1-3 and Figures 1-5 for the revised manuscript.")
parser.add_argument("--results", type=Path, default=ROOT / "results")
parser.add_argument("--output", type=Path, default=OUTPUT / "paper")
parser.add_argument("--weak-results", type=Path, default=None, help="Weak-structure reference records; defaults to <results>/weak_structure")
args = parser.parse_args()
R = args.results.resolve()
C = R / "controllers"
P = args.output.resolve()
verify(R)
(P / "tables").mkdir(parents=True, exist_ok=True)
(P / "figures").mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.2,'axes.labelsize':9.2,'axes.titlesize':9.2,'legend.fontsize':9.2,'xtick.labelsize':9.2,'ytick.labelsize':9.2,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'lines.linewidth':1.3,'pdf.fonttype':42,'ps.fonttype':42})
colors=['#0072B2','#D55E00','#009E73','#69448E']
lines=[r'\begin{table*}[t]',r'\centering',r'\caption{Test results with DRUNet proposals. $\widetilde\eta$ is median adaptive retention. PSNR (dB) and SSIM are per-image means; $\Delta$PSNR is adaptive minus fixed, with paired 95\% intervals. Both methods have zero test violations.}',r'\label{tab:main}',r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccc@{}}',r'\toprule',r'&&\multicolumn{3}{c}{Adaptive blend}&\multicolumn{2}{c}{Fixed blend}&\\',r'\cmidrule(lr){3-5}\cmidrule(lr){6-7}',r'Dataset&$\beta$&$\widetilde\eta$&PSNR&SSIM&$\eta$&PSNR / SSIM&$\Delta$PSNR [95\% CI]\\',r'\midrule']
for dataset,prefix in [('Kodak','icassp_budget'),('PolyU','icassp_polyu_budget')]:
    d=pd.read_csv(R/f'{prefix}_summary.csv')
    for i,row in enumerate(d[d.beta.isin([.2,.4,.6,.8])].itertuples()):
        lines.append(f'{dataset if i==0 else ""}&{row.beta:.1f}&{row.adaptive_eta_median:.3f}&{row.adaptive_psnr:.2f}&{row.adaptive_ssim:.3f}&{row.fixed_eta:.3f}&{row.fixed_psnr:.2f} / {row.fixed_ssim:.3f}&${row.delta_psnr:.2f}\\,[{row.delta_psnr_ci_low:.2f},{row.delta_psnr_ci_high:.2f}]$'+r'\\')
    if dataset=='Kodak':lines.append(r'\midrule')
lines.extend([r'\bottomrule',r'\end{tabular*}',r'\end{table*}'])
(P/'tables/main.tex').write_text('\n'.join(lines)+'\n')
noise=[r'\begin{table}[t]',r'\centering',r'\caption{Kodak noise sensitivity. $\Delta$ is adaptive-minus-fixed PSNR (dB). The final column is the maximum fixed violation rate over all four main budgets.}',r'\label{tab:noise}',r'\setlength{\tabcolsep}{2pt}',r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}cccccc@{}}',r'\toprule',r'&\multicolumn{2}{c}{$\beta=0.2$}&\multicolumn{2}{c}{$\beta=0.6$}&Max. fixed\\',r'$255\sigma$&$\widetilde\eta$&$\Delta$&$\widetilde\eta$&$\Delta$&viol. (\%)\\',r'\midrule']
for sig in [15,25,50]:
    d=pd.read_csv(R/('icassp_budget_summary.csv' if sig==25 else f'icassp_kodak_sigma{sig}_summary.csv')).set_index('beta')
    a,b=d.loc[.2],d.loc[.6];vi=d.loc[[.2,.4,.6,.8]].fixed_violation_percent.max()
    noise.append(f'{sig}&{a.adaptive_eta_median:.3f}&{a.delta_psnr:.2f}&{b.adaptive_eta_median:.3f}&{b.delta_psnr:.2f}&{vi:.2f}'+r'\\')
noise.extend([r'\bottomrule',r'\end{tabular*}',r'\end{table}']);(P/'tables/noise.tex').write_text('\n'.join(noise)+'\n')
prop=[r'\begin{table}[t]',r'\centering',r'\caption{Kodak proposal sensitivity at $\sigma=25/255$. Adaptive violations are zero. DRUNet results are in Table~\ref{tab:main}.}',r'\label{tab:proposal}',r'\setlength{\tabcolsep}{2pt}',r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lcccc@{}}',r'\toprule',r'Proposal&$\beta$&$\widetilde\eta$&PSNR&$\Delta$PSNR [95\% CI]\\',r'\midrule']
d=pd.read_csv(R/'icassp_proposal_summary.csv')
for row in d[d.proposal!='drunet'].itertuples():
    name={'wavelet':'Wavelet','nlm':'NLM'}[row.proposal]
    prop.append(f'{name}&{row.beta:.1f}&{row.adaptive_eta_median:.3f}&{row.adaptive_psnr:.2f}&${row.delta_psnr:.2f}\\,[{row.ci_low:.2f},{row.ci_high:.2f}]$'+r'\\')
prop.extend([r'\bottomrule',r'\end{tabular*}',r'\end{table}']);(P/'tables/proposal.tex').write_text('\n'.join(prop)+'\n')

fig,ax=plt.subplots(figsize=(86/25.4,1.95))
fig.subplots_adjust(left=.19,right=.97,bottom=.25,top=.82)
d=pd.read_csv(C/'paired_comparisons.csv')
for tag,label,color,mark in [('sigma25','Kodak',colors[0],'o'),('polyu','PolyU',colors[1],'^')]:
    g=d[(d.dataset==tag)&(d.comparison=='coarse_0p05')].sort_values('beta')
    ax.errorbar(g.beta,g.psnr_28_minus_other,yerr=[g.psnr_28_minus_other-g.ci_low,g.ci_high-g.psnr_28_minus_other],color=color,marker=mark,capsize=3,label=label)
ax.axhline(0,color='gray',lw=.7);ax.set_xticks([.2,.4,.6,.8]);ax.set_yticks([0,.2,.4]);ax.set(xlabel=r'Relative budget $\beta$',ylabel='PSNR gain (dB)',ylim=(-.045,.52));ax.grid(axis='y',alpha=.25)
fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.56,1.02))
fig.savefig(P/'figures/controllers.pdf');plt.close(fig)

fig,ax=plt.subplots(figsize=(86/25.4,2.12));fig.subplots_adjust(left=.21,right=.97,bottom=.27,top=.72)
d=pd.read_csv(C/'precision_summary.csv')
d['relative_time']=1.0/d.runtime_ratio_28_to_other
d['max_abs_psnr']=d.max_absolute_psnr_difference
labels=[('sigma15','Kodak, 15'),('sigma25','Kodak, 25'),('sigma50','Kodak, 50'),('polyu','PolyU')]
for (tag,label),color in zip(labels,colors):
    g=d[d.dataset==tag].set_index('method').loc[['bisection8','bisection12']]
    ax.plot(g.relative_time,g.max_abs_psnr,color=color)
    for method,mark in [('bisection8','o'),('bisection12','^')]:
        row=g.loc[method];ax.plot(row.relative_time,row.max_abs_psnr,marker=mark,color=color,mfc='white' if mark=='o' else color)
ax.set_yscale('log');ax.axhline(.01,color='gray',ls='--',lw=.8)
ax.set(xlabel='Mean control time / 28-step time',ylabel='Max. PSNR difference (dB)',ylim=(.0008,.1),xlim=(.34,.59))
ax.set_xticks([.35,.45,.55]);ax.set_yticks([.001,.01,.1],['0.001','0.01','0.1']);ax.minorticks_off()
handles=[Line2D([0],[0],color=c,label=l) for (_,l),c in zip(labels,colors)]
handles +=[Line2D([0],[0],marker=m,mfc='white' if m=='o' else '#333333',color='#333333',ls='none',label=l) for m,l in [('o','8 steps'),('^','12 steps')]]
fig.legend(handles=handles,ncol=2,loc='upper center',bbox_to_anchor=(.55,1.02),frameon=False,columnspacing=1,handlelength=1.2)
fig.savefig(P/'figures/precision.pdf');plt.close(fig)

# Exact two-pixel RGB example exposes why a feasible endpoint must be tested first.
eta=np.linspace(0,1,1001);eps=1e-8
vf=(np.sqrt(3*.8**2+eps**2)+eps)/2
fig,ax=plt.subplots(figsize=(86/25.4,1.75));fig.subplots_adjust(left=.18,right=.93,bottom=.30,top=.76)
for d0,slope,label,color,style in [(.8,-1.6,'Feasible proposal',colors[0],'-'),(.8,-1.2,'Infeasible proposal',colors[1],'--')]:
 y=(np.sqrt(3*(d0+slope*eta)**2+eps**2)+eps)/(2*vf)
 ax.plot(eta,y,label=label,color=color,ls=style)
ax.axhline(.8+.2*eps/vf,color='#333333',lw=.8,ls=':');ax.text(.52,.85,r'TV floor, $\beta=0.2$',fontsize=9.2)
ax.set(xlabel=r'Proposal weight $\eta$',ylabel=r'$g(\eta)/V_\epsilon(f)$',xlim=(0,1),ylim=(0,1.06));ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1])
fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.58,1.03),frameon=False,ncol=1,labelspacing=.15)
fig.savefig(P/'figures/geometry.pdf');plt.close(fig)
from experiments.paper_assets.overview import build as build_overview
from experiments.paper_assets.visual import build as build_visual
from experiments.weak_structure.paper import build as build_weak
build_overview(R, P)
build_visual(R, P)
build_weak(args.weak_results.resolve() if args.weak_results else R / 'weak_structure', P)
from experiments.paper_assets.latex import write_fragments
write_fragments(P)
print(f'Generated revised Tables 1-3 and Figures 1-5 in {P}; controllers.pdf is an additional diagnostic.')
