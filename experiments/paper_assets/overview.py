"""Technical overview: original arrays, TV operators, search state, numerical audit."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch,FancyArrowPatch,Polygon

from .font_layout import save_figure
from .reference_tv import tv_reference,output_audit
from methods.ectv import certify_candidate, tv_energy_float64


def build(records, output):
    records=Path(records)
    P=Path(output)
    P.mkdir(parents=True,exist_ok=True)
    z=np.load(records/'figure_inputs/overview_sample.npz', allow_pickle=False)
    f=z['kodim23_noisy'];v=z['kodim23_proposal'];beta=.6
    d=pd.read_csv(records/'figure_inputs/overview_row.csv',float_precision='round_trip')
    eta=float(d[(d.image=='kodim23')&(d.method=='adaptive')&(d.beta==beta)].eta.iloc[0])
    returned=certify_candidate(f,v,budget_override=beta*(tv_energy_float64(f)-1e-8),bisection_steps=28,feasibility_tolerance=1e-12)
    np.testing.assert_allclose(returned.eta,eta,rtol=0,atol=1e-12)
    eta=returned.eta
    u=returned.image
    (P/'arrays').mkdir(parents=True,exist_ok=True)
    np.savez_compressed(P/'arrays/overview_returned.npz',output=u)
    with np.load(P/'arrays/overview_returned.npz',allow_pickle=False) as saved:
        u=saved['output']
    audit=output_audit(f,v,u,beta)
    assert audit['accepted_under_tau'] and not output_audit(f,v,v,beta)['accepted_under_tau']
    
    ink='#1D2938';gray='#79818B';light='#D5D8DC';blue='#28668C';teal='#087F79';red='#BD5C59';orange='#C77423'
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9.2,'mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42})
    fig=plt.figure(figsize=(178/25.4,110/25.4),facecolor='white')
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,178),ylim=(0,110));ax.axis('off')
    text_records=[]
    def txt(x,y,s,size=9.2,color=ink,weight='normal',ha='center',va='center',**kw):
        t=ax.text(x,y,s,fontsize=size,color=color,fontweight=weight,ha=ha,va=va,**kw);text_records.append(t);return t
    def line(points,color=gray,lw=.8,**kw):
        xs,ys=zip(*points);ax.plot(xs,ys,color=color,lw=lw,clip_on=False,**kw)
    def arr(a,b,color=ink,lw=.9,**kw):
        q=FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,linewidth=lw,color=color,shrinkA=0,shrinkB=0,**kw);ax.add_patch(q)
    def route(points,color=ink,lw=.9,**kw):
        line(points[:-1],color,lw,**kw) if len(points)>2 else None
        arr(points[-2],points[-1],color,lw)
    def box(x,y,w,h,fill='#F3F4F6',edge=gray,rounding=.8,lw=.7):
        b=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={rounding}',fc=fill,ec=edge,lw=lw);ax.add_patch(b);return b
    def diamond(x,y,w,h,fill='#F2F7FA',edge=blue):
        ax.add_patch(Polygon([(x-w/2,y),(x,y+h/2),(x+w/2,y),(x,y-h/2)],closed=True,fc=fill,ec=edge,lw=.9))
    def photo(im,x,y,w,edge=gray):
        h=w*im.shape[0]/im.shape[1]
        ia=fig.add_axes([x/178,y/110,w/178,h/110]);ia.imshow(im,vmin=0,vmax=1,interpolation='none');ia.axis('off')
        ax.add_patch(Rectangle((x-.15,y-.15),w+.3,h+.3,fill=False,ec=edge,lw=.6))
        return h
    
    for xx in [38.5,83.8,140.5]:line([(xx,11),(xx,101)],color='#BCC1C6',lw=.5)
    for cx,s in [(20,'1. Proposal'),(61,'2. TV budget'),(112,'3. Segment search'),(160,'4. Array audit')]:txt(cx,105,s,size=9.6,weight='bold')
    
    # 1. Existing denoiser and explicitly retained observation/proposal inputs.
    txt(20,97,r'Observation $f$')
    photo(f,4,72,32)
    box(7,60,26,8,fill='#E8ECF1');txt(20,64,'Denoiser')
    arr((20,71.5),(20,68.5))
    photo(v,4,31,32)
    arr((20,59.5),(20,52.7))
    txt(20,27,r'Proposal $v$')
    box(7,13,26,7,fill='#F6F7F8');txt(20,16.5,r'$(f,v)$: float32')
    route([(4,83),(1.5,83),(1.5,16.5),(7,16.5)],gray,.65)
    route([(36,41),(37,41),(37,16.5),(33,16.5)],gray,.65)
    # Shared input bus. The denoiser is never inside the search feedback loop.
    route([(20,13),(20,7),(86,7),(86,92),(96,92)],gray,.8)
    route([(86,55),(91,55)],gray,.8)
    txt(89,95,r'$f,v$',size=9.2,color=gray)
    arr((36.4,85),(43,85),blue,.9)
    
    # 2. Forward differences, coupled spatial/channel norm, relative budget.
    box(42,47,38,52,fill='#F5F6F7',edge=gray,rounding=1.6)
    for c,(xx,yy,col) in enumerate([(44,82,'#D18B84'),(45.4,83.4,'#91AD8F'),(46.8,84.8,'#819DBA')]):
        box(xx,yy,9,9,fill=col,edge='#56616C',rounding=0,lw=.5)
        # Small grid lines evoke an RGB array, without inventing numeric elements.
        for k in [1,2]:
            line([(xx+3*k,yy),(xx+3*k,yy+9)],color='white',lw=.4)
            line([(xx,yy+3*k),(xx+9,yy+3*k)],color='white',lw=.4)
    txt(49.5,77.5,'RGB')
    dx=np.zeros_like(f,dtype=np.float64);dy=np.zeros_like(f,dtype=np.float64)
    dx[:,:-1]=f[:,1:].astype(np.float64)-f[:,:-1].astype(np.float64)
    dy[:-1]=f[1:].astype(np.float64)-f[:-1].astype(np.float64)
    for yy,a,label in [(86,np.linalg.norm(dx,axis=2),r'$D_x$'),(73,np.linalg.norm(dy,axis=2),r'$D_y$')]:
        ia=fig.add_axes([65/178,yy/110,12/178,8/110]);ia.imshow(a,cmap='magma',interpolation='none',aspect='auto',vmin=0,vmax=.6);ia.axis('off')
        txt(71,yy+10.5,label)
    route([(56,88),(60,88),(60,90),(64,90)],blue,.75)
    route([(60,88),(60,77),(64,77)],blue,.75)
    route([(77,90),(79,90),(79,71.5),(61,71.5),(61,70.3)],blue,.6)
    line([(71,73),(71,71.5)],color=blue,lw=.6)
    txt(61,67.5,'Joint RGB / x,y norm')
    txt(61,56.5,r'$V_\epsilon(z)=\frac{1}{HW}\sum_{i,j}\sqrt{\|\nabla z_{ij}\|_2^2+\epsilon^2}$',size=9.2)
    # TV and the user-selected beta are separate inputs to the budget computation.
    route([(44,47),(40,47),(40,29.5),(42,29.5)],blue,.8)
    txt(48,43.5,r'$V_\epsilon(f)$',size=9.2,color=blue)
    txt(61,42.5,r'$\beta=0.6$',color=orange)
    line([(46,38),(76,38)],color='#D9DEE2',lw=3)
    line([(46,38),(64,38)],color=orange,lw=3)
    ax.scatter([64],[38],s=25,color=orange,zorder=3)
    box(42,25.5,38,8,fill='#FBF1E5',edge=orange)
    arr((64,36.5),(64,33.7),orange,.8)
    txt(61,29.5,r'$B_\beta=\beta[V_\epsilon(f)-\epsilon]$',size=9.5)
    txt(61,19,r'$T=V_\epsilon(f)-B_\beta$',size=9.4,color=orange)
    # Budget enters the controller before the endpoint test.
    route([(80,29.5),(82,29.5),(82,98.5),(112,98.5),(112,97)],orange,.8)
    txt(88.5,100.5,r'$B_\beta$',size=9.2,color=orange)
    
    # 3. Proposal acceptance bypasses bisection; rejected proposals use the prefix.
    diamond(112,92,32,10)
    txt(112,92,r'Accept$(v)$?',size=9.4)
    route([(128,92),(142,92),(142,84),(144,84)],teal,.9)
    txt(134,95,'yes',color=teal)
    txt(134,88,r'$u=v,\ \eta=1$',color=teal)
    arr((112,87),(112,83),red,.8)
    txt(116,85,'no',color=red,ha='left')
    
    # Actual smoothed-TV curve, evaluated on the real-valued segment for geometry.
    ga=fig.add_axes([94/178,70/110,40/178,12/110])
    grid=np.linspace(0,1,81);f64=f.astype(np.float64);v64=v.astype(np.float64)
    gv=np.array([tv_reference(f64+x*(v64-f64)) for x in grid]);floor=audit['V_f']-audit['B']
    # Geometry of the zero-tolerance real segment is separate from the stored eta_L.
    gl,gu=0.,1.
    for _ in range(56):
        gm=(gl+gu)/2
        if tv_reference(f64+gm*(v64-f64))>=floor:gl=gm
        else:gu=gm
    geometry_eta=gl
    ga.axvspan(0,geometry_eta,color='#E6F3EF');ga.axvspan(geometry_eta,1,color='#FAEDED')
    ga.plot(grid,gv,color=blue,lw=1.2);ga.axhline(floor,color=orange,ls='--',lw=.8)
    ga.scatter([geometry_eta],[tv_reference(f64+geometry_eta*(v64-f64))],c=teal,s=12,zorder=5)
    ga.set(xlim=(0,1),ylim=(.03,.35));ga.set_xticks([0,geometry_eta,1],['0',r'$\eta^\star$','1']);ga.set_yticks([])
    ga.tick_params(axis='x',length=2,pad=1,labelsize=9.2)
    for side in ['right','top']:ga.spines[side].set_visible(False)
    ga.spines['left'].set_color(gray);ga.spines['bottom'].set_color(gray)
    ga.spines['left'].set_linewidth(.5);ga.spines['bottom'].set_linewidth(.5)
    ga.text(-.05,1.10,'Convex TV',ha='left',va='bottom',transform=ga.transAxes,fontsize=9.2,color=blue)
    ga.text(.29,.17,'feasible',ha='center',va='center',transform=ga.transAxes,fontsize=9.2,color=teal)
    ga.text(.97,(floor-.03)/(.35-.03)+.08,r'$T$',ha='right',va='bottom',transform=ga.transAxes,fontsize=9.2,color=orange)
    
    box(91,48.5,42,10.5,fill='#EEF4F7',edge=blue)
    txt(112,56.3,r'$\eta_M=(\eta_L+\eta_U)/2$',size=9.3)
    txt(112,51.3,r'$u_M=f+\eta_M(v-f)$',size=9.3)
    txt(112,62,r'$\eta_L=0,\ \eta_U=1,\ u_L=f$',size=9.2)
    arr((112,48.3),(112,45.5),blue,.8)
    diamond(112,40.5,31,10)
    txt(112,40.5,r'Accept$(u_M)$?',size=9.2)
    route([(96.5,40.5),(95,40.5),(95,34)],teal,.8)
    route([(127.5,40.5),(130,40.5),(130,34)],red,.8)
    txt(94.5,43.4,'yes',color=teal)
    txt(132,43.4,'no',color=red)
    box(90,25,22,9,fill='#E7F3F0',edge=teal)
    txt(101,31.5,r'$\eta_L\leftarrow\eta_M$',size=9.2,color=teal)
    txt(101,27.7,r'$u_L\leftarrow u_M$',size=9.2,color=teal)
    box(115,25,22,9,fill='#F9ECEA',edge=red)
    txt(126,29.5,r'$\eta_U\leftarrow\eta_M$',size=9.2,color=red)
    line([(101,25),(101,23),(138,23),(138,56)],color=gray,lw=.75)
    line([(126,25),(126,23)],color=gray,lw=.75)
    arr((138,56),(133,56),gray,.75)
    txt(132,20.2,'repeat',size=9.2,color=gray)
    box(93,10,35,8,fill='#E7F3F0',edge=teal)
    txt(110.5,14,r'$K$ steps: return $u_L$',size=9.2,color=teal)
    arr((101,23),(101,18),teal,.8)
    route([(128,14),(142,14),(142,84),(144,84)],teal,.9)
    
    # 4. Recompute the actual returned array and retain a compact audit record.
    txt(160,98,r'Returned $u$')
    photo(u,144,73,32,teal)
    arr((160,72.5),(160,65),teal,.9)
    box(145,53,30,12,fill='#E9F2F7',edge=blue)
    txt(160,61.2,'Recompute TV',size=9.2)
    txt(160,56.4,'float64',size=9.2,color=blue)
    arr((160,52.5),(160,48),blue,.8)
    box(144,39,32,9,fill='#E7F3F0',edge=teal)
    txt(160,43.5,r'$\widehat R\leq\widehat B+\tau$',size=10,color=teal)
    txt(160,35.5,r'$\tau=10^{-12}$',size=9.2)
    arr((160,32.8),(160,29.5),teal,.8)
    box(144,7,32,22,fill='#F5F6F7',edge=gray)
    txt(160,26,'Audit record',size=9.2,weight='bold')
    line([(145,23),(175,23)],color=light,lw=.6)
    for yy,label,value in [(20,r'$\eta$',f'{eta:.4f}'),(15,r'$\widehat R-\widehat B$',f"{audit['signed_residual']:.2e}"),(10,'dtype',str(u.dtype))]:
        txt(146,yy,label,size=9.2,ha='left');txt(174.5,yy,value,size=9.2,ha='right')
    
    # Native 178mm width. Include at textwidth, with no height/scale override.
    text_objects=text_records+list(ga.texts)+list(ga.get_xticklabels())
    save_figure(fig,P,'ICASSP_Overview_Technical',9.2,text_objects=text_objects)
    checks=dict(width_mm=178,height_mm=110,min_normal_font=9.2,
        sample='kodim23',beta=beta,eta=eta,audit=audit,
        geometry_eta_float64=geometry_eta,
        curve='float64 real segment; zero-tolerance boundary differs from retained float32 eta_L')
    (P/'verification/overview_values.json').write_text(json.dumps(checks,indent=2)+'\n')
    print('Generated vector overview: 178mm wide, >=9.2pt base text.')
    plt.close(fig)
