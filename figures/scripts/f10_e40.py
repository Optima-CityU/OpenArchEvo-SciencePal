import sys,json,re,os,numpy as np,matplotlib.pyplot as plt
import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); from style import *; setup()
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..')); s=json.load(open('results/data/lab_jobs.json'))  # run from repo root
def vals(c,seeds=(0,1)):
    d={}
    for k,v in s.items():
        if re.match(r'(1_40_\w+_|0_40_\w+_|1_60_\w+_)?tr-'+re.escape(c)+r'-e40e30s\d-',k) and v.get('ok'):
            d.setdefault(v['args']['seed'],v['metrics']['metrics']['wikitext2_ppl'])
    return [d[x] for x in seeds if x in d]
A=[('V','2-epoch val fitness',C['sig']),('O','OAF fitness',C['champ']),('L','OAF + research-lead agent',C['ctl'])]
fig,ax=plt.subplots(figsize=(8.4,4.0)); CAP=83; y=0; yt=[]; yl=[]
for arm,lb,col in A:
    for k in (1,2,3):
        name=f'{arm}_s{k}'; ids=json.load(open('results/e40_runs.json'))['runs'][name]['delivered']
        v=[x for c in ids for x in vals(c)]
        if v:
            vc=np.minimum(v,CAP); ax.scatter(vc,[y]*len(v),s=34,color=col,alpha=0.75,edgecolor='white',linewidth=0.8,zorder=3)
            m=np.mean([min(x,150) for x in v]); md=np.median(v)
            if m<CAP: ax.scatter([m],[y],marker='|',s=320,color=INK,zorder=4,linewidth=2)
            ax.text(CAP+5.6,y,f'mean {m:.1f}',va='center',ha='right',fontsize=8.6,color=INK)
            if any(x>CAP for x in v): ax.text(CAP-0.5,y-0.3,f'diverged ({", ".join(f"{x:.0f}" for x in v if x>CAP)})',va='center',ha='right',fontsize=8,color=INK2)
        yt.append(y); yl.append(f'{lb} · run {k}'); y+=1
    y+=0.6
ax.axvline(71.82,color=MUTED,lw=1,ls='--'); ax.text(71.95,-0.85,'DeltaNet 71.8',fontsize=8.5,color=INK2,ha='left')
ax.axvline(68.95,color=C['champ'],lw=0.8,ls=':'); ax.text(68.85,-0.85,'champion 68.95',fontsize=8.5,color=C['champ'],ha='right')
ax.set_yticks(yt); ax.set_yticklabels(yl,fontsize=9); ax.set_ylim(y-0.4,-1.2); ax.set_xlim(66.2,CAP+6.5); ax.grid(axis='y',visible=False)
ax.set_xlabel('30-epoch val PPL of each run\'s 2 delivered architectures × 2 seeds (bar = run mean)')
ax.set_title('A fresh co-evolution round: same proposer and budget, three ways to decide')
fig.text(0.01,-0.06,'9 search runs, 4 generations each, seed-paired across arms; Skill v0.5.1 for all. Divergent runs counted as 150 in the run mean.',fontsize=8,color=MUTED)
save(fig,'F10_e40_arms')
