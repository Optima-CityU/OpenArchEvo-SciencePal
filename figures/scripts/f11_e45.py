import sys,json,re,os,numpy as np,matplotlib.pyplot as plt
import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); from style import *; setup()
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..')); s=json.load(open('results/data/lab_jobs.json'))  # run from repo root
def v(c,tag,seeds=(0,1)):
    d={}
    for k,x in s.items():
        if re.match(r'(\w+_)?tr-'+re.escape(c)+rf'-{tag}e30s\d-',k) and x.get('ok'): d.setdefault(x['args']['seed'],x['metrics']['metrics']['wikitext2_ppl'])
    return [d[i] for i in seeds if i in d]
val=lambda c: v(c,'e40') or v(c,'e45')
CF={'V_s1':['r04e40v-d50e6b','r01e40v-2e9cf4'],'V_s2':['r02e40v-5ae8cf','r04e40v-49b52f'],'V_s3':['r04e40v-152ab7','r03e40v-a3b2b3'],
    'O_s1':['r02e40o-e29320','r01e40o-5d1f00'],'O_s2':['r02e40o-ab2506','r02e40o-68c7ec'],'O_s3':['r04e40o-0a22e9','r01e40o-3e33ab']}
m=lambda xs: float(np.mean([min(x,150) for x in xs]))
fig,axes=plt.subplots(1,2,figsize=(9.6,3.4),sharex=True)
for ax,pool,title in [(axes[0],'V','Pool searched with 2-epoch val'),(axes[1],'O','Pool searched with OAF')]:
    for i,k in enumerate((1,2,3)):
        run=f'{pool}_s{k}'; dl=json.load(open('results/e40_runs.json'))['runs'][run]['delivered']
        a=m([x for c in dl for x in val(c)]); b=m([x for c in CF[run] for x in val(c)])
        oaf,v2=(b,a) if pool=='V' else (a,b)
        good = oaf<v2
        ax.plot([oaf,v2],[i,i],color=GRID,lw=6,solid_capstyle='round',zorder=1)
        ax.scatter([v2],[i],s=70,color=C['sig'],zorder=3,edgecolor='white',linewidth=1)
        ax.scatter([oaf],[i],s=70,color=C['champ'],zorder=3,edgecolor='white',linewidth=1)
        ax.text(max(oaf,v2)+1.2,i,f'Δ {v2-oaf:+.1f}',va='center',fontsize=8.6,color=INK if good else INK2)
    ax.axvline(71.82,color=MUTED,lw=1,ls='--')
    ax.set_yticks([0,1,2]); ax.set_yticklabels([f'run {k}' for k in (1,2,3)]); ax.set_ylim(2.6,-0.8); ax.grid(axis='y',visible=False)
    ax.set_title(title,fontsize=10.5)
axes[0].text(71.95,-0.55,'DeltaNet 71.8',fontsize=8,color=INK2)
axes[0].scatter([],[],color=C['champ'],s=50,label='top-2 by OAF'); axes[0].scatter([],[],color=C['sig'],s=50,label='top-2 by 2-epoch val')
fig.legend(loc='upper right',ncol=2,fontsize=8.5,bbox_to_anchor=(0.99,1.0))
axes[0].set_xlim(68,104)
fig.supxlabel('30-epoch val PPL, mean of the 2 selected architectures × 2 seeds (lower is better)',fontsize=9.5,color=INK2,y=0.0)
fig.suptitle('Same candidate pool, only the selection rule changes (E45)',x=0.01,ha='left',fontweight='semibold',fontsize=11.5,color=INK)
fig.text(0.01,-0.08,'Δ = 2-epoch-val pick minus OAF pick; positive means OAF chose better. Stalled runs counted as 150. OAF wins 2/3 in each pool; both losses are run 1.',fontsize=8,color=MUTED)
fig.tight_layout()
save(fig,'F11_e45_counterfactual')
