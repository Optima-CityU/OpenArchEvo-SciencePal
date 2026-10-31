import sys,re,json,numpy as np,matplotlib.pyplot as plt
import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); from style import *; setup()
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..'))  # repo root
s=json.load(open('results/data/lab_jobs.json'))
def e30(c):
    d={}
    for k,v in s.items():
        if v.get('ok') and 'e30' in (v.get('protocol') or '') and re.match(r'(0_)?tr-'+re.escape(c)+r'-',k): d.setdefault(v['args']['seed'],v['metrics']['metrics']['wikitext2_ppl'])
    return list(d.values())
R=[('Champion: spiking gate after o_proj, no activation','r02a-e29092',C['champ']),
   ('… same gate moved before o_proj','e41-pre',C['ctl']),
   ('… gate reads the projected output instead of the input','e41-selfin',C['ctl']),
   ('… gate input not spiking (ANN gate)','e41-anngate',C['mag']),
   ('… with sigmoid activation (old gate)','r03a-263bee',C['sig']),
   ('Matched-parameter control: widened DeltaNet','ctl-deltanet-pm',C['dn']),
   ('Matched-parameter control: DeltaNet use_gate','ctl-deltanet-usegate',C['dn'])]
fig,ax=plt.subplots(figsize=(8.6,3.8))
for i,(lb,c,col) in enumerate(R):
    v=e30(c); m=np.mean(v)
    ax.plot([min(v),max(v)],[i,i],color=col,lw=1.5,alpha=0.6); ax.scatter(v,[i]*len(v),s=14,color=col,alpha=0.5)
    ax.scatter([m],[i],s=60,color=col,zorder=3,edgecolor='white',linewidth=1.2)
    ax.text(74.25,i,f'{m:.2f}' + ('' if i==0 else f'   {m-68.95:+.2f}'),va='center',fontsize=9.5,color=INK)
ax.axvline(71.82,color=MUTED,lw=1,ls='--'); ax.text(71.87,6.45,'DeltaNet 71.8',fontsize=8.5,color=INK2)
ax.set_yticks(range(len(R))); ax.set_yticklabels([r[0] for r in R],fontsize=9.2); ax.invert_yaxis(); ax.set_xlim(68.4,74.2); ax.grid(axis='y',visible=False)
ax.set_xlabel('30-epoch validation PPL (2–3 seeds; small dots = seeds)'); ax.set_title('One change at a time: no sigmoid and a spike-driven input both matter')
save(fig,'F9_champion_ablation')
