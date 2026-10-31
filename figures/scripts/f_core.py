import json,re,sys,csv,numpy as np,matplotlib.pyplot as plt
import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from style import *; setup()
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..'))  # repo root
from scipy.stats import spearmanr
s=json.load(open('results/data/lab_jobs.json')); S=json.load(open('results/data/curves_2ep.json'))
def e30(c,seed=None):
    xs=[(v['args'].get('seed'),v['metrics']['metrics']['wikitext2_ppl']) for k,v in s.items() if v.get('ok') and 'e30' in (v.get('protocol') or '') and re.match(r'(0_)?tr-'+re.escape(c)+r'-',k)]
    if seed is not None: xs=[x for x in xs if x[0]==seed]
    d={}
    for sd,x in xs: d.setdefault(sd,x)   # one value per seed (dedupe identical reruns)
    return list(d.values())
def two(c):
    for pat in (r'tr-'+re.escape(c)+r'-[0-9a-f]{6}$', r'tr-'+re.escape(c)+r'-e31be2s0-'):
        for j,v in S.items():
            if re.match(pat,j) and len(v['epochs'])==2:
                d={e:(t,va) for e,t,va in v['epochs']}; return d[2][1],d[2][0]
DN=71.8
# ---------- F4 selection rules
pick=json.load(open('results/data/e29_pick.json'))['pick']; st2=json.load(open('results/data/e29_step2.json'))
G=[('Lowest 2-epoch val',[c for _,c,_ in pick[:4]],C['sig']),('6-epoch ladder',st2['top4'],C['yellow']),('OAF (same 2-epoch log)',['r09v1-673a31','r14a-d60a72','r14a-acc9a5','r03a5-b5331e'],C['champ'])]
fig,ax=plt.subplots(figsize=(7.2,3.6))
for i,(nm,cs,col) in enumerate(G):
    ms=[]
    for j,c in enumerate(cs):
        v=e30(c); m=np.mean(v); ms.append(m); y=i+(j-1.5)*0.12
        ax.plot([min(v),max(v)],[y,y],color=col,lw=1.5,alpha=0.6); ax.scatter([m],[y],s=46,color=col,zorder=3,edgecolor='white',linewidth=1.2)
    ax.scatter([np.mean(ms)],[i],marker='|',s=900,color=INK,zorder=4,linewidth=2)
    ax.text(80.6,i,f'mean {np.mean(ms):.1f}   ·   {sum(m<DN for m in ms)}/4 beat expert',va='center',fontsize=9.5,color=INK)
ax.axvline(DN,color=MUTED,lw=1,ls='--'); ax.text(DN+0.1,-0.42,'DeltaNet 71.8',ha='left',fontsize=8.5,color=INK2)
ax.set_yticks(range(3)); ax.set_yticklabels([g[0] for g in G]); ax.invert_yaxis(); ax.set_xlim(67,80.3); ax.grid(axis='y',visible=False)
ax.set_xlabel('30-epoch validation PPL (dot = mean of 2 seeds, bar = range; lower is better)')
ax.set_title('Same 16 untested candidates, three ways to pick 4 for long training')
fig.text(0.01,-0.08,'Candidates fixed by a pre-registered rule before any long training. OAF = val₂ + 0.5·(val₂ − train₂), computed from the existing 2-epoch log (no extra GPU).',fontsize=8,color=MUTED)
save(fig,'F4_selection_rules')
# ---------- F5 OAF vs val2 scatter (held-out)
train={r['arch'] for r in csv.DictReader(open('prompts/panels/e32/train.csv'))}
allc=set()
for k,v in s.items():
    m=re.match(r'(?:0_)?tr-(r\w+?-[0-9a-f]{6})-',k)
    if m and v.get('ok') and 'e30' in (v.get('protocol') or '') and v['args'].get('seed')==0: allc.add(m.group(1))
pts=[]
for c in sorted(c for c in allc-train if 'e40' not in c and not c.startswith('e3')):
    t=two(c); y=e30(c,0)
    if t and y: pts.append((t[0],t[0]+0.5*(t[0]-t[1]),y[0]))
P=np.array(pts)
fig,axs=plt.subplots(1,2,figsize=(9.6,3.9),sharey=True)
for ax,(xi,nm,col) in zip(axs,[(0,'2-epoch val PPL (what the search used)',C['sig']),(1,'OAF score (same log)',C['champ'])]):
    rho=spearmanr(P[:,xi],P[:,2])[0]
    ax.scatter(P[:,xi],P[:,2],s=34,color=col,alpha=0.85,edgecolor='white',linewidth=0.8)
    ax.set_xlabel(nm); ax.set_title(f'Spearman ρ = {rho:.2f}',color=col)
    ax.axhline(DN,color=MUTED,lw=0.8,ls='--')
axs[0].set_ylabel('30-epoch val PPL'); axs[0].set_yscale('log'); axs[0].set_yticks([70,80,90,110]); axs[0].get_yaxis().set_major_formatter(plt.FuncFormatter(lambda y,_:f'{y:.0f}'))
axs[0].text(axs[0].get_xlim()[0],DN+0.6,' DeltaNet',fontsize=8.5,color=INK2)
fig.suptitle(f'Held-out architectures (n = {len(P)}): the score already in the logs ranks long-training quality',x=0.01,ha='left',fontsize=12,fontweight='semibold',y=1.04)
_r=np.random.default_rng(0); _b=[]
for _ in range(4000):
    i=_r.integers(0,len(P),len(P)); _b.append(spearmanr(P[i,1],P[i,2])[0])
lo,hi=np.percentile(_b,[2.5,97.5])
fig.text(0.01,-0.06,f'Pre-registered E31 cohort: architectures never used to choose the formula or λ (E40 deliveries excluded: they were selected by these scores). Bootstrap 95% CI for OAF ρ: [{lo:.2f}, {hi:.2f}].',fontsize=8,color=MUTED)
save(fig,'F5_oaf_scatter')
# ---------- F3 fidelity curve
F=json.load(open('results/data/fidelity_curve.json'))
_ns={'__file__':'analysis/fidelity_curve.py'}; exec(open('analysis/fidelity_curve.py').read().split("print('n architectures")[0],_ns); E=_ns['E']; names=_ns['names']; final=_ns['final']
ks=list(range(1,30)); rv=[spearmanr([E[c][k][1] for c in names],final)[0] for k in ks]
fig,ax=plt.subplots(figsize=(7.4,3.6))
ax.axhline(0,color=MUTED,lw=0.8); ax.plot(ks,rv,color=C['champ'],marker='o',ms=3.5)
ax.axvspan(0.5,2.5,color=C['sig'],alpha=0.08); ax.text(1.5,0.92,'search\nscore',ha='center',fontsize=8.5,color=C['sig'])
ax.axvline(6,color=C['yellow'],lw=1,ls=':'); ax.text(6.3,-0.5,'6-epoch ladder',fontsize=8.5,color=C['yellow'])
ax.set_ylim(-0.6,1.05); ax.set_xlim(0.5,30); ax.set_xlabel('epoch k within a 30-epoch run'); ax.set_ylabel('Spearman ρ with final 30-epoch result')
ax.set_title(f'Rankings only become trustworthy after ~15–20 epochs (n = {len(names)} architectures)')
save(fig,'F3_fidelity')
# ---------- F7 gate activation ladder
GL=[('identity  g','champ','search (Skill v0.3.3)',e30('r02a-e29092')),('g/√(1+0.1g²)','champ','GPT-6.1 (panel)',e30('e35-softclip')),('|g|','ctl','E33 test',e30('e33-abs')),
    ('tanh g','ctl','E33 test',e30('e33-tanh')),('1 + g','ctl','E36 test',e30('e36-onepg')),('g·σ(g)  SiLU','mag','4 of 5 models (panel)',e30('e35-silu')),
    ('1 + tanh g','ctl','E21 panel',e30('e21-i5-tanh')),('σ(g) − 0.5','ctl','E36 test',e30('e36-sigm05')),('σ(g)  old gate','sig','search (Skill v0.3.3)',e30('r03a-263bee')),('2·σ(g)','ctl','E33 test',e30('e33-2sig'))]
fig,ax=plt.subplots(figsize=(7.6,4.4))
for i,(nm,ck,src,v) in enumerate(GL):
    m=np.mean(v); ax.plot([min(v),max(v)],[i,i],color=C[ck],lw=1.5,alpha=0.6); ax.scatter(v,[i]*len(v),s=12,color=C[ck],alpha=0.5)
    ax.scatter([m],[i],s=55,color=C[ck],zorder=3,edgecolor='white',linewidth=1.2); ax.text(75.0,i,f'{m:.2f}   {src}',va='center',fontsize=8.8,color=INK2)
ax.axvline(DN,color=MUTED,lw=1,ls='--'); ax.text(DN+0.05,9.45,'DeltaNet 71.8',ha='left',fontsize=8.5,color=INK2)
ax.set_yticks(range(len(GL))); ax.set_yticklabels([g[0] for g in GL]); ax.invert_yaxis(); ax.set_xlim(68,74.8); ax.grid(axis='y',visible=False)
ax.set_xlabel('30-epoch val PPL (small dots = seeds)'); ax.set_title('Who proposed which gate, and what the GPU decided')
fig.text(0.01,-0.05,'All variants change only the activation of the spiking gate after o_proj in the champion program; same parameters.',fontsize=8,color=MUTED)
save(fig,'F7_gate_ladder')
# ---------- F6 E30 arms
t=open('docs/FINAL_ANALYSIS_1030.md').read()
rows=re.findall(r'\| (V1|A|V2|A5|V2nk) \| \S+ \| \S+ \| [\d.]+ \| ([\d.]+) \|',t); d={}
for a,v in rows: d.setdefault(a,[]).append(float(v))
order=[('V1','Skill v0.1'),('A','Skill v0.3.3'),('A5','Skill v0.5'),('V2','v0.3.3 + research-lead agent'),('V2nk','lead agent, no archive knowledge')]
fig,ax=plt.subplots(figsize=(7.6,3.6)); cols=[C['dn'],C['ctl'],C['yellow'],C['champ'],C['mag']]
for i,((a,lb),col) in enumerate(zip(order,cols)):
    v=np.array(d[a]); vc=np.minimum(v,90); ax.scatter(vc,[i]*len(v),s=50,color=col,edgecolor='white',linewidth=1,zorder=3)
    for x in v[v>90]: ax.annotate(f'{x:.0f} (diverged)',(90,i),xytext=(85.6,i+0.38),fontsize=8,color=INK2,arrowprops=dict(arrowstyle='->',color=MUTED,lw=0.8))
    ax.text(91.2,i,f'median {np.median(v):.1f}',va='center',fontsize=9,color=INK)
ax.axvline(DN,color=MUTED,lw=1,ls='--'); ax.set_yticks(range(5)); ax.set_yticklabels([o[1] for o in order]); ax.invert_yaxis(); ax.set_xlim(67,90.5); ax.grid(axis='y',visible=False)
ax.set_xlabel('30-epoch val PPL of the architecture each search run delivered (by its 2-epoch score)')
ax.set_title('Re-judging the search arms with the long-training ruler (5 runs each)')
save(fig,'F6_arms_30ep')
print('done')
