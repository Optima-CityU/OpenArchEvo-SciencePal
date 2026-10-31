"""E45 counterfactual selection inside the E40 candidate pools (30-epoch best val PPL)."""
import json,re,os,numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
s=json.load(open(ROOT/'results/data/lab_jobs.json')); E=json.load(open(ROOT/'results/e40_runs.json'))['runs']
def v(c,tag,seeds=(0,1)):
    d={}
    for k,x in s.items():
        if re.match(r'(\w+_)?tr-'+re.escape(c)+rf'-{tag}e30s\d-',k) and x.get('ok'): d.setdefault(x['args']['seed'],x['metrics']['metrics']['wikitext2_ppl'])
    return [d[i] for i in seeds if i in d]
def val(c):
    x=v(c,'e40') or v(c,'e45'); return x
CF={'V_s1':['r04e40v-d50e6b','r01e40v-2e9cf4'],'V_s2':['r02e40v-5ae8cf','r04e40v-49b52f'],'V_s3':['r04e40v-152ab7','r03e40v-a3b2b3'],
    'O_s1':['r02e40o-e29320','r01e40o-5d1f00'],'O_s2':['r02e40o-ab2506','r02e40o-68c7ec'],'O_s3':['r04e40o-0a22e9','r01e40o-3e33ab']}
m=lambda xs: np.mean([min(x,150) for x in xs]) if xs else None
print('| 运行 | 实际交付（本组规则）| 30 轮均值 | 反事实（另一规则）| 30 轮均值 |\n|---|---|---|---|---|')
for run,cf in CF.items():
    dl=E[run]['delivered']
    a=[x for c in dl for x in val(c)]; b=[x for c in cf for x in val(c)]
    rule='OAF' if run[0]=='V' else 'val₂'
    print(f"| {run} | {', '.join(dl)} | {m(a):.2f}（n={len(a)}） | {rule}: {', '.join(cf)} | {'—' if not b else f'{m(b):.2f}（n={len(b)}）'} |")
print('\n种子 0（预注册主判定）:')
for run,cf in CF.items():
    dl=E[run]['delivered']
    a=[x for c in dl for x in val(c)[:0]+v(c,'e40',(0,))]; b=[x for c in cf for x in (v(c,'e40',(0,)) or v(c,'e45',(0,)))]
    print(run, 'deliv', [round(x,2) for x in a], m(a), 'cf', [round(x,2) for x in b], m(b))
