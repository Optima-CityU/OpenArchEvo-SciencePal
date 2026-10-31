#!/usr/bin/env python3
"""E40 scoring per pre-registration: run = unit; run mean over 2 delivered archs x seeds 0,1; paired by seed index."""
import json,glob,re,numpy as np,os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
s=json.load(open(ROOT/'results/data/lab_jobs.json'))  # trimmed GPU job store
E=json.load(open(ROOT/'results/e40_runs.json'))['runs']
def e30(c):
    d={}
    for k,v in s.items():
        if re.match(r'(1_40_\w+_|0_40_\w+_)?tr-'+re.escape(c)+r'-e40e30s\d-',k):
            if v.get('ok'):
                x=v['metrics']['metrics']['wikitext2_ppl']; d.setdefault(v['args']['seed'],x if x<1e4 else float('nan'))
            else: d.setdefault(v['args']['seed'],float('nan'))
    return d
DN=71.8; rows={}
print('| 组 | 运行 | 交付架构（自身适应度） | 30 轮 s0 / s1 | 运行均值 | 运行内最好 |\n|---|---|---|---|---|---|')
for arm in 'VOL':
    for k in (1,2,3):
        name=f'{arm}_s{k}'
        if name not in E: continue
        ids=E[name]['delivered']
        fit={c['id']:c.get('fitness') for g in E[name]['generations'] for c in g['candidates']}
        vals=[]; cells=[]
        for c in ids:
            d=e30(c); xs=[d.get(0),d.get(1)]; vals+= [x for x in xs if x is not None]
            cells.append(f"{c} ({fit[c]:.1f}): {' / '.join('—' if x is None else ('发散' if np.isnan(x) else f'{x:.2f}') for x in xs)}")
        fin=[x for x in vals if not np.isnan(x)]
        rm=np.mean([x if not np.isnan(x) else 150.0 for x in vals]) if vals else None
        rows[name]=(rm,min(fin) if fin else None,len(vals),sum(np.isnan(x) or x>DN+3 for x in vals))
        print(f"| {arm} | {name} | {'<br>'.join(c.split(':')[0] for c in cells)} | {'<br>'.join(c.split(': ')[1] for c in cells)} | {'—' if rm is None else f'{rm:.2f}'} | {'—' if not fin else f'{min(fin):.2f}'} |")
print('\n（发散按 150 计入运行均值；括号内是该架构在本组适应度下的 2 轮分数）')
for a in 'VOL':
    r=[rows.get(f'{a}_s{k}') for k in (1,2,3)]; r=[x for x in r if x and x[0] is not None]
    if r: print(f'- {a}: 运行均值 {np.mean([x[0] for x in r]):.2f}；运行内最好 {np.mean([x[1] for x in r if x[1]]):.2f}；交付训练中比 DeltaNet 差 3 以上（含发散） {sum(x[3] for x in r)}/{sum(x[2] for x in r)}（{len(r)} 次运行有结果）')
for k in (1,2,3):
    v,o,l=(rows.get(f'{a}_s{k}',(None,))[0] for a in 'VOL')
    print(f'- 配对 s{k}: V {v} / O {o} / L {l}')
