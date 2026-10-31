#!/usr/bin/env python3
"""E29/E30/E31/E32/E33/E35 scoring from results/data (trimmed GPU job store + 2-epoch curves). Prints markdown.
The E30 section needs the raw run directories (runs/e*/*_s*), which are not part of the release; it is skipped when absent."""
import json, re, glob, csv, zipfile, importlib.util, sys
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'evo')); import op_analysis
D = ROOT / 'results' / 'data'
s = json.load(open(D / 'lab_jobs.json'))
def e30(c, seed=None):
    xs = [(v['args'].get('seed'), v['metrics']['metrics']['wikitext2_ppl']) for k, v in s.items()
          if v.get('ok') and 'e30' in (v.get('protocol') or '') and re.match(r'(0_)?tr-' + re.escape(c) + r'-', k)]
    if seed is not None: xs = [x for x in xs if x[0] == seed]
    return [x for _, x in xs]
m = lambda xs: float(np.mean(xs)) if xs else None
f = lambda x: '—' if x is None else f'{x:.2f}'
S = json.load(open(D / 'curves_2ep.json'))
def two(c, seed=0):
    pats = [r'tr-' + re.escape(c) + r'-[0-9a-f]{6}$', r'tr-' + re.escape(c) + r'-e31be2s0-'] if seed == 0 else [r'tr-' + re.escape(c) + rf'-e(39e2|26ne2)s{seed}-']
    for pat in pats:
        for j, v in S.items():
            if re.match(pat, j) and len(v['epochs']) == 2 and v.get('finished'):
                d = {e: (t, va) for e, t, va in v['epochs']}; return d[2][1], d[2][0]
DN = 71.82
print('## E29 / E31 前瞻选择')
pick = json.load(open(D / 'e29_pick.json'))['pick']; v6 = json.load(open(D / 'e29_step2.json'))['v6'] if (D / 'e29_step2.json').exists() else {}
groups = {'2 轮前 4（对照）': [c for _, c, _ in pick[:4]], '6 轮前 4（阶梯）': json.load(open(D / 'e29_step2.json'))['top4'] if v6 else [],
          'OAF 前 4': ['r09v1-673a31', 'r14a-d60a72', 'r14a-acc9a5', 'r03a5-b5331e']}
print('| 组 | 架构（30 轮两种子均值） | 组均值 | 好于 DeltaNet 的个数 |\n|---|---|---|---|')
for g, cs in groups.items():
    vals = [(c, m(e30(c))) for c in cs]
    ok = [v for _, v in vals if v is not None]
    print(f"| {g} | {'; '.join(f'{c} {f(v)}' for c, v in vals)} | {f(m(ok))} | {sum(v < DN for v in ok)}/{len(ok)} |")
print('\n## E30 每次运行交出的架构（2 轮最好）在 30 轮')
op_analysis.B = 10 ** 6
rows = []
for d in sorted((ROOT / 'runs').glob('e*/*_s*')):
    if d.parent.name == 'e3' or not (d / 'events.jsonl').exists(): continue
    R = [r for r in op_analysis.run_children(d)[:26] if r['value'] is not None]
    if not R: continue
    b = min(R, key=lambda r: r['value'])
    arm = d.name.split('_')[0]
    rows.append((arm, d.name, b['id'], b['value'], m(e30(b['id'], 0)) or m(e30(b['id']))))
print('| 组 | 运行 | 架构 | 2 轮 | 30 轮 |\n|---|---|---|---|---|')
for r in rows: print(f'| {r[0]} | {r[1]} | {r[2]} | {r[3]:.2f} | {f(r[4])} |')
ok = [r for r in rows if r[4] is not None]
if len(ok) > 4: print(f'- Q1 ρ(2 轮最好, 30 轮) = {spearmanr([r[3] for r in ok], [r[4] for r in ok])[0]:.2f}（n = {len(ok)}）')
for arm in ('V1', 'A', 'V2', 'A5', 'V2nk'):
    v = [r[4] for r in ok if r[0] == arm]
    if v: print(f'- {arm}: 30 轮 均值 {np.mean(v):.2f} 中位数 {np.median(v):.2f}（n = {len(v)}），好于 DeltaNet {sum(x < DN for x in v)}，差 3 以上 {sum(x > DN + 3 for x in v)}')
print('\n## E31 / E32 样本外：2 轮信号 vs 30 轮（种子 0）')
test = [r['arch'] for r in csv.DictReader(open(ROOT / 'prompts/panels/e32/test.csv'))]
cand = sorted(set(test) | {r[2] for r in rows} | {c for _, c, _ in pick})
train_names = {r['arch'] for r in csv.DictReader(open(ROOT / 'prompts/panels/e32/train.csv'))}
pts = []
for c in cand:
    if c in train_names: continue
    t = two(c); y = m(e30(c, 0))
    if t and y: pts.append((c, t[0], t[1], y))
if len(pts) >= 5:
    v2 = [p[1] for p in pts]; tr = [p[2] for p in pts]; y = [p[3] for p in pts]
    oaf = [a + 0.5 * (a - b) for a, b in zip(v2, tr)]
    print(f'- n = {len(pts)}；ρ(val₂) = {spearmanr(v2, y)[0]:.2f}；ρ(OAF) = {spearmanr(oaf, y)[0]:.2f}')
    z = zipfile.ZipFile(ROOT / 'prompts/panels/e32/programs.zip'); code = lambda a: z.read(f'programs/{a}.py').decode() if f'programs/{a}.py' in z.namelist() else ''
    rowd = {r['arch']: r for r in csv.DictReader(open(ROOT / 'prompts/panels/e32/test.csv'))}
    for p in sorted(glob.glob(str(ROOT / 'results/panels/e32/pred/*.py'))):
        tag = Path(p).stem
        try:
            spec = importlib.util.spec_from_file_location(tag, p); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            q = [(mod.score(dict(rowd[c]), code(c)), yy) for c, _, _, yy in pts if c in rowd]
            print(f'- E32 {tag}: ρ = {spearmanr(*zip(*q))[0]:.2f}（n = {len(q)}）')
        except Exception as e: print('-', tag, 'ERR', str(e)[:80])
print('\n## E33 / E35 门激活（30 轮种子 0）')
for k in ('e33-2sig', 'e33-tanh', 'e33-abs', 'e35-silu', 'e35-softclip'):
    print(f'- {k}: {f(m(e30(k)))}')

print('\n## E39 种子 1 的 2 轮日志（全部有 30 轮真值的架构）')
allc = set()
for k, v in s.items():
    mm = re.match(r'(?:0_)?tr-(r\w+?-[0-9a-f]{6})-', k)
    if mm and v.get('ok') and 'e30' in (v.get('protocol') or '') and v['args'].get('seed') == 0: allc.add(mm.group(1))
allc |= {'i00-deltanet', 'r03a-263bee', 'r04v2-bf4d6a'}
pp = []
for c in sorted(allc):
    y = m(e30(c, 0)); ts = [two(c, sd) for sd in (0, 1, 2, 3)]
    if y and all(ts): pp.append(([t[0] for t in ts], [t[0] + 0.5 * (t[0] - t[1]) for t in ts], y))
if len(pp) >= 5:
    yy = [p[2] for p in pp]
    for sd in range(4):
        print(f'- 种子 {sd}：n = {len(pp)}；ρ(val₂) = {spearmanr([p[0][sd] for p in pp], yy)[0]:.2f}；ρ(OAF) = {spearmanr([p[1][sd] for p in pp], yy)[0]:.2f}')
    print(f'- 4 种子平均：ρ(val₂) = {spearmanr([np.mean(p[0]) for p in pp], yy)[0]:.2f}；ρ(OAF) = {spearmanr([np.mean(p[1]) for p in pp], yy)[0]:.2f}')
pp0 = []
for c in sorted(allc):
    t = two(c, 0); y = m(e30(c, 0))
    if t and y: pp0.append((t[0], t[0] + 0.5 * (t[0] - t[1]), y))
if len(pp0) >= 5:
    a = np.array(pp0); print(f'- 全部（含训练集）种子 0：n = {len(pp0)}；ρ(val₂) = {spearmanr(a[:,0], a[:,2])[0]:.2f}；ρ(OAF) = {spearmanr(a[:,1], a[:,2])[0]:.2f}')
