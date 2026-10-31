#!/usr/bin/env python3
"""Rank agreement between epoch-k val PPL and final 30-epoch result, from per-epoch curves of finished 30-epoch jobs.
One curve per architecture (seed 0 preferred). Also tests overfitting signals at epoch k (train/val gap)."""
import json, re, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
from scipy.stats import spearmanr
C = json.load(open(ROOT / 'results/data/curves_30ep.json'))
def cand(j):
    m = re.match(r'tr-(r\w+?-[0-9a-f]{6}|i\d\d-(?:deltanet|hgrn2|mamba2-ssd|s\d{4})|e21-i\d-[a-z]+(?:-[a-z]+)?|ctl-deltanet-(?:pm|usegate))', j)
    return m.group(1) if m else None
best = {}
for j, v in C.items():
    c = cand(j); ep = v['epochs']
    if not c or not v.get('finished') or len(ep) < 30: continue
    s0 = ('s0' in j) or not re.search(r's[12]-', j)
    if c not in best or (s0 and not best[c][0]): best[c] = (s0, ep)
names = sorted(best)
E = {c: {e: (tr, va) for e, tr, va in best[c][1]} for c in names}
final = np.array([min(va for tr, va in E[c].values()) for c in names])
print('n architectures with full 30-epoch curves:', len(names))
out = []
for k in (1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25):
    vk = np.array([E[c][k][1] for c in names]); gap = np.array([E[c][k][1] - E[c][k][0] for c in names])
    r = spearmanr(vk, final)[0]; rg = spearmanr(gap, final)[0]
    # best val so far up to k + lambda*gap  (simple combined score)
    out.append((k, round(r, 2), round(rg, 2)))
    print(f'epoch {k:2d}: rho(val_k, final) = {r:.2f}   rho(train/val gap_k, final) = {rg:.2f}')
# top-tier: architectures whose epoch-2 val is within 1.5 PPL of the best
v2 = np.array([E[c][2][1] for c in names]); top = v2 <= v2.min() + 3
print('top tier (epoch-2 within 3 PPL of best):', int(top.sum()))
for k in (2, 6, 10, 15, 20):
    vk = np.array([E[c][k][1] for c in names])
    print(f'  top tier epoch {k}: rho = {spearmanr(vk[top], final[top])[0]:.2f}')
json.dump({'names': names, 'final': final.tolist(), 'rho': out}, open(ROOT / 'results/data/fidelity_curve.json', 'w'))
