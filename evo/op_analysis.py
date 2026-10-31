#!/usr/bin/env python3
"""Offline (zero-GPU) analysis of the variation operator each arm implements (E7 + E9, later E10).

For every child trained within the first B trainings of a run (budget order as in e7_curves):
  delta     = child e2 PPL - best measured e2 PPL among its parents (negative = better than parent)
  lethal    = failed training, diverged, or PPL > 130
  code_sim / fp_sim = max over parents of the similarity recorded at registration (code embedding / behaviour fingerprint)
  dparams   = relative change in params/block vs the closest parent
Per arm: lethality, delta quantiles, P(delta < 0), edit size, role mix.
  python evo/op_analysis.py [budget] -> docs/OPERATOR_analysis.md + runs/op_analysis.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent
B = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 15


def load(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def run_children(d):
    pop = {c["id"]: c for c in load(d / "population.jsonl")}
    ev = load(d / "events.jsonl")
    infra = {(e["id"], e["failure_ts"]) for e in ev if e.get("event") == "infra_retrain"}
    outs = [(e.get("budget_ts", e["ts"]), e["id"], e.get("value"), None) for e in ev if e.get("event") == "measured"]
    outs += [(e["ts"], e["id"], None, e.get("error", "")) for e in ev if e.get("event") == "train_failed" and (e["id"], e["ts"]) not in infra]
    outs.sort()
    outs = outs[:B]
    reg = {e["id"]: e for e in ev if e.get("event") == "registered"}
    start = {e["id"]: e for e in ev if e.get("event") == "proposal_start"}
    meas = {}
    for c in pop.values():
        v = (c.get("measured") or {}).get("wikitext2_ppl")
        if v is not None:
            meas[c["id"]] = v
    rows = []
    for ts, cid, v, err in outs:
        ps = (start.get(cid) or {}).get("parents") or pop.get(cid, {}).get("meta", {}).get("parents") or []
        pv = [meas[p] for p in ps if p in meas]
        sims = (reg.get(cid) or {}).get("parent_similarity") or {}
        cs = [s[0] for s in sims.values() if s]
        fs = [s[-1] for s in sims.values() if s]
        pp = lambda x: ((pop.get(x, {}).get("meta", {}).get("cpu_demo")) or {}).get("params_per_block")
        mine = pp(cid)
        pps = [pp(p) for p in ps if pp(p)]
        dpar = None
        if mine and pps:
            ref = min(pps, key=lambda q: abs(q - mine))
            dpar = (mine - ref) / ref
        if v is None:
            kind = "diverged" if "diverged" in (err or "") else "error"
        elif v > 130:
            kind = ">130"
        else:
            kind = "ok"
        rows.append({"id": cid, "role": (start.get(cid) or {}).get("role"), "n_parents": len(ps), "value": v, "kind": kind,
                     "lethal": kind != "ok", "delta": (v - min(pv)) if (v is not None and pv and v <= 130) else None,
                     "code_sim": max(cs) if cs else None, "fp_sim": max(fs) if fs else None, "dparams": dpar})
    return rows


def q(a, p):
    return round(float(np.percentile(a, p)), 2) if len(a) else None


def main():
    dirs = sorted((ROOT / "runs/e7").glob("V?_s?")) + sorted((ROOT / "runs/e10").glob("?_s?")) + sorted((ROOT / "runs/e16").glob("A5_s?")) + sorted((ROOT / "runs/e17").glob("V2nk_s?"))
    R = {d.name: run_children(d) for d in dirs}
    arms = {}
    for name, rows in R.items():
        arms.setdefault(name.split("_")[0], []).extend(rows)
    L = [f"# 变异算子分析（零 GPU，离线）：每次运行的前 {B} 个训练\n",
         "delta = 子代 e2 PPL − 父代中最好的 e2 PPL（负数表示比父代好）；致死 = 训练失败、发散或 PPL > 130。\n",
         "| 组 | 运行数 | 子代 | 致死率 | 报错 / 发散 / >130 | delta 10% / 50% / 90% 分位 | P(delta<0) | 代码相似度中位 | 指纹相似度中位 | 参数变化中位（绝对值） | 角色 探索/重组/简化 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    S = {}
    for arm, rows in arms.items():
        n = len(rows)
        if not n:
            continue
        d = [r["delta"] for r in rows if r["delta"] is not None]
        k = {t: sum(1 for r in rows if r["kind"] == t) for t in ("error", "diverged", ">130")}
        cs = [r["code_sim"] for r in rows if r["code_sim"] is not None]
        fs = [r["fp_sim"] for r in rows if r["fp_sim"] is not None]
        dp = [abs(r["dparams"]) for r in rows if r["dparams"] is not None]
        ro = {t: sum(1 for r in rows if r["role"] == t) for t in ("explore", "recombine", "simplify")}
        runs = sum(1 for x in R if x.startswith(arm + "_"))
        S[arm] = {"delta": d, "code_sim": cs, "fp_sim": fs, "dparams": dp, "lethal": [r["lethal"] for r in rows]}
        L.append(f"| {arm} | {runs} | {n} | {sum(r['lethal'] for r in rows) / n:.0%} | {k['error']} / {k['diverged']} / {k['>130']} | "
                 f"{q(d, 10)} / {q(d, 50)} / {q(d, 90)} | {np.mean([x < 0 for x in d]):.0%} | {q(cs, 50)} | {q(fs, 50)} | "
                 f"{q(dp, 50)} | {ro['explore']}/{ro['recombine']}/{ro['simplify']} |")
    if "V1" in S and "V2" in S:
        L += ["", "V1 vs V2（双侧 Mann–Whitney，探索性）："]
        for key, lab in (("delta", "delta"), ("code_sim", "代码相似度"), ("fp_sim", "指纹相似度"), ("dparams", "参数变化")):
            a, b = S["V1"][key], S["V2"][key]
            if a and b:
                L.append(f"- {lab}：V1 中位 {q(a, 50)}，V2 中位 {q(b, 50)}，p = {mannwhitneyu(a, b).pvalue:.3f}")
        L.append(f"- 上尾：最好的 delta（越负越好）V1 {min(S['V1']['delta']):.2f}，V2 {min(S['V2']['delta']):.2f}；"
                 f"delta 最好 10% 分位 V1 {q(S['V1']['delta'], 10)}，V2 {q(S['V2']['delta'], 10)}")
    (ROOT / "runs/op_analysis.json").write_text(json.dumps(R, indent=1))
    (ROOT / "docs/OPERATOR_analysis.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
