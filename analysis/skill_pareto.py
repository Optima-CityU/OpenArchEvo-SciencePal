#!/usr/bin/env python3
"""E4: Skill version ladder + selection gate, as a multi-objective comparison.

Pools the 4 repetitions of each Skill version (fixed plan of 24 proposals, DeepSeek proposer, controller CPU checks) and
reports, with Wilson 90% intervals on pooled counts:
  feasibility  = first-try CPU pass rate (maximise)
  honesty      = 1 - (functional duplicates of the base parent + contract-violating attempts) / proposals (maximise)
  calibration  = correct parameter/firing-rate predictions / predictions (maximise)
  cost         = DeepSeek tokens per accepted proposal (minimise; includes repairs)
Then a Pareto (non-dominated) sort over the point estimates.

  python evo/skill_pareto.py -> runs/ladder/pareto.json + docs/fig_skill_pareto.png
"""
import glob
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERS = {"v01": "v0.1", "v02": "v0.2", "v032": "v0.3.2", "v04": "v0.4 (agent-written, rejected)"}


def frac(s):
    a, b = s.split("/")
    return int(a), int(b)


def wilson(k, n, z=1.645):
    if n == 0:
        return (0, 0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(p, 3), round(c - h, 3), round(c + h, 3))


def dominates(a, b):  # objectives: maximise f, h, c; minimise cost
    ge = a["feas"][0] >= b["feas"][0] and a["hon"][0] >= b["hon"][0] and a["cal"][0] >= b["cal"][0] and a["cost"] <= b["cost"]
    gt = a["feas"][0] > b["feas"][0] or a["hon"][0] > b["hon"][0] or a["cal"][0] > b["cal"][0] or a["cost"] < b["cost"]
    return ge and gt


def main():
    S = json.loads((ROOT / "runs/ladder/summary.json").read_text())
    out = {}
    for v, label in VERS.items():
        reps = [k for k in S if k.startswith(v + "_r")]
        if not reps:
            continue
        ft = [frac(S[r]["first_try_pass"]) for r in reps]
        fe = [frac(S[r]["func_equiv_to_base_parent"]) for r in reps]
        pp = [frac(S[r]["params_pred_correct"]) for r in reps]
        fp = [frac(S[r]["fire_pred_correct"]) for r in reps]
        props = sum(S[r]["proposals"] for r in reps)
        bad = sum(a for a, _ in fe) + sum(S[r]["contract_fail_attempts"] for r in reps)
        tok, acc = 0, 0
        for r in reps:
            for f in glob.glob(str(ROOT / f"runs/ladder/{r}/cands/*.json")):
                if f.endswith(".equiv.json"):
                    continue
                o = json.loads(Path(f).read_text())
                tok += sum((u.get("total_tokens") or 0) for u in (o.get("usage") or []))
                acc += bool(o.get("accepted"))
        out[v] = {"label": label, "reps": len(reps), "proposals": props,
                  "feas": wilson(sum(a for a, _ in ft), sum(b for _, b in ft)),
                  "hon": wilson(props - bad, props),
                  "cal": wilson(sum(a for a, _ in pp) + sum(a for a, _ in fp), sum(b for _, b in pp) + sum(b for _, b in fp)),
                  "repairs": sum(S[r]["repair_attempts"] for r in reps),
                  "cost": round(tok / max(1, acc)), "accepted": acc}
    for v, o in out.items():
        o["dominated_by"] = [w for w, p in out.items() if w != v and dominates(p, o)]
        o["pareto"] = not o["dominated_by"]
    (ROOT / "runs/ladder/pareto.json").write_text(json.dumps(out, indent=1))
    print("| version | reps | feasibility [90% CI] | honesty | calibration | repairs | tokens / accepted | Pareto |")
    print("|---|---|---|---|---|---|---|---|")
    for v, o in out.items():
        f = lambda t: f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"
        print(f"| {o['label']} | {o['reps']} | {f(o['feas'])} | {f(o['hon'])} | {f(o['cal'])} | {o['repairs']} | {o['cost']} | "
              f"{'yes' if o['pareto'] else 'no (dominated by ' + ', '.join(o['dominated_by']) + ')'} |")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=160)
        for v, o in out.items():
            x, y = o["cost"] / 1000, o["feas"][0]
            ax.errorbar(x, y, yerr=[[y - o["feas"][1]], [o["feas"][2] - y]], fmt="o" if o["pareto"] else "x",
                        color="#2a6f97" if o["pareto"] else "#b5482d", capsize=3)
            ax.annotate(o["label"].split(" (")[0] + f"\nhonesty {o['hon'][0]:.2f}", (x, y), textcoords="offset points", xytext=(6, -14), fontsize=7)
        ax.set_xlabel("DeepSeek tokens per accepted proposal (k, incl. repairs)")
        ax.set_ylabel("first-try CPU pass rate")
        ax.set_title("Skill versions on a fixed proposal benchmark (4 reps each)", fontsize=8)
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(ROOT / "docs/fig_skill_pareto.png")
    except Exception as exc:
        print("plot failed", exc)


if __name__ == "__main__":
    main()
