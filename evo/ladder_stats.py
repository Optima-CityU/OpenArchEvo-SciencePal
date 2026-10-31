#!/usr/bin/env python3
"""Skill-version ladder statistics (fixed proposal plan, DeepSeek generator, controller CPU checks).

Per run: first-try pass, accepted, contract-violation attempts, repairs, functional equivalence of accepted children
vs their base parent (first parent), parameter/firing prediction calibration. Writes runs/ladder/summary.json.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import os
LADDER = os.environ.get("LADDER_DIR", "runs/ladder")
SKILL_DIR = Path(os.environ.get("OAE_SKILL_DIR", ROOT / "skills" / "runtime"))
LIB = SKILL_DIR / "references" / "library" / "code"


def pred_ok(p, meas, ref):
    m, r = meas.get(p.get("metric")), ref.get(p.get("metric"))
    if m is None or r is None:
        return None
    d = p.get("direction")
    if p.get("metric") == "params_per_block":
        tol = (p.get("tolerance_rel") or 0.05) * r
    else:
        tol = p.get("tolerance_abs") or 0.01
    if d == "similar":
        return abs(m - r) <= tol
    if d == "increase":
        return m - r > tol
    if d == "decrease":
        return r - m > tol
    return None


def main(runs):
    lib_idx = {}
    for l in open(ROOT / "skills" / "v0.2" / "references" / "library.jsonl"):
        d = json.loads(l)
        if "key" in d:
            lib_idx[d["key"]] = d.get("cpu_demo_v1") or {}
    out = {}
    for run in runs:
        rd = ROOT / LADDER / run
        ev = [json.loads(l) for l in (rd / "events.jsonl").read_text().splitlines() if l.strip()]
        first = [e for e in ev if e["event"] == "cpu_check" and e["attempt"] == 0]
        contract = sum(1 for e in ev if e["event"] == "cpu_check" and e.get("stage") == "spiking_contract" and not e["passed"])
        cands = [json.loads(p.read_text()) for p in sorted((rd / "cands").glob("*.json")) if not p.name.endswith(".equiv.json")]
        acc = [c for c in cands if c.get("accepted")]
        eq, n_eq, cal = 0, 0, {"params_per_block": [0, 0], "fire_rate": [0, 0]}
        for c in acc:
            base = c["parents"][0]
            child = rd / "cands" / f"{c['id']}.py"
            child.write_text(c["candidate"]["code"])
            j = rd / "cands" / f"{c['id']}.equiv.json"
            if not j.exists():
                subprocess.run([sys.executable, str(SKILL_DIR / "scripts" / "func_equiv.py"), str(LIB / f"{base}.py"), str(child),
                                "--json", str(j)], capture_output=True, timeout=600)
            try:
                r = json.loads(j.read_text())
                n_eq += 1
                eq += bool(r.get("equivalent"))
            except Exception:
                pass
            for p in c["candidate"].get("predicted_signature") or []:
                ref = lib_idx.get(p.get("reference_parent") or base, {})
                ok = pred_ok(p, c.get("cpu_demo") or {}, ref)
                if ok is not None and p.get("metric") in cal:
                    cal[p["metric"]][0] += ok
                    cal[p["metric"]][1] += 1
        out[run] = {"proposals": len(cands), "first_try_pass": f"{sum(e['passed'] for e in first)}/{len(first)}",
                    "accepted": f"{len(acc)}/{len(cands)}", "contract_fail_attempts": contract,
                    "repair_attempts": sum(1 for e in ev if e["event"] == "cpu_check" and e["attempt"] > 0),
                    "func_equiv_to_base_parent": f"{eq}/{n_eq}",
                    "params_pred_correct": f"{cal['params_per_block'][0]}/{cal['params_per_block'][1]}",
                    "fire_pred_correct": f"{cal['fire_rate'][0]}/{cal['fire_rate'][1]}"}
        print(run, json.dumps(out[run]))
    (ROOT / LADDER / "summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(p.name for p in (ROOT / LADDER).iterdir() if (p / "DONE").exists()))
