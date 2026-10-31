#!/usr/bin/env python3
"""Sibling-duplicate metric for the fixed Skill benchmark (E19).

The fixed proposal plan asks for the same (role, parents) job several times (within a rep and across reps 0/1).
A proposer that collapses onto one idea writes children that compute the same function as each other, even when
each one differs from its parent. Per Skill: group accepted children by (role, parents) over both reps, run the
functional-equivalence test pairwise inside each group (union-find), and count

  distinct_new = number of equivalence classes in which no member is equivalent to its base parent.

usage: LADDER_DIR=runs/ladder3 python3 evo/sibling_equiv.py name1 name2 ...   (names without _r0/_r1)
"""
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LADDER = ROOT / os.environ.get("LADDER_DIR", "runs/ladder2")
FE = Path(os.environ.get("OAE_SKILL_DIR", ROOT / "skills" / "runtime")) / "scripts" / "func_equiv.py"
CACHE = LADDER / "sibling_cache"


def cands(name):
    out = []
    for rep in (0, 1):
        rd = LADDER / f"{name}_r{rep}" / "cands"
        for p in sorted(rd.glob("*.json")):
            if p.name.endswith(".equiv.json"):
                continue
            c = json.loads(p.read_text())
            if not c.get("accepted"):
                continue
            py = rd / f"{c['id']}.py"
            if not py.exists():
                py.write_text(c["candidate"]["code"])
            eq = rd / f"{c['id']}.equiv.json"
            par_eq = None
            if eq.exists():
                try:
                    par_eq = bool(json.loads(eq.read_text()).get("equivalent"))
                except Exception:
                    pass
            out.append({"key": (c.get("role") or c["id"].split("-")[1], tuple(c["parents"])), "py": py, "uid": f"r{rep}-{c['id']}",
                        "par_eq": par_eq})
    return out


def pair(a, b):
    CACHE.mkdir(parents=True, exist_ok=True)
    j = CACHE / f"{a['uid']}__{b['uid']}.json"
    if not j.exists():
        try:
            subprocess.run([sys.executable, str(FE), str(a["py"]), str(b["py"]), "--json", str(j)], capture_output=True, timeout=900)
        except subprocess.TimeoutExpired:
            pass
    try:
        return bool(json.loads(j.read_text()).get("equivalent"))
    except Exception:
        return False


def analyse(name, workers=2):
    cs = cands(name)
    groups = {}
    for c in cs:
        groups.setdefault(c["key"], []).append(c)
    jobs = [(g, a, b) for g, m in groups.items() for a, b in itertools.combinations(m, 2)]
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda t: pair(t[1], t[2]), jobs))
    parent = {c["uid"]: c["uid"] for c in cs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for (g, a, b), eq in zip(jobs, res):
        if eq:
            parent[find(a["uid"])] = find(b["uid"])
    classes = {}
    for c in cs:
        classes.setdefault(find(c["uid"]), []).append(c)
    distinct_new = sum(1 for m in classes.values() if not any(x["par_eq"] for x in m))
    return {"accepted": len(cs), "groups": len(groups), "pairs": len(jobs), "sibling_equiv_pairs": sum(res),
            "classes": len(classes), "distinct_new": distinct_new,
            "effective_new": sum(1 for c in cs if not c["par_eq"])}


if __name__ == "__main__":
    out = {}
    for n in sys.argv[1:]:
        out[n] = analyse(n)
        print(n, out[n], flush=True)
    (LADDER / "sibling_summary.json").write_text(json.dumps(out, indent=1))
