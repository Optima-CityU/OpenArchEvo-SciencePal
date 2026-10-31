#!/usr/bin/env python3
"""E8: do different LLMs, acting as the same research-lead agent, have different research preferences, and do they complement
and check each other? Proposal-only (no GPU).

  python evo/e8_models.py prep                 # one dossier per source run (E7 V2 states after generation 1)
  python evo/e8_models.py run [--rep N]        # each (source, model) on a separate reused SciencePal thread
  python evo/e8_models.py propose              # DeepSeek proposer + controller CPU check for every direction
  python evo/e8_models.py crosscheck           # each model reviews another model's directions (fact check)
  python evo/e8_models.py analyze              # -> runs/e8/summary.json, docs/E8_models.md, docs/fig_e8_models.png

Deterministic measurements only: mechanism categories by keyword detectors (same rules for every model), numbers cited in
the analysis checked against the dossier, CPU feasibility and parameter change of the generated code.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "evo"), str(ROOT / "harness"), str(ROOT / "lab")]
OUT = ROOT / "runs" / "e8"
MODELS = {"gpt61": "codex/gpt-6.1-sol-high", "kimik3": "kimi-k3", "glm53": "zai/glm-5.3", "dsv4pro": "deepseek/deepseek-v4-pro",
          "dsv4flash": None}  # None = platform default (reported as platform::deepseek/deepseek-v4-flash)
SOURCES = ["V2_s1", "V2_s2", "V2_s3"]

MECH = {  # keyword detectors applied to direction text (and later to code diffs); identical for every model
    "residual gate": r"residual[- ]gate|residualgate|residual gating|modulation_factor",
    "short conv": r"short ?conv|conv1d|convolution",
    "decay / beta / gate dynamics": r"\bdecay|beta|forget gate|dt_bias|logsigmoid|write[- ]gate",
    "output gating / norm": r"output gat|g_proj|fusedrmsnormgated|o_gate|rmsnorm|layernorm|normali[sz]",
    "spike-rate regularisation": r"regulari[sz]|firing[- ]rate loss|spike[- ]rate loss|spike_reg",
    "spike-driven modulation": r"spike[- ]activity|spike[- ]driven|spike[- ]count|mean spike|firing rate",
    "MLP / FFN": r"\bmlp\b|ffn|swiglu|glu|d_ffn|expansion",
    "state / memory size": r"d_state|num_slots|state dim|expand_v|expand_k|head dim|num_heads|state size",
    "remove / simplify": r"\bremov|\bdrop|delete|prune|simplif|merge|shar(e|ing)",
    "recurrence / feedback": r"feedback|recurren|closed[- ]loop|controller|homeosta",
}


def threads_pool():
    ts = []
    for m in sorted(glob.glob(str(ROOT / "runs/sp/mgr_*/meta.json"))):
        try:
            t = json.loads(Path(m).read_text()).get("thread_id")
        except Exception:
            continue
        if t and t not in ts:
            ts.append(t)
    used = set(json.loads((OUT / "used_threads.json").read_text())) if (OUT / "used_threads.json").exists() else set()
    return [t for t in ts if t not in used]


import threading
_LOCK = threading.Lock()


def mark_used(t):
    with _LOCK:
        p = OUT / "used_threads.json"
        try:
            used = json.loads(p.read_text()) if p.exists() else []
        except Exception:
            used = []
        used.append(t)
        p.write_text(json.dumps(used))


def cmd_prep(a):
    import lead
    OUT.mkdir(parents=True, exist_ok=True)
    for s in SOURCES:
        state = ROOT / "runs/e7" / s
        pop = lead.load(state)
        rnd = max((c.get("round") or 0) for c in pop) + 1
        d = OUT / "dossiers" / s
        d.mkdir(parents=True, exist_ok=True)
        slots = lead.fixed_slots(state, rnd)
        (d / "slots.json").write_text(json.dumps(slots, indent=1))
        L = [f"# Research-lead dossier, generation {rnd}\n",
             "## Results table (all candidates; PPL = 2-epoch WikiText-2 val PPL; delta = vs first parent; params at CPU size)\n",
             "| id | gen | role | parents | val PPL | delta vs parent | params/block | status | mechanism / hypothesis |",
             "|---|---|---|---|---|---|---|---|---|", *lead.table(pop), "", lead.KNOWLEDGE, "## Fixed slots for this generation\n"]
        for k, sl in enumerate(slots):
            L.append(f"- slot {k+1}: island {sl['island']}, role {sl['role']}, parents {sl['parents']}, weight [quality, size] = {sl['weight']}")
        (d / "lead_dossier.md").write_text("\n".join(L) + "\n")
        import io
        import zipfile
        ids = {p for sl in slots for p in sl["parents"]}
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for c in pop:
                if c["id"] in ids:
                    z.writestr(f"{c['id']}.py", c["code"])
        (d / "code.zip").write_bytes(buf.getvalue())
        print("dossier", s, "gen", rnd, (d / "lead_dossier.md").stat().st_size)


def run_one(c, t, model_name, files, prompt, wd, tag):
    from lab_client import Lab
    box = Lab(t, c)
    for name, data in files.items():
        box._write(f"{wd}/{name}", data)
    run = None
    for attempt in range(40):
        try:
            run = c.continue_thread(t, prompt, meta={"name": f"e8_{tag}"}, model_name=model_name)
            break
        except Exception as exc:
            print(tag, "start failed", attempt, str(exc)[:150], flush=True)
            time.sleep(90)
    if not run:
        return None
    st = c.wait(run["agent_run_id"], deadline_s=2400)
    rec = {"tag": tag, "thread": t, "run_id": run["agent_run_id"], "status": st.get("status"), "elapsed": st.get("elapsed_s")}
    try:
        r = [x for x in c.thread_runs(t)["agent_runs"] if x["id"] == run["agent_run_id"]][0]
        rec.update(model_reported=r.get("model_name"), tokens=r.get("tokens_used"))
    except Exception:
        pass
    for k in range(6):
        try:
            rec["output"] = json.loads(box.read(f"{wd}/out.json"))
            break
        except Exception as exc:
            rec["fetch_error"] = str(exc)[:120]
            time.sleep(20)
    return rec


def cmd_run(a):
    import lead
    from sp_client import Client
    c = Client()
    pool = threads_pool()
    jobs = []
    for s in SOURCES:
        d = OUT / "dossiers" / s
        for mk, mn in MODELS.items():
            tag = f"{s}_{mk}_rep{a.rep}"
            if (OUT / "leads" / f"{tag}.json").exists():
                continue
            jobs.append((s, mk, mn, tag, pool.pop(0)))
    (OUT / "leads").mkdir(parents=True, exist_ok=True)

    def go(j):
        s, mk, mn, tag, t = j
        mark_used(t)
        d = OUT / "dossiers" / s
        wd = f"/workspace/e8/{tag}"
        prompt = ("New task (ignore all earlier tasks in this conversation). The files lead_dossier.md and code.zip are in "
                  f"{wd}/. " + lead.PROMPT.replace("/workspace/directives.json", f"{wd}/out.json").replace(
                      "Find the attached lead_dossier.md and code.zip (under /workspace; otherwise use find, excluding /proc).", f"Use {wd}/."))
        rec = run_one(c, t, mn, {"lead_dossier.md": (d / "lead_dossier.md").read_bytes(), "code.zip": (d / "code.zip").read_bytes()},
                      prompt, wd, tag)
        if rec:
            rec.update(source=s, model=mk)
            (OUT / "leads" / f"{tag}.json").write_text(json.dumps(rec, indent=1))
            print(tag, rec.get("status"), rec.get("model_reported"), rec.get("tokens"), "slots", len((rec.get("output") or {}).get("slots", [])), flush=True)

    with ThreadPoolExecutor(3) as ex:  # stays under the concurrency limit together with E7's lead runs
        list(ex.map(go, jobs))


def classify(text):
    t = (text or "").lower()
    return [k for k, p in MECH.items() if re.search(p, t)]


def cmd_propose(a):
    """Every direction -> DeepSeek proposer (Skill v0.3.3) -> controller CPU check. Records feasibility and params change."""
    import os
    os.environ["SKILL_MD_PATH"] = str(ROOT / "skills" / "v0.3.3" / "SKILL.md")
    import controller as C
    from llm_propose import chat, parse_json, cpu_check
    from openarchevo.candidate import Candidate
    items = []
    for f in sorted((OUT / "leads").glob("*.json")):
        rec = json.loads(f.read_text())
        s = rec["source"]
        slots = json.loads((OUT / "dossiers" / s / "slots.json").read_text())
        pop = {c["id"]: c for c in (json.loads(l) for l in (ROOT / "runs/e7" / s / "population.jsonl").read_text().splitlines() if l.strip())}
        for k, sl in enumerate((rec.get("output") or {}).get("slots", [])[:9]):
            items.append((rec["tag"], k, slots[k], sl.get("direction", ""), pop))
    (OUT / "proposals").mkdir(exist_ok=True)

    def go(it):
        tag, k, slot, direction, pop = it
        outp = OUT / "proposals" / f"{tag}_slot{k+1}.json"
        if outp.exists():
            return
        parents = [Candidate.from_dict(pop[p]) for p in slot["parents"] if p in pop]
        for p in parents:
            if not p.meta.get("cpu_demo"):
                C._cpu_demo(p, OUT / "attempts")
        msgs = C._messages_from_cands(slot["role"], parents, f"{tag}{k}", {p.id: p.meta["cpu_demo"] for p in parents})
        w = slot["weight"]
        msgs[-1]["content"] = (f"RESEARCH-LEAD DIRECTION: {direction}\nSub-problem preference: quality weight {w[0]:.3f}, size weight {w[1]:.3f}.\n"
                               "Follow the direction unless it would violate the contract; say in `notes` how you followed it.\n\n" + msgs[-1]["content"])
        d = OUT / "attempts" / f"{tag}_slot{k+1}"
        d.mkdir(parents=True, exist_ok=True)
        cand, check, usage = None, None, []
        for att in range(3):
            try:
                text, u, secs = chat(msgs)
                usage.append(u)
                cand = parse_json(text)
            except Exception as exc:
                cand = None
                continue
            check = cpu_check(cand.get("code", ""), d)
            if check.get("passed"):
                break
            msgs += [{"role": "assistant", "content": text},
                     {"role": "user", "content": f"The controller's CPU check failed at stage `{check.get('stage')}`: {check.get('error')}\n"
                      "Fix the code with minimal changes, return the full JSON object again."}]
        ppar = (parents[0].meta.get("cpu_demo") or {}).get("params_per_block") if parents else None
        res = {"tag": tag, "slot": k + 1, "role": slot["role"], "weight": slot["weight"], "direction": direction,
               "passed": bool(check and check.get("passed")), "first_try": bool(check and check.get("passed") and att == 0),
               "params": (check or {}).get("params_per_block"), "parent_params": ppar,
               "mechanism_summary": (cand or {}).get("mechanism_summary"), "code": (cand or {}).get("code") if check and check.get("passed") else None,
               "parent_code": parents[0].code if parents else None, "usage": usage}
        outp.write_text(json.dumps(res))

    with ThreadPoolExecutor(int(a.workers)) as ex:
        list(ex.map(go, items))
    print("proposals", len(list((OUT / "proposals").glob("*.json"))))


def cmd_crosscheck(a):
    """Each model reviews the analysis + directions of another model for the same dossier and lists factual errors."""
    from sp_client import Client
    c = Client()
    pool = threads_pool()
    order = list(MODELS)
    jobs = []
    for s in SOURCES:
        for i, rev in enumerate(order):
            tgt = order[(i + 1) % len(order)]
            src = OUT / "leads" / f"{s}_{tgt}_rep0.json"
            tag = f"{s}_{rev}_reviews_{tgt}"
            if not src.exists() or (OUT / "cross" / f"{tag}.json").exists():
                continue
            jobs.append((s, rev, tgt, tag, src, pool.pop(0)))
    _run_cross(c, jobs)


def _run_cross(c, jobs):
    (OUT / "cross").mkdir(parents=True, exist_ok=True)

    def go(j):
        s, rev, tgt, tag, src, t = j
        mark_used(t)
        wd = f"/workspace/e8/{tag}"
        target = json.loads(src.read_text()).get("output") or {}
        prompt = (f"New task (ignore all earlier tasks in this conversation). In {wd}/ you find lead_dossier.md (facts) and plan.json "
                  "(another research lead's analysis and 9 slot directions for this dossier). Act as a strict reviewer. Work unattended; "
                  "no web search. Check every factual claim in plan.json against lead_dossier.md and the code in code.zip: numbers, which "
                  "mechanisms a parent already has, what the archive knowledge says. Also flag directions that ignore the slot's fixed "
                  "parents/role/weight or duplicate each other. Write " + f"{wd}/out.json" + " as {\"errors\": [{\"slot\": n or "
                  "\"analysis\", \"claim\": \"...\", \"why_wrong\": \"...\", \"evidence\": \"...\"}], \"weak_directions\": [{\"slot\": n, "
                  "\"why\": \"...\"}], \"missing_ideas\": [\"<=3 promising directions the plan missed, grounded in the dossier\"]}. "
                  "Validate it with python and print it.")
        d = OUT / "dossiers" / s
        rec = run_one(c, t, MODELS[rev], {"lead_dossier.md": (d / "lead_dossier.md").read_bytes(), "code.zip": (d / "code.zip").read_bytes(),
                                          "plan.json": json.dumps(target, indent=1).encode()}, prompt, wd, tag)
        if rec:
            rec.update(source=s, reviewer=rev, target=tgt)
            (OUT / "cross" / f"{tag}.json").write_text(json.dumps(rec, indent=1))
            print(tag, rec.get("status"), rec.get("model_reported"), "errors", len((rec.get("output") or {}).get("errors", [])), flush=True)

    with ThreadPoolExecutor(3) as ex:
        list(ex.map(go, jobs))


def cmd_panel(a):
    """Fair reviewer comparison: every model reviews the SAME two plans (Kimi's and DeepSeek-Flash's plan for V2_s3)."""
    global cmd_crosscheck_jobs
    from sp_client import Client
    c = Client()
    pool = threads_pool()
    jobs = []
    for tgt in ("kimik3", "dsv4flash"):
        for rev in MODELS:
            if rev == tgt:
                continue
            tag = f"V2_s3_{rev}_reviews_{tgt}"
            if (OUT / "cross" / f"{tag}.json").exists():
                continue
            jobs.append(("V2_s3", rev, tgt, tag, OUT / "leads" / f"V2_s3_{tgt}_rep0.json", pool.pop(0)))
    print("panel jobs", len(jobs), flush=True)
    _run_cross(c, jobs)


def numbers_in(text):
    return [float(x) for x in re.findall(r"(?<![\w.])(\d{2,6}(?:\.\d{1,3})?)(?![\w.])", text or "")]


def cmd_analyze(a):
    import numpy as np
    rows = []
    for f in sorted((OUT / "leads").glob("*.json")):
        rec = json.loads(f.read_text())
        exp = MODELS[rec["model"]] or "deepseek-v4-flash"
        if rec.get("model_reported") and exp.split("/")[-1] not in rec["model_reported"]:
            print("model fallback, excluded:", rec["tag"], rec.get("model_reported"))
            continue
        out = rec.get("output") or {}
        dossier = (OUT / "dossiers" / rec["source"] / "lead_dossier.md").read_text()
        dn = set(round(x, 2) for x in numbers_in(dossier)) | set(round(x, 1) for x in numbers_in(dossier))
        an = out.get("analysis", "")
        nums = numbers_in(an) + [x for s in out.get("slots", []) for x in numbers_in(s.get("direction", ""))]
        bad = [x for x in nums if round(x, 2) not in dn and round(x, 1) not in dn and x not in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)]
        mechs = [classify(s.get("direction", "")) for s in out.get("slots", [])]
        rows.append({"tag": rec["tag"], "model": rec["model"], "source": rec["source"], "model_reported": rec.get("model_reported"),
                     "tokens": rec.get("tokens"), "elapsed": rec.get("elapsed"), "n_slots": len(out.get("slots", [])),
                     "mechs": mechs, "numbers_cited": len(nums), "numbers_not_in_dossier": len(bad), "bad_numbers": bad[:10],
                     "words": len(" ".join(s.get("direction", "") for s in out.get("slots", [])).split())})
    props = [json.loads(f.read_text()) for f in sorted((OUT / "proposals").glob("*.json"))]
    by_model = collections.defaultdict(lambda: {"runs": 0, "mech": collections.Counter(), "tokens": [], "nums": 0, "bad": 0,
                                                "passed": 0, "first": 0, "n": 0, "shrink": 0, "words": []})
    for r in rows:
        m = by_model[r["model"]]
        m["runs"] += 1
        m["tokens"].append(r["tokens"] or 0)
        m["nums"] += r["numbers_cited"]
        m["bad"] += r["numbers_not_in_dossier"]
        m["words"].append(r["words"])
        for ms in r["mechs"]:
            for x in set(ms):
                m["mech"][x] += 1
    for p in props:
        mk = p["tag"].split("_")[2]
        m = by_model[mk]
        m["n"] += 1
        m["passed"] += p["passed"]
        m["first"] += p["first_try"]
        if p["passed"] and p.get("params") and p.get("parent_params"):
            m["shrink"] += p["params"] < 0.97 * p["parent_params"]
    # complementarity: mechanism categories covered by each model alone vs the union, per source dossier
    cover = collections.defaultdict(dict)
    for r in rows:
        cover[r["source"]][r["model"]] = set(x for ms in r["mechs"] for x in ms)
    comp = {}
    for s, d in cover.items():
        union = set().union(*d.values()) if d else set()
        comp[s] = {"union": len(union), **{k: len(v) for k, v in d.items()},
                   "unique": {k: sorted(v - set().union(*[w for kk, w in d.items() if kk != k])) for k, v in d.items()}}
    cross = []
    for f in sorted((OUT / "cross").glob("*.json")):
        rec = json.loads(f.read_text())
        o = rec.get("output") or {}
        cross.append({"reviewer": rec["reviewer"], "target": rec["target"], "source": rec["source"], "errors": len(o.get("errors", [])),
                      "weak": len(o.get("weak_directions", [])), "missing": len(o.get("missing_ideas", [])), "tokens": rec.get("tokens")})
    summary = {"rows": rows, "by_model": {k: {**v, "mech": dict(v["mech"]), "tokens_mean": float(np.mean(v["tokens"])) if v["tokens"] else None,
                                              "words_mean": float(np.mean(v["words"])) if v["words"] else None} for k, v in by_model.items()},
               "complementarity": comp, "crosscheck": cross}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=list))
    L = ["# E8: different LLMs as the same research lead (proposal-only)\n", "| model | runs | tokens / run | words of directions | numbers cited | not in dossier | proposals | CPU pass | first-try | shrink |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in summary["by_model"].items():
        L.append(f"| {k} ({MODELS[k] or 'platform default'}) | {v['runs']} | {v['tokens_mean'] and round(v['tokens_mean']/1e6, 2)}M | {v['words_mean'] and round(v['words_mean'])} | "
                 f"{v['nums']} | {v['bad']} | {v['n']} | {v['passed']} | {v['first']} | {v['shrink']} |")
    L += ["", "## Mechanism categories in the directions (count of directions mentioning each)", "",
          "| category | " + " | ".join(summary["by_model"]) + " |", "|---|" + "---|" * len(summary["by_model"])]
    for cat in MECH:
        L.append(f"| {cat} | " + " | ".join(str(v["mech"].get(cat, 0)) for v in summary["by_model"].values()) + " |")
    L += ["", "## Coverage of mechanism categories per dossier: each model vs union", ""]
    for s, d in comp.items():
        L.append(f"- {s}: union {d['union']}; " + "; ".join(f"{k} {d.get(k)}" for k in MODELS if k in d) + f"; unique {d['unique']}")
    if cross:
        L += ["", "## Cross-checks (reviewer -> target): errors flagged / weak directions / missing ideas", ""]
        for x in cross:
            L.append(f"- {x['source']}: {x['reviewer']} -> {x['target']}: {x['errors']} / {x['weak']} / {x['missing']}")
    (ROOT / "docs/E8_models.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "run", "propose", "crosscheck", "panel", "analyze"])
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    {"prep": cmd_prep, "run": cmd_run, "propose": cmd_propose, "crosscheck": cmd_crosscheck, "panel": cmd_panel, "analyze": cmd_analyze}[a.cmd](a)
