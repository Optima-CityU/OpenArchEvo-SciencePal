#!/usr/bin/env python3
"""E15: Skill evolution round 2 — SciencePal agents (5 models) each propose a revised proposer SKILL.md from the
week's failure evidence; a fixed CPU benchmark decides (pre-registered in docs/PREREG_night4.md).

  python evo/skill_evo2.py pack      -> build/skill_evo2/evidence.md (+ SKILL_v033.md)
  python evo/skill_evo2.py run       -> runs/skill_evo2/<model>.json with the candidate SKILL text
  python evo/skill_evo2.py extract   -> build/skill_evo2/cand_<model>/SKILL.md
"""
import collections
import difflib
import glob
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "harness"), str(ROOT / "lab"), str(ROOT / "evo")]
B = ROOT / "build" / "skill_evo2"  # working dir; the evidence actually used is in prompts/skill_evo/round2/
OUT = ROOT / "runs/skill_evo2"
MODELS = {"gpt61": "codex/gpt-6.1-sol-high", "kimik3": "kimi-k3", "glm53": "zai/glm-5.3", "dsv4pro": "deepseek/deepseek-v4-pro",
          "dsv4flash": None}


def load(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def pack():
    B.mkdir(parents=True, exist_ok=True)
    (B / "SKILL_v033.md").write_text((ROOT / "skills" / "v0.3.3" / "SKILL.md").read_text())
    L = ["# Evidence for revising the OpenArchEvo proposer Skill (v0.3.3)", "",
         "You are improving the instructions that a DeepSeek proposer model follows when it writes one child architecture",
         "program per request (roles explore / recombine / simplify). Below is what happened to its proposals this week.", ""]
    L += [(ROOT / "docs/PROPOSAL_fate.md").read_text(), ""]
    L += ["## Where proposals failed the CPU check (stage counts, Skill v0.3.3 runs)", ""]
    c = collections.Counter()
    for d in glob.glob(str(ROOT / "runs/e10/A_s?")) + glob.glob(str(ROOT / "runs/e7/V2_s?")):
        for e in load(Path(d) / "events.jsonl"):
            if e.get("event") == "cpu_check" and not e.get("passed"):
                c[e.get("stage")] += 1
    L += [f"- {k}: {v}" for k, v in c.most_common()] + [""]
    L += ["## Examples of proposals rejected because they behave exactly like their parent (real, from Skill v0.3.3 runs)", ""]
    n = 0
    for d in sorted(glob.glob(str(ROOT / "runs/e10/A_s?"))):
        ev = load(Path(d) / "events.jsonl")
        pop = {x["id"]: x for x in load(Path(d) / "population.jsonl")}
        st = {e["id"]: e for e in ev if e.get("event") == "proposal_start"}
        for e in ev:
            if e.get("event") == "proposal_rejected" and e.get("reason") == "parent_fingerprint_match" and n < 4:
                cid = e["id"]; s = st.get(cid) or {}
                par = (s.get("parents") or [None])[0]
                f = Path(d) / "attempts" / cid / "child.py"
                if par not in pop or not f.exists():
                    continue
                diff = [l for l in difflib.unified_diff(pop[par]["code"].splitlines(), f.read_text().splitlines(), lineterm="", n=0)
                        if l.startswith(("+", "-")) and not l.startswith(("+++", "---")) and l.strip() not in "+-"
                        and not l[1:].strip().startswith("#")]
                L += [f"### {cid} (role {s.get('role')}, parent {par}): {len(diff)} changed non-comment lines", "```diff"] + diff[:30] + ["```", ""]
                n += 1
    L += ["## Other facts", "",
          "- Without per-slot research directions (arm A), 60% of v0.3.3 proposals were near-duplicates or behaved like their parent,",
          "  and one run had nothing new to train for 3 generations. With directions from a research-lead agent (V2): 41%.",
          "- Many 'simplify' children change parameters slightly but compute the same function (dead code, re-parameterisation).",
          "- The best mechanism found this week (spiking output gate on DeltaNet, confirmed with 3 seeds and a parameter-matched",
          "  control) came from 'explore'/'recombine' slots with a new live computation path.",
          "- A Skill candidate written by an agent last week (v0.4) claimed improvements in its own back-test but did worse on the",
          "  fixed benchmark (more repairs, lower first-try pass) and was rejected: longer is not better."]
    (B / "evidence.md").write_text("\n".join(L) + "\n")
    print(B / "evidence.md", (B / "evidence.md").stat().st_size)


PROMPT = """New task (ignore all earlier tasks in this conversation). Files are in {wd}/: SKILL_v033.md (the current proposer
instructions) and evidence.md (what happened to its proposals this week). Work unattended; never ask questions.

Write an improved version of the Skill as {wd}/SKILL_candidate.md (the COMPLETE file, same front-matter format with
version: 0.5.0-{tag}). Goal, in order of priority:
1. far fewer proposals that are near-duplicates or compute the same function as their parent (every child must add, remove
   or change a LIVE computation path that affects the output), without lowering the rate of proposals that pass the CPU check;
2. fewer CPU-check failures at the stages listed in evidence.md (forward / import).
Constraints: keep the contract (§1), the output format (§5), the demo mode (§6) and all script names unchanged; you may edit
§2–§4 and the appendix/checklists. The new file must not be more than 15% longer than SKILL_v033.md. Do not add facts you
cannot back with evidence.md. Then write {wd}/out.json: {{"changes": ["<each change, one line>"], "why": "<= 120 words",
"predicted_effect": "<one sentence>"}}. Print DONE when both files exist."""


def run():
    from sp_client import Client
    from e8_models import threads_pool, mark_used, run_one
    c = Client()
    OUT.mkdir(parents=True, exist_ok=True)
    pool = threads_pool()
    jobs = [(k, m, pool.pop(0)) for k, m in MODELS.items() if not (OUT / f"{k}.json").exists()]

    def go(j):
        k, m, t = j
        mark_used(t)
        wd = f"/workspace/skill_evo2/{k}"
        rec = run_one(c, t, m, {"SKILL_v033.md": (B / "SKILL_v033.md").read_bytes(), "evidence.md": (B / "evidence.md").read_bytes()},
                      PROMPT.format(wd=wd, tag=k), wd, f"skillevo2_{k}")
        if not rec:
            return
        from lab_client import Lab
        box = Lab(t, c)
        for i in range(6):
            try:
                rec["skill_text"] = box.read(f"{wd}/SKILL_candidate.md").decode(errors="replace")
                break
            except Exception as exc:
                rec["skill_fetch_error"] = str(exc)[:120]
                time.sleep(20)
        (OUT / f"{k}.json").write_text(json.dumps(rec, indent=1))
        print(k, rec.get("status"), rec.get("model_reported"), rec.get("tokens"), len(rec.get("skill_text") or ""), flush=True)

    with ThreadPoolExecutor(2) as ex:
        list(ex.map(go, jobs))


def extract():
    base = len((B / "SKILL_v033.md").read_text())
    for f in sorted(OUT.glob("*.json")):
        r = json.loads(f.read_text())
        t = r.get("skill_text") or ""
        ok = t.startswith("---") and "SpikingFLABlock" in t and len(t) <= base * 1.2
        d = B / f"cand_{f.stem}"
        d.mkdir(exist_ok=True)
        if t:
            (d / "SKILL.md").write_text(t)
        print(f.stem, r.get("model_reported"), "len", len(t), f"({len(t) / base:.2f}x)", "valid" if ok else "INVALID")


if __name__ == "__main__":
    {"pack": pack, "run": run, "extract": extract}[sys.argv[1]]()
