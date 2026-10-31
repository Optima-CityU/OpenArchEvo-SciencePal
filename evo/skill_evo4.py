#!/usr/bin/env python3
"""E19: Skill evolution round 3 (base v0.5.0, evidence from the E16 GPU loop) — SciencePal agents (5 models) each propose a revised proposer SKILL.md from the
week's failure evidence; a fixed CPU benchmark decides (pre-registered in docs/PREREG_night4.md).

  python evo/skill_evo3.py pack      -> build/skill_evo4/evidence.md (+ SKILL_v033.md)
  python evo/skill_evo3.py run       -> runs/skill_evo4/<model>.json with the candidate SKILL text
  python evo/skill_evo3.py extract   -> build/skill_evo4/cand_<model>/SKILL.md
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
B = ROOT / "build" / "skill_evo4"  # working dir; the evidence actually used is in prompts/skill_evo/round4/
OUT = ROOT / "runs/skill_evo4"
MODELS = {"gpt61": "codex/gpt-6.1-sol-high", "kimik3": "kimi-k3", "glm53": "zai/glm-5.3", "dsv4pro": "deepseek/deepseek-v4-pro",
          "dsv4flash": None}


def load(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def pack():
    B.mkdir(parents=True, exist_ok=True)
    (B / "SKILL_v051.md").write_text((ROOT / "skills" / "v0.5.1" / "SKILL.md").read_text())
    L = ["# Evidence for revising the OpenArchEvo proposer Skill (v0.5.0)", "",
         "v0.5.0 was written by an agent last round from the failures of v0.3.3 and won a fixed CPU benchmark. It was then used",
         "inside the real GPU evolution loop (arm A5: 5 runs x 6 generations x 9 proposals, no research-lead agent).",
         "Below is what happened to its proposals, next to the older arms.", ""]
    L += [(ROOT / "docs/PROPOSAL_fate.md").read_text(), ""]
    c, rej = collections.Counter(), collections.Counter()
    for d in glob.glob(str(ROOT / "runs/e16/A5_s?")):
        for e in load(Path(d) / "events.jsonl"):
            if e.get("event") == "cpu_check" and not e.get("passed"):
                c[e.get("stage")] += 1
            if e.get("event") == "proposal_rejected":
                rej[e.get("reason")] += 1
    L += ["## v0.5.0 in the GPU loop (A5): rejection reasons", ""] + [f"- {k}: {v}" for k, v in rej.most_common()] + [""]
    L += ["'near_duplicate' = the child's behaviour fingerprint (loss, firing, output statistics on probe data) is almost the",
          "same as a parent's, although the code changed. It is now the largest remaining source of waste.", ""]
    L += ["## v0.5.0 in the GPU loop: CPU-check failures by stage (attempt level, most were repaired)", ""]
    L += [f"- {k}: {v}" for k, v in c.most_common()] + [""]
    for reason, title, k in (("near_duplicate", "rejected as near-duplicates of a parent", 5),
                             ("parent_fingerprint_match", "rejected because they behave exactly like their parent", 2)):
        L += [f"## Examples of v0.5.0 proposals {title} (real, from A5 runs)", ""]
        n = 0
        for d in sorted(glob.glob(str(ROOT / "runs/e16/A5_s?"))):
            ev = load(Path(d) / "events.jsonl")
            pop = {x["id"]: x for x in load(Path(d) / "population.jsonl")}
            st = {e["id"]: e for e in ev if e.get("event") == "proposal_start"}
            for e in ev:
                if e.get("event") != "proposal_rejected" or e.get("reason") != reason or n >= k:
                    continue
                cid = e["id"]; s0 = st.get(cid) or {}
                par = (s0.get("parents") or [None])[0]
                f = Path(d) / "attempts" / cid / "child.py"
                if par not in pop or not f.exists():
                    continue
                diff = [l for l in difflib.unified_diff(pop[par]["code"].splitlines(), f.read_text().splitlines(), lineterm="", n=0)
                        if l.startswith(("+", "-")) and not l.startswith(("+++", "---")) and l.strip() not in "+-"
                        and not l[1:].strip().startswith("#")]
                sim = e.get("parent_similarity")
                L += [f"### {cid} (role {s0.get('role')}, parent {par}): {len(diff)} changed non-comment lines"
                      + (f"; similarity to parents {json.dumps({p: [round(x, 3) for x in v] for p, v in sim.items()})}" if sim else ""),
                      "```diff"] + diff[:30] + ["```", ""]
                n += 1
    L += ["## Other facts", "",
          "- The search scores children by 2-epoch WikiText-2 perplexity. A 30-epoch check showed that the best 2/6-epoch",
          "  finding (a spiking output gate on DeltaNet) learns fast early but ends 1.7 PPL worse than DeltaNet at 30 epochs,",
          "  while a parameter-free change (elu+1 feature map on q/k) was 1.1 PPL better at 30 epochs. Children that only add",
          "  capacity or speed up early learning can look good to the proxy without being better.",
          "- In the fixed benchmark, effective new proposals (accepted and not equivalent to the parent) are already 43/44 for",
          "  v0.5.0. The open problem is that repeated requests with the same parents and role tend to yield the same idea,",
          "  and in the GPU loop many children stay behaviourally too close to their parent.",
          "- Last rounds: an agent-written v0.4 was rejected (longer, more repairs); v0.5.0 won by removing parent-equivalent",
          "  children without lowering the CPU pass rate. Longer is not better."]
    (B / "evidence.md").write_text("\n".join(L) + "\n")
    print(B / "evidence.md", (B / "evidence.md").stat().st_size)


PROMPT = """New task (ignore all earlier tasks in this conversation). Files are in {wd}/: SKILL_v051.md (the current proposer
instructions, including GPU-validated experience cards C1-C6) and evidence.md (new GPU evidence from 10/29-10/30). Work unattended; never ask questions.

Write an improved version of the Skill as {wd}/SKILL_candidate.md (the COMPLETE file, same front-matter format with
version: 0.7.0-{tag}). Goal, in order of priority:
1. bring the new evidence into the proposer's guidance (update or add experience cards; state clearly what is established and
   what failed), so that proposals are more likely to be good after 30 epochs, not just after 2;
2. keep the first-try CPU pass rate and the diversity of proposals (distinct ideas for repeated parent/role requests).
Constraints: keep the contract (sec 1), the output format (sec 5), the demo mode (sec 6) and all script names unchanged. The new file must
not be more than 15% longer than SKILL_v051.md. Do not add claims that evidence.md or the existing cards do not support.
Then write {wd}/out.json: {{"changes": ["<each change, one line>"], "why": "<= 120 words",
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
        wd = f"/workspace/skill_evo4/{k}"
        rec = run_one(c, t, m, {"SKILL_v051.md": (B / "SKILL_v051.md").read_bytes(), "evidence.md": (B / "evidence.md").read_bytes()},
                      PROMPT.format(wd=wd, tag=k), wd, f"skillevo4_{k}")
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
    base = len((B / "SKILL_v051.md").read_text())
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
