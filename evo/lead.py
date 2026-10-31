#!/usr/bin/env python3
"""V2 'research lead' agent (one SciencePal run per generation): reads a deterministic dossier of the run (all measured
results, the generation's fixed slots) plus checked archive knowledge, and writes a research direction for each slot.

Evolutionary computation keeps control of what to trust: parents and roles of the 9 slots are sampled by the island
rule (as in V1), each slot carries a MOEA/D weight vector (quality vs size), and the controller enforces parents, roles
and weights on the returned directives. The agent only supplies the content of each direction.

  python evo/lead.py dossier --state S --round R
  python evo/lead.py run     --state S --round R
  python evo/lead.py collect --state S --round R      -> S/directives_rRR.json
"""
from __future__ import annotations

import argparse
import io
import json
import random
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "evo")]
ROLES = ["explore", "recombine", "simplify"]

KNOWLEDGE = """## Checked knowledge distilled from the OpenArchEvo v1 archive (157 trained programs, earlier protocol)
An archive-analyst agent wrote mechanism cards; an independent detector re-computed their statistics. Use as priors, not facts:
- Spike-activity-modulated residual gate (a small SpikingLinear gate reads mean spike activity and scales the attention/MLP
  residual branch): best median in the archive, -6.7 PPL vs family seed (n=10, 70% improved >2 PPL; CI crosses 0).
  Mostly tried in Mamba2/MesaNet families, rarely in DeltaNet/HGRN2/GSA.
- Firing-rate regularisation: neutral on average (median ~0).
- Output gating (g_proj / FusedRMSNormGated): neutral to harmful (median +2.9).
- Removing core components wholesale (all short convolutions, norms, gates): usually neutral-to-harmful; simplify one thing at a time.
- Spike-driven modulation of beta/decay gates and extra normalisation variants: harmful on average in the archive (median +3 to +4).
- Kernel families differ: KDA-like blocks are fragile.
## Pitfalls the controller will block
- Edits that do not change behaviour (dead code, renames) are rejected by fingerprint and functional-equivalence gates.
- Every nn.Linear must receive binary spikes (spiking contract); no analog shortcuts; causal; sub-quadratic.
- The 2-epoch fitness is noisy (about 0.5-1.0 PPL between seeds); do not over-interpret single differences below 1 PPL.
"""

PROMPT = """You are the research lead of island-based evolutionary search over native spiking sequence-model blocks (programs define
`SpikingFLABlock`). Work unattended; never ask questions; no web search.

Find the attached lead_dossier.md and code.zip (under /workspace; otherwise use find, excluding /proc). Read the dossier fully;
open code from the zip only where needed.

The next generation has 9 proposal slots. Their parents, roles and preference weights are FIXED by the evolutionary rules (listed
in the dossier). A separate proposer writes the code from your direction; deterministic code then checks, fingerprints, selects
and trains (2-epoch WikiText-2 validation PPL, lower is better; params per block is the second objective).

For EACH slot write a concrete research direction (2-4 sentences): which mechanism to change in the given parent(s), why the
evidence (results table, checked knowledge) supports it, and what you expect. Respect the slot's weight: [1,0] = only quality
matters; [0,1] = make the block smaller/cheaper while keeping quality; in between = trade-off. Make the 9 directions different
from each other and from directions that already failed (see the table). Do not invent numbers: cite only numbers in the dossier.

Write /workspace/directives.json:
{"analysis": "<=150 words: what the measured results so far suggest (cite ids and numbers from the table)",
 "slots": [{"slot": 1, "direction": "..."}, ... 9 entries in the dossier's slot order]}
Validate it with python (json.load, 9 slots). Print the file content as your final message.
"""


def load(state: Path):
    return [json.loads(l) for l in (state / "population.jsonl").read_text().splitlines() if l.strip()]


def weights(i: int):
    return [round(1 - i / 16, 3), round(i / 16, 3)]  # quality-centred: size weight 0 .. 0.5


def fixed_slots(state: Path, rnd: int):
    import controller as C
    run = C.Run(state)
    cands = run.load("population.jsonl")
    sim = run.similarity()
    measured = [c for c in cands if c.fitness is not None]
    for c in cands:
        c.predicted_fitness = run.predicted_fitness(c, measured, sim)
    im = run.islands(cands, sim, rnd)
    rng = random.Random(run.cfg["seed"] * 1000 + rnd)
    out = []
    for i in range(3):
        for k in range(3):
            role = ROLES[(i + k) % 3]
            if role == "recombine":
                nonempty = [x for x in im.islands if x]
                pick = rng.sample(nonempty, min(2, len(nonempty)))
                ps = [im._select_within(x, 1)[0] for x in pick] if len(pick) == 2 else im.sample_parents(2)
            else:
                ps = im.sample_parents(1)[:1]
            out.append({"island": i, "role": role, "parents": [p.id for p in ps], "weight": weights(i * 3 + k)})
    return out


def table(pop):
    by = {c["id"]: c for c in pop}
    rows = []
    for c in sorted(pop, key=lambda c: (c.get("round") or 0, c["id"])):
        m = (c.get("measured") or {}).get("wikitext2_ppl")
        par = (c.get("parent_ids") or [None])[0]
        pm = (by.get(par, {}).get("measured") or {}).get("wikitext2_ppl")
        pp = ((c.get("meta") or {}).get("cpu_demo") or {}).get("params_per_block")
        st = "seed" if c["meta"].get("seed") else ("trained" if m is not None else ("train failed" if c["meta"].get("train_failed") else "not trained"))
        summ = (c["meta"].get("mechanism_summary") or c["meta"].get("hypothesis") or "")
        summ = (json.dumps(summ) if not isinstance(summ, str) else summ)[:220].replace("\n", " ")
        rows.append(f"| {c['id']} | {c.get('round')} | {c['meta'].get('role', 'seed')} | {','.join(c.get('parent_ids') or [])} | "
                    f"{'' if m is None else round(m, 2)} | {'' if (m is None or pm is None) else round(m - pm, 2)} | {pp} | {st} | {summ} |")
    return rows


def cmd_dossier(a):
    state = Path(a.state)
    pop = load(state)
    out = ROOT / "build" / "lead" / state.name / f"r{a.round:02d}"
    out.mkdir(parents=True, exist_ok=True)
    slots = fixed_slots(state, a.round)
    (out / "slots.json").write_text(json.dumps(slots, indent=1))
    L = [f"# Research-lead dossier, generation {a.round}\n",
         ("## Results table (all candidates; score = overfitting-aware 2-epoch fitness OAF = val + 0.5*(val - train), lower is better; it predicts 30-epoch quality better than val PPL; delta = vs first parent; params at CPU size)\n"
          if json.loads((state / "config.json").read_text()).get("fitness") == "oaf" else
          "## Results table (all candidates; PPL = 2-epoch WikiText-2 val PPL; delta = vs first parent; params at CPU size)\n"),
         "| id | gen | role | parents | val PPL | delta vs parent | params/block | status | mechanism / hypothesis |", "|---|---|---|---|---|---|---|---|---|",
         *table(pop), "", (KNOWLEDGE if json.loads((state / "config.json").read_text()).get("lead_knowledge", True)
                            else "## Knowledge\nNo archive knowledge is provided in this run: base your directions only on the results table above.\n"),
         "## Fixed slots for this generation\n"]
    for k, s in enumerate(slots):
        L.append(f"- slot {k+1}: island {s['island']}, role {s['role']}, parents {s['parents']}, weight [quality, size] = {s['weight']}")
    (out / "lead_dossier.md").write_text("\n".join(L) + "\n")
    ids = {p for s in slots for p in s["parents"]}
    best = sorted([c for c in pop if (c.get("measured") or {}).get("wikitext2_ppl")], key=lambda c: c["measured"]["wikitext2_ppl"])[:6]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for c in pop:
            if c["id"] in ids or c in best:
                z.writestr(f"{c['id']}.py", c["code"])
    (out / "code.zip").write_bytes(buf.getvalue())
    (out / "prompt.md").write_text(PROMPT)
    print("dossier", out, "bytes", (out / "lead_dossier.md").stat().st_size)


LEAD_THREADS: dict[str, str] = {}  # optional {run_name: SciencePal thread id}; or put them in runs/lead_threads.json


def cmd_run(a):
    """Reuse one existing SciencePal thread per run (its sandbox persists): upload files, continue the thread, read the output.
    Avoids creating a new sandbox per generation (the platform's sandbox service returned 503 on 10-23 evening)."""
    import time
    state = Path(a.state)
    t = LEAD_THREADS.get(state.name)
    extra = ROOT / "runs" / "lead_threads.json"  # 10-25: threads for later arms (E17)
    if not t and extra.exists():
        t = json.loads(extra.read_text()).get(state.name)
    if not t:
        return cmd_run_new(a)
    sys.path[:0] = [str(ROOT / "harness"), str(ROOT / "lab")]
    from sp_client import Client
    from lab_client import Lab
    out = ROOT / "build" / "lead" / state.name / f"r{a.round:02d}"
    c = Client()
    box = Lab(t, c)
    wd = f"/workspace/lead/r{a.round:02d}"
    for f in ("lead_dossier.md", "code.zip"):
        box._write(f"{wd}/{f}", (out / f).read_bytes())
    msg = (f"New task (ignore earlier tasks in this conversation). The files lead_dossier.md and code.zip are in {wd}/. "
           + PROMPT.replace("/workspace/directives.json", f"{wd}/directives.json").replace(
               "Find the attached lead_dossier.md and code.zip (under /workspace; otherwise use find, excluding /proc).", f"Use {wd}/."))
    run = None
    for attempt in range(60):  # 10-25: lab supervisor sessions share the account's ~5 concurrent slots; wait up to 2 h
        try:
            run = c.continue_thread(t, msg, meta={"name": f"lead_{state.name}_r{a.round:02d}"})
            break
        except Exception as exc:
            print("start failed", attempt, str(exc)[:200], flush=True)
            time.sleep(120)
    if not run:
        return
    st = c.wait(run["agent_run_id"], deadline_s=2400)
    print("lead run", st.get("status"), st.get("elapsed_s"), flush=True)
    dst = ROOT / "runs" / "sp" / f"lead_{state.name}_r{a.round:02d}-{time.strftime('%Y%m%dT%H%M%S')}"
    (dst / "files").mkdir(parents=True, exist_ok=True)
    (dst / "meta.json").write_text(json.dumps({"thread_id": t, "run_id": run["agent_run_id"], "status": st.get("status")}))
    for k in range(6):
        try:
            (dst / "files" / "directives.json").write_bytes(box.read(f"{wd}/directives.json"))
            break
        except Exception as exc:
            print("fetch retry", k, str(exc)[:100], flush=True)
            time.sleep(20)


def cmd_run_new(a):
    state = Path(a.state)
    out = ROOT / "build" / "lead" / state.name / f"r{a.round:02d}"
    tasks = [{"name": f"lead_{state.name}_r{a.round:02d}", "prompt": str(out / "prompt.md"),
              "attach": [str(out / "lead_dossier.md"), str(out / "code.zip")], "expect": ["directives.json"], "deadline": 2400}]
    (out / "lane.json").write_text(json.dumps(tasks))
    import glob as _g, time as _t
    for attempt in range(4):  # 10-30: retry when the platform returns 503 / no directives
        t0 = _t.time()
        subprocess.run([sys.executable, str(ROOT / "harness" / "lane.py"), str(out / "lane.json")], cwd=str(ROOT), check=False)
        got = [f for f in _g.glob(str(ROOT / "runs" / "sp" / f"lead_{state.name}_r{a.round:02d}-*" / "files" / "directives.json"))
               if Path(f).stat().st_mtime >= t0 - 5]
        if got:
            break
        print("lead attempt", attempt, "produced no directives; retrying in 120 s", flush=True)
        _t.sleep(120)


def cmd_collect(a):
    state = Path(a.state)
    out = ROOT / "build" / "lead" / state.name / f"r{a.round:02d}"
    slots = json.loads((out / "slots.json").read_text())
    runs = sorted((ROOT / "runs" / "sp").glob(f"lead_{state.name}_r{a.round:02d}-*"))
    got, meta, analysis = {}, {}, None
    if runs:
        f = runs[-1] / "files" / "directives.json"
        meta = json.loads((runs[-1] / "meta.json").read_text()) if (runs[-1] / "meta.json").exists() else {}
        try:
            d = json.loads(f.read_text())
            analysis = d.get("analysis")
            for k, s in enumerate(d.get("slots", [])):
                got[int(s.get("slot", k + 1)) - 1] = s.get("direction")
        except Exception as exc:
            print("bad directives", exc)
    dirs = []
    for i in range(3):
        sl = []
        for k in range(3):
            j = i * 3 + k
            s = slots[j]
            sl.append({"role": s["role"], "parents": s["parents"], "weight": s["weight"],
                       "direction": got.get(j) or "(no lead direction: follow the role)"})
        dirs.append({"island": i, "slots": sl, "avoid": [], "manager_run": meta.get("run_id"), "fallback": not got})
    (state / f"directives_r{a.round:02d}.json").write_text(json.dumps(dirs, indent=1))
    (state / f"lead_analysis_r{a.round:02d}.txt").write_text(analysis or "")
    print("directives", sum(1 for v in got.values() if v), "/ 9 from the lead; fallback" if not got else "")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dossier", "run", "collect"])
    ap.add_argument("--state", required=True)
    ap.add_argument("--round", type=int, required=True)
    a = ap.parse_args()
    {"dossier": cmd_dossier, "run": cmd_run, "collect": cmd_collect}[a.cmd](a)
