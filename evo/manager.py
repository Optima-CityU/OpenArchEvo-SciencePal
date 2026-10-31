#!/usr/bin/env python3
"""Island manager agents (SciencePal) for the controlled B2-vs-M comparison.

  python evo/manager.py dossier --state runs/evo_M --round R          # writes build/managers/R/island_*.{md,zip}
  python evo/manager.py run     --state runs/evo_M --round R          # one SciencePal run per island (slot-aware)
  python evo/manager.py collect --state runs/evo_M --round R          # -> runs/evo_M/directives_rR.json

A manager sees only its own island (members, hypotheses, measurements, previous directives and what came of
them) plus a short global leaderboard. It decides the roles, parents and research direction of the island's
3 proposal slots. Selection, training and fitness stay deterministic (controller), exactly as in arm B2.
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPT = """You are the island manager agent for island {island} of an evolutionary search over native spiking sequence-model blocks (OpenArchEvo; programs define `SpikingFLABlock`). Work unattended; never ask questions; no web search needed.

Find the attached files island_{island}.md and island_{island}_code.zip (under /workspace; otherwise use find, excluding /proc). Read island_{island}.md fully; open member code from the zip only where needed.

Your job: plan the next generation of YOUR island. The island gets exactly 3 proposal slots. A separate proposer writes the code; deterministic code then checks, fingerprints, selects (NSGA-II on predicted fitness and novelty across all islands) and trains. Lower validation PPL is better (protocol sp-wt2-sltt-s128-e2: 2-epoch WikiText-2, seed noise about 0.5-1.0 PPL). You cannot change selection or training.

For each slot choose: role (explore = one new mechanism in one parent; recombine = transplant a named mechanism from one parent into another, parents may come from the global leaderboard; simplify = remove/merge components of one parent) and parents (exact ids from the dossier), and write a concrete research direction (2-4 sentences: which mechanism, why, what you expect). Use the evidence: what improved or failed on this island, previous directives and their outcomes, contract violations, functional duplicates (edits that changed nothing), divergence. Avoid repeating exhausted directions. Keep diversity across the 3 slots unless the evidence clearly favours exploitation, and say which you chose and why.

{diversity_rule}Write /workspace/directive_island{island}.json with exactly this JSON structure:
{{"island": {island}, "strategy": "exploit|explore|mixed", "rationale": "<=120 words on the evidence you used",
  "slots": [{{"role": "explore|recombine|simplify", "parents": ["id", ...], "direction": "..."}}, x3],
  "avoid": ["short phrases of directions not to repeat"], "expected": "<=60 words: what result would show your plan worked"}}
Validate the JSON with python (json.load) and that every parent id appears in the dossier. Print the file content as your final message.
"""


DIVERSITY_RULE = """Island identity (diversity rule): the search keeps several islands so that different mechanism families are explored in
parallel. Your island's identity is its own members' families (listed in the dossier). At least 2 of your 3 slots must use a parent
from YOUR island as the base parent, and your directions must differ from the other islands' current directions (listed under
"Other islands' plans"). Use the global leaderboard only as a donor of mechanisms for at most one recombine slot. A plan in which
all islands chase the same global leader is a failure even if it looks promising.

"""


def weights(isl: int, n_isl: int = 3, per: int = 3):
    """MOEA/D weight vectors (quality, efficiency): n_isl*per evenly spaced vectors, island i owns a contiguous segment."""
    n = n_isl * per
    return [[round(1 - k / (n - 1), 3), round(k / (n - 1), 3)] for k in range(isl * per, (isl + 1) * per)]


MOEAD_RULE = """Decomposition rule (MOEA/D): the search is decomposed into 9 sub-problems, each a different trade-off between
quality (lower validation PPL) and efficiency (fewer parameters per block, cheaper per token). Your island owns 3 of them:
slot 1 -> weight {w0}, slot 2 -> weight {w1}, slot 3 -> weight {w2} (each is [quality weight, efficiency weight]).
A slot succeeds if its child improves the weighted Tchebycheff score g = max(w_q * |PPL - PPL*| / range, w_e * |params - params*| / range)
relative to its parent (lower g is better; * = best known value, see the objectives table). Plan each slot for ITS weight: a
quality-heavy slot may add capacity if justified, an efficiency-heavy slot should remove, share or shrink components without
breaking the mechanism. Neighbouring sub-problems (islands {nb}) may donate parents for a recombine slot; their plans are listed so
you do not duplicate them. Copy each slot's weight into the JSON as "weight": [w_q, w_e].

"""


def load(state: Path):
    return [json.loads(l) for l in (state / "population.jsonl").read_text().splitlines() if l.strip()]


def ppl(c):
    return (c.get("measured") or {}).get("wikitext2_ppl")


def fmt(c):
    m = c.get("meta", {})
    flags = [f for f in ("invalid", "train_failed") if m.get(f)]
    pf = c.get("predicted_fitness")
    hyp = (m.get("hypothesis") or c.get("rationale") or "")[:500].replace("\n", " ")
    return (f"- **{c['id']}** ({c.get('name') or ''}) round {c.get('round')} role={m.get('role', 'seed')} parents={c.get('parent_ids')}\n"
            f"  measured val PPL: {ppl(c) if ppl(c) is not None else 'not trained'}; predicted PPL: "
            f"{(-pf if isinstance(pf, (int, float)) and pf > -1e5 else 'n/a') if ppl(c) is None else '-'}; "
            f"cpu params/block {((m.get('cpu_demo') or {}).get('params_per_block'))}; flags {flags or 'none'}\n"
            f"  hypothesis/rationale: {hyp}\n")


def cmd_dossier(a):
    state = Path(a.state)
    pop = load(state)
    out = ROOT / "build" / "managers" / state.name / f"r{a.round:02d}"
    out.mkdir(parents=True, exist_ok=True)
    ev = [json.loads(l) for l in (state / "events.jsonl").read_text().splitlines() if l.strip()]
    leader = sorted([c for c in pop if ppl(c) is not None and not c["meta"].get("invalid")], key=ppl)[:8]
    fe = {}
    fp = ROOT / "runs" / "func_equiv" / "sweep.json"
    if fp.exists():
        for o in json.loads(fp.read_text()):
            if o.get("equivalent"):
                fe.setdefault(o["b"], []).append(o["a"])
    fixed = {}
    if a.mode == "moead_rp":
        sys.path.insert(0, str(ROOT / "evo"))
        import random as _r
        import controller as C
        run = C.Run(state)
        cands = run.load("population.jsonl")
        sim = run.similarity()
        measured = [c for c in cands if c.fitness is not None]
        for c in cands:
            c.predicted_fitness = run.predicted_fitness(c, measured, sim)
        im = run.islands(cands, sim, a.round)
        rng = _r.Random(run.cfg["seed"] * 1000 + a.round)
        roles = ["explore", "recombine", "simplify"]
        for i in range(3):
            for k in range(3):
                role = roles[(i + k) % 3]
                if role == "recombine":
                    nonempty = [x for x in im.islands if x]
                    pick = rng.sample(nonempty, min(2, len(nonempty)))
                    ps = [im._select_within(x, 1)[0] for x in pick] if len(pick) == 2 else im.sample_parents(2)
                else:
                    ps = im.sample_parents(1)[:1]
                fixed[(i, k)] = (role, [p.id for p in ps])
        (out / "fixed_parents.json").write_text(json.dumps({f"{i}_{k}": v for (i, k), v in fixed.items()}, indent=1))
    for isl in range(3):
        mem = [c for c in pop if c.get("island") == isl and c["meta"].get("alive", True)]
        hist = [c for c in pop if c.get("island") == isl and not c["meta"].get("seed")]
        prev = [e for e in ev if e.get("event") == "proposal_start" and (e.get("directive") or {}).get("island") == isl]
        by = {c["id"]: c for c in pop}
        lines = [f"# Island {isl} dossier — round {a.round}\n",
                 "Protocol: sp-wt2-sltt-s128-e2 (val PPL, lower is better; seed noise 0.5-1.0). Contract: every nn.Linear gets binary spikes; causal; sub-quadratic.\n",
                 f"## Current members ({len(mem)})\n", *map(fmt, mem),
                 "\n## All non-seed candidates ever registered on this island\n", *(map(fmt, hist) if hist else ["(none yet)\n"]),
                 "\n## Previous manager directives for this island and what came of them\n"]
        if not prev:
            lines.append("(first generation: no previous directives)\n")
        for e in prev:
            c = by.get(e["id"])
            res = "rejected before registration" if not c else (f"val PPL {ppl(c)}" if ppl(c) is not None else
                                                              ("train failed/diverged" if c["meta"].get("train_failed") else "registered, not selected for training"))
            dup = f"; functionally equivalent to {fe[e['id']]}" if e["id"] in fe else ""
            lines.append(f"- round {e['id'][1:3]} {e.get('role')} parents {e.get('parents')}: \"{(e['directive'] or {}).get('direction','')[:300]}\" -> {e['id']}: {res}{dup}\n")
        if a.mode in ("moead", "moead_div", "moead_rp"):
            allm = [c for c in pop if not c["meta"].get("invalid")]
            par = [((c["meta"].get("cpu_demo") or {}).get("params_per_block")) for c in allm]
            pv = [x for x in par if x]
            q = [ppl(c) if ppl(c) is not None else (-c["predicted_fitness"] if isinstance(c.get("predicted_fitness"), (int, float)) and c["predicted_fitness"] > -1e5 else None) for c in allm]
            qv = [x for x in q if x]
            lines.append(f"\n## Objectives table (quality = measured val PPL, else surrogate prediction; efficiency = params/block at CPU size)\n"
                         f"Ideal point: PPL* = {min(qv):.2f}, params* = {min(pv) if pv else 'n/a'}; ranges: PPL {min(qv):.2f}-{max(qv):.2f}, "
                         f"params {min(pv) if pv else 'n/a'}-{max(pv) if pv else 'n/a'}\n")
            for c, pq, pp in zip(allm, q, par):
                if c.get("island") == isl or c in leader:
                    lines.append(f"- {c['id']} (island {c.get('island')}): PPL {pq if pq is None else round(pq, 2)}{' (pred)' if ppl(c) is None else ''}, params {pp}\n")
        if a.diverse or a.mode in ("diverse", "moead", "moead_div"):
            lines.append("\n## Other islands' plans (latest directives) and members' families\n")
            for oi in range(3):
                if oi == isl:
                    continue
                fam = sorted({(c.get("parent_ids") or [c["id"]])[0] for c in pop if c.get("island") == oi and c["meta"].get("alive", True)})
                last = [e for e in ev if e.get("event") == "proposal_start" and (e.get("directive") or {}).get("island") == oi]
                dirs = "; ".join((e["directive"].get("direction") or "")[:160] for e in last[-3:]) or "(none yet)"
                lines.append(f"- island {oi}: base lineages {fam}; recent directions: {dirs}\n")
        rej = [e for e in ev if e.get("event") == "proposal_rejected"]
        lines += ["\n## Global leaderboard (all islands, measured, valid)\n",
                  *[f"- {c['id']} island {c.get('island')}: {ppl(c):.2f} — {(c['meta'].get('hypothesis') or c.get('name') or '')[:160]}\n" for c in leader],
                  f"\n## Search-wide notes\n- rejections so far: {len(rej)} (reasons: {sorted(set(e.get('reason') for e in rej))})\n"
                  "- Known pitfalls from last night: an analog (non-spiking) MLP gave the best PPL but is a contract violation; "
                  "edits that only add dead code or rename variables train to bit-identical PPL; two candidates diverged (very large PPL).\n"]
        (out / f"island_{isl}.md").write_text("".join(lines))
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            extra_ids = {p for (i, k), (_, ps) in fixed.items() if i == isl for p in ps}
            extra = [c for c in pop if c["id"] in extra_ids]
            for c in {c["id"]: c for c in mem + leader + extra}.values():
                z.writestr(f"{c['id']}.py", c["code"])
        (out / f"island_{isl}_code.zip").write_bytes(buf.getvalue())
        rule = ""
        if a.diverse or a.mode == "diverse":
            rule = DIVERSITY_RULE
        elif a.mode == "moead_rp":
            w = weights(isl)
            fx = [fixed[(isl, k)] for k in range(3)]
            rule = MOEAD_RULE.format(w0=w[0], w1=w[1], w2=w[2], nb=[j for j in (isl - 1, isl + 1) if 0 <= j < 3])
            rule += ("Parents and roles are FIXED by the evolutionary selection rule (you may not change them): "
                     + "; ".join(f"slot {k+1}: role {r}, parents {ps}" for k, (r, ps) in enumerate(fx))
                     + ". Your job is only the research direction of each slot, for its weight. The member code of these parents is in the zip.\n\n")
        elif a.mode in ("moead", "moead_div"):
            w = weights(isl)
            rule = MOEAD_RULE.format(w0=w[0], w1=w[1], w2=w[2], nb=[j for j in (isl - 1, isl + 1) if 0 <= j < 3])
            if a.mode == "moead_div":
                rule += DIVERSITY_RULE
        (out / f"prompt_{isl}.md").write_text(PROMPT.format(island=isl, diversity_rule=rule))
    print("dossiers in", out)


def cmd_run(a):
    state = Path(a.state)
    out = ROOT / "build" / "managers" / state.name / f"r{a.round:02d}"
    tasks = [{"name": f"mgr_{state.name}_r{a.round:02d}_i{i}", "prompt": str(out / f"prompt_{i}.md"),
              "attach": [str(out / f"island_{i}.md"), str(out / f"island_{i}_code.zip")],
              "expect": [f"directive_island{i}.json"], "deadline": 1800} for i in range(3)
             if (out / f"island_{i}.md").exists()]
    tf = out / "lane_tasks.json"
    tf.write_text(json.dumps(tasks, indent=1))
    subprocess.run([sys.executable, str(ROOT / "harness" / "lane.py"), str(tf)], cwd=str(ROOT), check=False)


def cmd_collect(a):
    state = Path(a.state)
    pop = {c["id"] for c in load(state)}
    dirs = []
    for i in range(3):
        runs = sorted((ROOT / "runs" / "sp").glob(f"mgr_{state.name}_r{a.round:02d}_i{i}-*"))
        d = None
        if runs:
            f = runs[-1] / "files" / f"directive_island{i}.json"
            meta = json.loads((runs[-1] / "meta.json").read_text()) if (runs[-1] / "meta.json").exists() else {}
            if f.exists():
                try:
                    d = json.loads(f.read_text())
                    d["manager_run"] = meta.get("run_id")
                    for k, sl in enumerate(d.get("slots", [])):
                        sl["parents"] = [p for p in sl.get("parents", []) if p in pop]
                        if a.mode in ("moead", "moead_div", "moead_rp") and k < 3:
                            sl["weight"] = weights(i)[k]
                        if a.mode == "moead_rp" and k < 3:
                            fp = json.loads((ROOT / "build" / "managers" / state.name / f"r{a.round:02d}" / "fixed_parents.json").read_text())[f"{i}_{k}"]
                            sl["role"], sl["parents"] = fp[0], fp[1]
                except Exception as exc:
                    print("bad directive", i, exc)
                    d = None
        if d is None:
            print("island", i, "no directive -> rule-based slots for this island")
            d = {"island": i, "slots": [], "fallback": True}
        dirs.append(d)
    (state / f"directives_r{a.round:02d}.json").write_text(json.dumps(dirs, indent=1))
    print(json.dumps([{k: d.get(k) for k in ("island", "strategy", "fallback")} | {"n_slots": len(d.get("slots", []))} for d in dirs]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dossier", "run", "collect"])
    ap.add_argument("--state", required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--diverse", action="store_true", help="diversity-aware managers (island identity rule + other islands' plans)")
    ap.add_argument("--mode", choices=["naive", "diverse", "moead", "moead_div", "moead_rp"], default="naive")
    a = ap.parse_args()
    {"dossier": cmd_dossier, "run": cmd_run, "collect": cmd_collect}[a.cmd](a)
