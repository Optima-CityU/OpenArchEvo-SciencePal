#!/usr/bin/env python3
"""Small-scale population controller for the hackathon run (protocol-agnostic core).

Uses the OpenArchEvo engine's IslandModel and SimilarityModel unchanged and adds:
role-conditioned parent sampling, a pluggable generator (DeepSeek via LiteLLM or a
SciencePal agent), CPU contract checks in a credential-free subprocess, three views
(difflib code, local bge-small rationale embedding, CPU fingerprint), and a complete
event log of every attempt, rejection, registration and selection.

State lives in --state DIR:
  archive.jsonl      measured candidates (fitness from the active training protocol)
  population.jsonl   every registered candidate (measured or not)
  events.jsonl       append-only log of everything that happened
  config.json        frozen run configuration (written by `init`)

Commands:
  init      --seeds key1 key2 ...       register seed programs from the Skill library
  propose   --n N                       generate N proposals (balanced roles) and register them
  select    --k K                       pick K unmeasured candidates for training (writes queue file)
  ingest    --results FILE.jsonl        add measured metrics, recompute fitness
  report                                print a summary
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [os.environ.get("OPENARCHEVO_DIR", str(ROOT.parent / "OpenArchEvo")), str(ROOT / "harness"), str(ROOT / "lab")]

import numpy as np  # noqa: E402

from openarchevo.candidate import Candidate  # noqa: E402
from openarchevo.inner_loop.islands import IslandModel  # noqa: E402
from openarchevo.novelty import novelty_within  # noqa: E402
from openarchevo.novelty.similarity import SimilarityModel  # noqa: E402
from openarchevo.nsga2 import nsga2_select  # noqa: E402

SKILL_DIR = Path(os.environ.get("OAE_SKILL_DIR", ROOT / "skills" / "runtime"))  # proposer Skill runtime (scripts + parent library)
LIB = SKILL_DIR / "references" / "library"
ROLES = ("explore", "recombine", "simplify")
DEFAULT_CFG = {
    "protocol_train": "cpu-micro-v1", "metric": "val_bpb", "lower_is_better": True,
    "protocol_fp": "cpu-fp-v1", "fp_args": {"seq_len": 128, "batch": 4, "structure_seq_len": 256},
    "n_islands": 3, "island_capacity": 5, "inter_island_prob": 0.5, "reset_interval_s": None,
    "selection_pressure": 1.0, "fitness_rank_weight": 0.2, "generator": "deepseek",
    "repair_attempts": 2, "seed": 0, "embedder": "BAAI/bge-small-en-v1.5",
}


class Run:
    def __init__(self, state: Path):
        self.state = state
        self.cfg = json.loads((state / "config.json").read_text()) if (state / "config.json").exists() else None

    # -------------------------------------------------------------- persistence
    def log(self, event: str, **kw):
        with open(self.state / "events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "event": event, **kw}, ensure_ascii=False) + "\n")

    def load(self, name: str) -> list[Candidate]:
        p = self.state / name
        return [Candidate.from_dict(json.loads(l)) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []

    def save(self, name: str, cands: list[Candidate]):
        tmp = self.state / (name + ".tmp")
        tmp.write_text("".join(json.dumps(c.to_dict(), ensure_ascii=False) + "\n" for c in cands))
        tmp.replace(self.state / name)

    # -------------------------------------------------------------- views
    _emb_model = None

    def embed(self, text: str) -> list[float]:
        if Run._emb_model is None:
            import torch
            torch.set_num_threads(1)
            from sentence_transformers import SentenceTransformer
            Run._emb_model = SentenceTransformer(self.cfg["embedder"], device="cpu")
        v = Run._emb_model.encode([text[:4000]], normalize_embeddings=True)[0]
        return [float(x) for x in v]

    def fingerprint(self, code: str, cand_id: str = "x") -> tuple[list[float] | None, dict]:
        if self.cfg.get("fp_backend") == "lab":
            return self._fingerprint_lab(code, cand_id)
        a = self.cfg["fp_args"]
        tmp = self.state / "tmp" / (hashlib.sha1(code.encode()).hexdigest()[:12] + ".py")
        tmp.parent.mkdir(exist_ok=True)
        tmp.write_text(code)
        env = {k: v for k, v in os.environ.items() if not any(s in k for s in ("KEY", "TOKEN", "SECRET"))}
        t0 = time.time()
        try:
            out = subprocess.run([sys.executable, str(ROOT / "evo" / "fp_cpu.py"), str(tmp), str(a["seq_len"]),
                                  str(a["batch"]), str(a["structure_seq_len"])], env=env, timeout=600,
                                 capture_output=True, text=True)
            lines = [l for l in out.stdout.splitlines() if l.startswith("{")]
            if not lines:
                return None, {"error": "no fingerprint output", "stderr_tail": (out.stderr or "")[-800:],
                              "seconds": round(time.time() - t0, 1)}
            line = lines[-1]
            d = json.loads(line)
            from openarchevo.fingerprint.schema import FEATURE_KEYS
            return [float(d[k]) for k in FEATURE_KEYS], {"seconds": round(time.time() - t0, 1)}
        except Exception as exc:
            return None, {"error": f"{type(exc).__name__}: {str(exc)[:200]}", "seconds": round(time.time() - t0, 1)}

    def _fingerprint_lab(self, code: str, cand_id: str) -> tuple[list[float] | None, dict]:
        sys.path.insert(0, str(ROOT / "lab"))
        from lab_client import Lab
        lab = Lab(self.cfg["lab_thread"])
        jid = lab.submit("fingerprint", cand_id, code, self.cfg.get("lab_fp_args", {"neuron_type": "sltt"}),
                         self.cfg["protocol_fp"], timeout_s=1800)
        res = lab.wait([jid], timeout_s=2400, poll_s=20).get(jid)
        if not res or not res.get("ok"):
            return None, {"error": (res or {}).get("error", "timeout"), "log_tail": (res or {}).get("log_tail", "")[-600:], "job": jid}
        return [float(x) for x in res["fingerprint"]], {"seconds": res.get("seconds"), "job": jid}

    # -------------------------------------------------------------- engine objects
    def similarity(self) -> SimilarityModel:
        return SimilarityModel("difflib", 1e-6, self.state / "sim_cache.json", 0)

    def islands(self, pop: list[Candidate], sim: SimilarityModel, rnd: int = 0) -> IslandModel:
        c = self.cfg
        im = IslandModel(sim, c["n_islands"], c["island_capacity"], inter_island_prob=c["inter_island_prob"],
                         selection_pressure=c["selection_pressure"], fitness_rank_weight=c["fitness_rank_weight"],
                         reset_interval_s=c["reset_interval_s"], seed=c["seed"] * 1000 + rnd)
        # restore island membership from the population file (island field), seeds clustered once at init
        for cand in pop:
            if cand.meta.get("alive", True) and cand.island is not None:
                im._add(cand, int(cand.island))
        return im

    def predicted_fitness(self, cand: Candidate, measured: list[Candidate], sim: SimilarityModel) -> float:
        """Transparent surrogate: similarity-weighted mean of measured neighbours' fitness."""
        if cand.fitness is not None and not cand.meta.get("invalid"):
            return cand.fitness
        if cand.meta.get("train_failed") or cand.meta.get("invalid"):  # failed / contract-violating: never a parent
            return -1e6
        if not measured:
            return 0.0
        s = np.array([max(1e-6, sim.similarity(cand, m)) for m in measured])
        f = np.array([m.fitness for m in measured])
        w = s ** 8
        return float((w * f).sum() / w.sum())


def fitness_from_metric(v: float, cfg: dict) -> float:
    return -float(v) if cfg["lower_is_better"] else float(v)


# ------------------------------------------------------------------ commands
def cmd_init(run: Run, a):
    run.state.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULT_CFG, **json.loads(a.cfg or "{}"))
    (run.state / "config.json").write_text(json.dumps(cfg, indent=1))
    run.cfg = cfg
    sim = run.similarity()
    seeds = []
    prefetched = {}
    if cfg.get("fp_backend") == "lab":  # submit all seed fingerprints at once, then collect
        sys.path.insert(0, str(ROOT / "lab"))
        from lab_client import Lab
        lab = Lab(cfg["lab_thread"])
        jobs = {}
        from collect import collect, find
        store = collect()
        for k in a.seeds:
            hit = find(store, k, "fingerprint", cfg["protocol_fp"], cfg.get("lab_fp_args", {}))
            if hit:
                prefetched[k] = ([float(x) for x in hit["fingerprint"]], {"seconds": hit.get("seconds"), "job": hit["job_id"], "reused": True})
                continue
            pending = [i["name"][:-5] for i in lab.ls(f"/workspace/lab/queue") + lab.ls(f"/workspace/lab/running")
                       if i["name"].startswith(f"fi-{k}-") and i["name"].endswith(".json") and "metrics" not in i["name"]]
            jobs[pending[0] if pending else lab.submit("fingerprint", k, (LIB / "code" / f"{k}.py").read_text(),
                                                         cfg.get("lab_fp_args", {}), cfg["protocol_fp"], timeout_s=1800)] = k
        res = lab.wait(list(jobs), timeout_s=3 * 3600, poll_s=60)
        for jid, k in jobs.items():
            r = res.get(jid)
            prefetched[k] = ([float(x) for x in r["fingerprint"]], {"seconds": r.get("seconds"), "job": jid}) \
                if r and r.get("ok") else (None, {"error": (r or {}).get("error", "timeout"), "job": jid})
    for k in a.seeds:
        code = (LIB / "code" / f"{k}.py").read_text()
        rp = LIB / "rationale" / f"{k}.md"
        rat = rp.read_text() if rp.exists() else ""
        fp, info = prefetched[k] if k in prefetched else run.fingerprint(code, k)
        c = Candidate(id=k, code=code, rationale=rat, name=k, fingerprint=fp, embedding=run.embed(rat or k),
                      round=0, meta={"seed": True, "fp_info": info, "alive": True})
        seeds.append(c)
        run.log("seed_registered", id=k, fp_ok=fp is not None, fp_info=info)
    im = IslandModel(sim, cfg["n_islands"], cfg["island_capacity"], seed=cfg["seed"], reset_interval_s=None)
    im.seed_islands(seeds)
    run.save("population.jsonl", seeds)
    sim.save()
    run.log("init", seeds=a.seeds, islands=im.sizes(), cfg=cfg)
    print("islands", im.sizes(), {c.id: c.island for c in seeds})


def cmd_propose(run: Run, a):
    from llm_propose import build_messages, chat, parse_json, cpu_check  # reuse the tested generator
    pop = run.load("population.jsonl")
    measured = [c for c in pop if c.fitness is not None]
    sim = run.similarity()
    for c in pop:
        c.predicted_fitness = run.predicted_fitness(c, measured, sim)
    rnd = int(a.round)
    im = run.islands(pop, sim, rnd)
    n_existing = sum(1 for c in pop if c.round == rnd)
    rng = random.Random(run.cfg["seed"] * 1000 + rnd)
    arm = run.cfg.get("arm", "B")
    for i in range(a.n):
        role = ROLES[(n_existing + i) % 3]
        if arm == "A":
            # best-only: the current measured best (ties -> seed order); recombine becomes a single-parent transplant
            ranked = sorted([c for c in pop if c.fitness is not None and not c.meta.get("invalid")], key=lambda c: -c.fitness) or pop[:1]
            parents = ranked[:1]
        elif role == "recombine":
            nonempty = [isl for isl in im.islands if isl]
            pick = rng.sample(nonempty, min(2, len(nonempty)))
            parents = [im._select_within(isl, 1)[0] for isl in pick] if len(pick) == 2 else im.sample_parents(2)
        else:
            parents = im.sample_parents(1)[:1]
        pid = uuid.uuid4().hex[:12]
        cid = f"r{rnd:02d}-{pid[:6]}"
        attempt_dir = run.state / "attempts" / cid
        attempt_dir.mkdir(parents=True, exist_ok=True)
        run.log("proposal_start", id=cid, role=role, parents=[p.id for p in parents],
                parent_islands=[p.island for p in parents], generator=run.cfg["generator"])
        sig_ref = {p.id: (p.meta.get("cpu_demo") or _cpu_demo(p, attempt_dir)) for p in parents}
        msgs = build_messages(role, [p.id for p in parents], pid, sig_ref) if all(
            (LIB / "code" / f"{p.id}.py").exists() for p in parents) else _messages_from_cands(role, parents, pid, sig_ref)
        cand, check, usage = None, None, []
        for att in range(1 + run.cfg["repair_attempts"]):
            try:
                text, u, secs = chat(msgs)
                usage.append({**u, "seconds": secs})
            except Exception as exc:
                run.log("llm_error", id=cid, attempt=att, error=str(exc)[:300])
                break
            (attempt_dir / f"raw_{att}.txt").write_text(text)
            try:
                cand = parse_json(text)
            except Exception as exc:
                run.log("parse_error", id=cid, attempt=att, error=str(exc)[:200])
                msgs += [{"role": "assistant", "content": text}, {"role": "user", "content": "Return only the JSON object."}]
                continue
            check = cpu_check(cand.get("code", ""), attempt_dir)
            run.log("cpu_check", id=cid, attempt=att, passed=check.get("passed"), stage=check.get("stage"),
                    error=(check.get("error") or "")[:300])
            if check.get("passed"):
                break
            msgs += [{"role": "assistant", "content": text},
                     {"role": "user", "content": f"The controller's CPU check failed at stage `{check.get('stage')}`: "
                      f"{check.get('error')}\nTraceback tail: {check.get('traceback_tail')}\nFix the code with minimal "
                      "changes, keep hypothesis and predictions unchanged, return the full JSON object again."}]
        if not (cand and check and check.get("passed")):
            run.log("proposal_rejected", id=cid, reason="infeasible" if cand else "no_candidate", usage=usage)
            continue
        fp, info = run.fingerprint(cand["code"], cid)
        if fp is None:
            run.log("proposal_rejected", id=cid, reason="fingerprint_failed", info=info)
            continue
        c = Candidate(id=cid, code=cand["code"], rationale=cand.get("rationale", ""), name=cand.get("name", ""),
                      fingerprint=fp, embedding=run.embed(cand.get("rationale", "") or cand.get("hypothesis", "")),
                      round=rnd, parent_ids=[p.id for p in parents],
                      meta={"role": role, "proposal_id": pid, "generator": run.cfg["generator"],
                            "hypothesis": cand.get("hypothesis"), "predicted_signature": cand.get("predicted_signature"),
                            "untestable_claims": cand.get("untestable_claims"),
                            "mechanism_summary": cand.get("mechanism_summary"),
                            "cpu_demo": {"params_per_block": check.get("params_per_block"), "fire_rate": check.get("fire_rate")},
                            "fp_info": info, "usage": usage, "alive": True,
                            "code_sha256": hashlib.sha256(cand["code"].encode()).hexdigest()})
        c.predicted_fitness = run.predicted_fitness(c, measured, sim)
        if any(np.allclose(np.asarray(fp), np.asarray(p.fingerprint), rtol=1e-6, atol=1e-8) for p in parents if p.fingerprint):
            run.log("proposal_rejected", id=cid, reason="parent_fingerprint_match")
            continue
        comps = {p.id: list(sim.components(c, p)) for p in parents}
        if arm == "A":
            target = None if sim.has_near_duplicate(c, pop) else 0
            if target is not None:
                c.island = 0
        else:
            target = im.register(c)
        if target is None:
            run.log("proposal_rejected", id=cid, reason="near_duplicate", parent_similarity=comps)
            continue
        alive_ids = {m.id for isl in im.islands for m in isl} if arm != "A" else {m.id for m in pop} | {c.id}
        pop.append(c)
        for m in pop:
            m.meta["alive"] = m.id in alive_ids or m.fitness is not None
        run.log("registered", id=cid, island=target, parent_similarity=comps, predicted_fitness=c.predicted_fitness,
                island_sizes=im.sizes(), survival_dropped=[m.id for m in pop if not m.meta["alive"]])
        run.save("population.jsonl", pop)
        sim.save()
        print(cid, role, [p.id for p in parents], "-> island", target, flush=True)


def cmd_propose_batch(run: Run, a):
    """Batch generation: sample all parents first, generate in parallel, fingerprint in one GPU batch, then register."""
    from concurrent.futures import ThreadPoolExecutor
    from llm_propose import chat, parse_json, cpu_check
    pop = run.load("population.jsonl")
    measured = [c for c in pop if c.fitness is not None]
    sim = run.similarity()
    for c in pop:
        c.predicted_fitness = run.predicted_fitness(c, measured, sim)
    rnd = int(a.round)
    im = run.islands(pop, sim, rnd)
    n_existing = sum(1 for c in pop if c.round == rnd)
    rng = random.Random(run.cfg["seed"] * 1000 + rnd + n_existing)
    arm = run.cfg.get("arm", "B")
    if getattr(a, "resume", False):
        sys.path.insert(0, str(ROOT / "lab"))
        from lab_client import Lab
        good, jobs = _resume_state(run, rnd)
        run.log("resume", round=rnd, n=len(good))
        print("resuming", len(good), "proposals", flush=True)
        _register(run, a, good, jobs, Lab(run.cfg["lab_thread"]), pop, measured, sim, im, arm, rnd)
        return
    plan = []
    directives = json.loads(Path(a.directives).read_text()) if getattr(a, "directives", None) else None
    by_id = {c.id: c for c in pop}
    dslots = []
    if directives:
        for d in directives:  # one directive per island: {"island": i, "slots": [{"role","parents","direction"}], "avoid": [...]}
            for sl in d.get("slots", []):
                dslots.append((d, sl))
    for i in range(a.n):
        role = ROLES[(n_existing + i) % 3]
        if dslots and i < len(dslots):
            d, sl = dslots[i]
            role = sl.get("role") if sl.get("role") in ROLES else role
            ps = [by_id[x] for x in sl.get("parents", []) if x in by_id and not by_id[x].meta.get("invalid")]
            parents = ps[:2] if role == "recombine" else ps[:1]
            if not parents or (role == "recombine" and len(parents) < 2):
                run.log("directive_fallback", slot=i, island=d.get("island"), reason="invalid parents", slot_spec=sl)
                parents = im.sample_parents(2 if role == "recombine" else 1)[: 2 if role == "recombine" else 1]
            pid = uuid.uuid4().hex[:12]
            cid = f"r{rnd:02d}{arm.lower()}-{pid[:6]}"
            plan.append((cid, pid, role, parents))
            run.log("proposal_start", id=cid, role=role, parents=[p.id for p in parents], parent_islands=[p.island for p in parents],
                    generator=run.cfg["generator"], arm=arm, directive={"island": d.get("island"), "direction": sl.get("direction"), "weight": sl.get("weight"),
                                                                       "avoid": d.get("avoid"), "manager_run": d.get("manager_run")})
            wtxt = ""
            if sl.get("weight"):
                w = sl["weight"]
                wtxt = (f"Sub-problem preference for this slot: quality (lower PPL) weight {w[0]:.3f}, efficiency (fewer parameters / "
                        f"less compute per token) weight {w[1]:.3f}. Design the child for this trade-off.\n")
            DIRECTIVE_TEXT[cid] = (f"ISLAND MANAGER DIRECTIVE (island {d.get('island')}): {sl.get('direction','')}\n" + wtxt +
                                   f"Avoid: {'; '.join(d.get('avoid') or []) or 'nothing specific'}\n"
                                   "Follow the directive unless it would violate the contract; say in `notes` how you followed it.")
            continue
        if arm == "A":
            ranked = sorted([c for c in pop if c.fitness is not None and not c.meta.get("invalid")], key=lambda c: -c.fitness) or pop[:1]
            parents = ranked[:1]
        elif role == "recombine":
            nonempty = [isl for isl in im.islands if isl]
            pick = rng.sample(nonempty, min(2, len(nonempty)))
            parents = [im._select_within(isl, 1)[0] for isl in pick] if len(pick) == 2 else im.sample_parents(2)
        else:
            parents = im.sample_parents(1)[:1]
        pid = uuid.uuid4().hex[:12]
        cid = f"r{rnd:02d}{arm.lower()}-{pid[:6]}"
        plan.append((cid, pid, role, parents))
        run.log("proposal_start", id=cid, role=role, parents=[p.id for p in parents],
                parent_islands=[p.island for p in parents], generator=run.cfg["generator"], arm=arm)
    for _, _, _, parents in plan:
        for p in parents:
            if not p.meta.get("cpu_demo"):
                _cpu_demo(p, run.state / "attempts")

    def gen(item):
        cid, pid, role, parents = item
        d = run.state / "attempts" / cid
        d.mkdir(parents=True, exist_ok=True)
        sig_ref = {p.id: p.meta["cpu_demo"] for p in parents}
        msgs = _messages_from_cands(role, parents, pid, sig_ref)
        if cid in DIRECTIVE_TEXT:
            msgs[-1]["content"] = DIRECTIVE_TEXT[cid] + "\n\n" + msgs[-1]["content"]
        cand, check, usage = None, None, []
        for att in range(1 + run.cfg["repair_attempts"]):
            try:
                text, u, secs = chat(msgs)
                usage.append({**u, "seconds": secs})
            except Exception as exc:
                run.log("llm_error", id=cid, attempt=att, error=str(exc)[:300])
                break
            (d / f"raw_{att}.txt").write_text(text)
            try:
                cand = parse_json(text)
            except Exception as exc:
                run.log("parse_error", id=cid, attempt=att, error=str(exc)[:200])
                msgs += [{"role": "assistant", "content": text}, {"role": "user", "content": "Return only the JSON object."}]
                cand = None
                continue
            check = cpu_check(cand.get("code", ""), d)
            run.log("cpu_check", id=cid, attempt=att, passed=check.get("passed"), stage=check.get("stage"),
                    error=(check.get("error") or "")[:300])
            if check.get("passed"):
                break
            msgs += [{"role": "assistant", "content": text},
                     {"role": "user", "content": f"The controller's CPU check failed at stage `{check.get('stage')}`: "
                      f"{check.get('error')}\nTraceback tail: {check.get('traceback_tail')}\nFix the code with minimal "
                      "changes, keep hypothesis and predictions unchanged, return the full JSON object again."}]
        ok = bool(cand and check and check.get("passed"))
        if not ok:
            run.log("proposal_rejected", id=cid, reason="infeasible" if cand else "no_candidate", usage=usage)
        return (item, cand if ok else None, check, usage)

    with ThreadPoolExecutor(int(a.workers)) as ex:
        outs = list(ex.map(gen, plan))
    good = [o for o in outs if o[1]]
    for (cid, _, _, _), _, _, usage in outs:
        run.log("llm_usage", id=cid, usage=usage)
    if run.cfg.get("fp_backend") == "cpu":  # no-GPU study: fingerprints on the controller's CPUs
        def fpc(o):
            (cid, _, _, _), cand, _, _ = o
            fp, info = run.fingerprint(cand["code"], cid)
            return cid, ({"ok": True, "fingerprint": fp, "seconds": info.get("seconds")} if fp else {"ok": False, "error": info.get("error")})
        with ThreadPoolExecutor(int(run.cfg.get("fp_par", 2))) as ex:
            res = dict(ex.map(fpc, good))
        _register(run, a, good, {}, None, pop, measured, sim, im, arm, rnd, fps=res)
        return
    # fingerprints in one batch on the GPU lab
    sys.path.insert(0, str(ROOT / "lab"))
    from lab_client import Lab
    from feeder import enqueue
    lab = Lab(run.cfg["lab_thread"])
    jobs = {}
    for (cid, pid, role, parents), cand, check, usage in good:
        jid = f"fi-{cid}-{uuid.uuid4().hex[:6]}"
        job = {"job_id": jid, "kind": "fingerprint", "program": f"programs/{cid}.py", "args": run.cfg.get("lab_fp_args", {}),
               "protocol": run.cfg["protocol_fp"], "cand_id": cid, "timeout_s": 1800}
        enqueue(job, cand["code"], feed=run.cfg.get("lab_feed", ""))
        jobs[jid] = cid
        run.log("fp_enqueued", id=cid, job=jid)
    _register(run, a, good, jobs, lab, pop, measured, sim, im, arm, rnd)


def _register(run, a, good, jobs, lab, pop, measured, sim, im, arm, rnd, fps=None):
    if fps is None:
        res = lab.wait(list(jobs), timeout_s=float(a.fp_timeout), poll_s=60) if jobs else {}
        fps = {jobs[j]: r for j, r in res.items()}
    for (cid, pid, role, parents), cand, check, usage in good:
        r = fps.get(cid)
        if not r or not r.get("ok"):
            run.log("proposal_rejected", id=cid, reason="fingerprint_failed", info=(r or {}).get("error", "timeout"))
            continue
        fp = [float(x) for x in r["fingerprint"]]
        c = Candidate(id=cid, code=cand["code"], rationale=cand.get("rationale", ""), name=cand.get("name", ""),
                      fingerprint=fp, embedding=run.embed(cand.get("rationale", "") or cand.get("hypothesis", "")),
                      round=rnd, parent_ids=[p.id for p in parents],
                      meta={"role": role, "proposal_id": pid, "generator": run.cfg["generator"], "arm": arm,
                            "hypothesis": cand.get("hypothesis"), "predicted_signature": cand.get("predicted_signature"),
                            "untestable_claims": cand.get("untestable_claims"),
                            "mechanism_summary": cand.get("mechanism_summary"),
                            "cpu_demo": {"params_per_block": check.get("params_per_block"), "fire_rate": check.get("fire_rate")},
                            "fp_info": {"seconds": r.get("seconds")}, "usage": usage, "alive": True,
                            "skill_sha": SKILL_SHA(), "directive": DIRECTIVE_TEXT.get(cid),
                            "code_sha256": hashlib.sha256(cand["code"].encode()).hexdigest()})
        c.predicted_fitness = run.predicted_fitness(c, measured, sim)
        comps = {p.id: list(sim.components(c, p)) for p in parents}
        if any(np.allclose(np.asarray(fp), np.asarray(p.fingerprint), rtol=1e-6, atol=1e-8) for p in parents if p.fingerprint):
            run.log("proposal_rejected", id=cid, reason="parent_fingerprint_match", parent_similarity=comps)
            continue
        if arm == "A":
            target = None if sim.has_near_duplicate(c, pop) else 0
            if target is not None:
                c.island = 0
        else:
            target = im.register(c)
        if target is None:
            run.log("proposal_rejected", id=cid, reason="near_duplicate", parent_similarity=comps)
            continue
        pop.append(c)
        alive_ids = {m.id for isl in im.islands for m in isl} if arm != "A" else {m.id for m in pop}
        for m in pop:
            m.meta["alive"] = m.id in alive_ids or m.fitness is not None or m.meta.get("seed", False)
        run.log("registered", id=cid, island=target, parent_similarity=comps, predicted_fitness=c.predicted_fitness,
                island_sizes=im.sizes() if arm != "A" else None)
        run.save("population.jsonl", pop)
        sim.save()
        print(cid, role, [p.id for p in parents], "-> island", target, flush=True)



def _resume_state(run, rnd):
    """Rebuild (good, jobs) for proposals of round `rnd` that were fingerprint-enqueued after the last restart but never
    registered or rejected (the controller process died while waiting for fingerprints)."""
    from llm_propose import parse_json
    ev = [json.loads(l) for l in (run.state / "events.jsonl").read_text().splitlines() if l.strip()]
    last = max([i for i, e in enumerate(ev) if e.get("event") == "restart"], default=-1)
    ev = ev[last + 1:]
    by = {c.id: c for c in run.load("population.jsonl")}
    starts = {e["id"]: e for e in ev if e.get("event") == "proposal_start" and int(e["id"][1:3]) == rnd}
    done = {e["id"] for e in ev if e.get("event") in ("registered", "proposal_rejected")}
    good, jobs = [], {}
    for e in ev:
        if e.get("event") != "fp_enqueued" or e["id"] in done or e["id"] not in starts:
            continue
        cid, st = e["id"], starts[e["id"]]
        d = run.state / "attempts" / cid
        raws = sorted(d.glob("raw_*.txt"))
        cand = None
        for r in reversed(raws):
            try:
                cand = parse_json(r.read_text())
                break
            except Exception:
                continue
        if cand is None:
            continue
        check = json.loads((d / "check.json").read_text()) if (d / "check.json").exists() else {}
        parents = [by[p] for p in st["parents"] if p in by]
        if st.get("directive"):
            dd = st["directive"]
            DIRECTIVE_TEXT[cid] = (f"ISLAND MANAGER DIRECTIVE (island {dd.get('island')}): {dd.get('direction','')}")
        good.append(((cid, cid.split("-")[1], st["role"], parents), cand, check, []))
        jobs[e["job"]] = cid
    return good, jobs


DIRECTIVE_TEXT: dict = {}


def SKILL_SHA() -> str:
    from llm_propose import SKILL, SKILL_MD_DEFAULT
    p = Path(os.environ.get("SKILL_MD_PATH", SKILL_MD_DEFAULT))
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def _cpu_demo(cand: Candidate, d: Path) -> dict:
    from llm_propose import cpu_check
    r = cpu_check(cand.code, d / f"parent_{cand.id}")  if False else cpu_check(cand.code, _mk(d / f"parent_{cand.id}"))
    cand.meta["cpu_demo"] = {"params_per_block": r.get("params_per_block"), "fire_rate": r.get("fire_rate")}
    return cand.meta["cpu_demo"]


def _mk(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _messages_from_cands(role, parents, pid, sig_ref):
    """Like llm_propose.build_messages, but for parents that are new candidates (not in the library)."""
    from llm_propose import SKILL, SKILL_MD_DEFAULT
    skill = Path(os.environ.get("SKILL_MD_PATH", SKILL_MD_DEFAULT)).read_text()
    system = ("You are a research operator in an evolutionary search over native spiking sequence-model blocks. "
              "Follow the skill below exactly, but you cannot run commands: output the final candidate JSON directly; "
              "the controller runs the CPU check.\n\n" + skill)
    ref = "\n".join(f"- {k}: params_per_block={v['params_per_block']}, fire_rate={(v['fire_rate'] or 0):.4f} (cpu-demo-v1)"
                    for k, v in sig_ref.items())
    blocks = "\n\n".join(f"### Parent {p.id}\n#### Design rationale (intent only)\n{p.rationale[:3000]}\n#### Code\n```python\n{p.code}\n```"
                         for p in parents)
    user = (f"ROLE: {role}\nPARENTS: {', '.join(p.id for p in parents)}\nPROPOSAL_ID: {pid}\nPROTOCOL: cpu-demo-v1\n\n"
            f"Reference measurements of the parents under cpu-demo-v1:\n{ref}\n\n{blocks}\n\nReturn ONLY one JSON object "
            "(no prose, no markdown fences) with the fields of SKILL.md §5. Set self_check to {\"passed\": null, "
            "\"attempts\": 0} and repair_log to []. The `code` field must contain the complete child program.")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def cmd_select(run: Run, a):
    pop = run.load("population.jsonl")
    sim = run.similarity()
    measured = [c for c in pop if c.fitness is not None]
    pending = {l.strip() for l in (run.state / "queued.txt").read_text().splitlines()} if (run.state / "queued.txt").exists() else set()
    cands = [c for c in pop if c.fitness is None and c.meta.get("alive", True) and c.id not in pending
             and not c.meta.get("seed") and not c.meta.get("invalid")]
    if not cands:
        print("nothing to select")
        return
    for c in cands:
        c.predicted_fitness = run.predicted_fitness(c, measured, sim)
    if run.cfg.get("k_from_round2") and max((c.round or 0) for c in cands) >= 2:
        a.k = int(run.cfg["k_from_round2"])  # night-4 change for GPU throughput, same for both arms
    ref = cands + measured
    nov = novelty_within(sim.matrix(ref))[: len(cands)]
    obj = np.column_stack([[c.predicted_fitness for c in cands], nov])
    if run.cfg.get("arm", "B") == "A":
        idx = list(np.argsort([-c.predicted_fitness for c in cands], kind="stable")[: min(a.k, len(cands))])
    else:
        idx = nsga2_select(obj, min(a.k, len(cands)))
    chosen = [cands[i] for i in idx]
    if run.cfg.get("fe_gate"):  # v2: functional-equivalence gate (shared weights, train+eval forward, gradients) before training
        order = list(idx) + [i for i in np.argsort([-c.predicted_fitness for c in cands], kind="stable") if i not in idx]
        by_all = {c.id: c for c in pop}
        chosen, blocked = [], []
        for i in order:
            if len(chosen) >= min(a.k, len(cands)):
                break
            c = cands[i]
            par = by_all.get((c.parent_ids or [None])[0])
            eq = _func_equiv(run, par, c) if par is not None else False
            if eq:
                blocked.append(c.id)
                c.meta["fe_equivalent_to"] = par.id
                c.meta["alive"] = False
                run.log("fe_blocked", id=c.id, parent=par.id)
            else:
                chosen.append(c)
        idx = [cands.index(c) for c in chosen]
        run.save("population.jsonl", pop)
    with open(run.state / "queued.txt", "a") as f:
        for c in chosen:
            f.write(c.id + "\n")
    q = run.state / "queue" / f"batch_{time.strftime('%Y%m%dT%H%M%S')}"
    q.mkdir(parents=True)
    for c in chosen:
        (q / f"{c.id}.py").write_text(c.code)
    run.log("selected_for_training", ids=[c.id for c in chosen], objectives=obj[idx].tolist(),
            rule="nsga2[predicted_fitness, novelty_within(candidates+measured)]", batch_dir=str(q))
    print("selected", [c.id for c in chosen], "->", q)


def _func_equiv(run, parent, child) -> bool:
    d = run.state / "fe"
    d.mkdir(exist_ok=True)
    (d / f"{parent.id}.py").write_text(parent.code)
    (d / f"{child.id}.py").write_text(child.code)
    j = d / f"{parent.id}__{child.id}.json"
    try:
        subprocess.run([sys.executable, str(SKILL_DIR / "scripts" / "func_equiv.py"), str(d / f"{parent.id}.py"), str(d / f"{child.id}.py"),
                        "--json", str(j)], capture_output=True, timeout=900)
        return bool(json.loads(j.read_text()).get("equivalent"))
    except Exception:
        return False


def cmd_ingest(run: Run, a):
    pop = run.load("population.jsonl")
    by = {c.id: c for c in pop}
    for line in Path(a.results).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        c = by.get(r["id"])
        if not c:
            continue
        v = r.get(run.cfg["metric"])
        if v is not None and (not np.isfinite(float(v)) or float(v) > 1e4):
            r = {**r, "ok": False, "error": f"diverged ({run.cfg['metric']}={v})"}
        if r.get("ok") and r.get(run.cfg["metric"]) is not None:
            c.measured[run.cfg["metric"]] = float(r[run.cfg["metric"]])
            c.fitness = fitness_from_metric(r[run.cfg["metric"]], run.cfg)
            c.meta["train"] = {k: r.get(k) for k in ("protocol", "seconds", "val_fire_rate", "params", "seed")}
            run.log("measured", id=c.id, metric=run.cfg["metric"], value=r[run.cfg["metric"]], protocol=r.get("protocol"))
        else:
            c.meta["train_failed"] = r.get("error")
            run.log("train_failed", id=c.id, error=r.get("error"))
    run.save("population.jsonl", pop)


def cmd_train_lab(run: Run, a):
    """Submit training jobs to the SciencePal lab for given ids (or the latest selection) and ingest results."""
    sys.path.insert(0, str(ROOT / "lab"))
    from lab_client import Lab
    pop = run.load("population.jsonl")
    by = {c.id: c for c in pop}
    ids = a.ids or [l.strip() for l in (run.state / "queued.txt").read_text().splitlines() if l.strip()
                    and by.get(l.strip()) is not None and by[l.strip()].fitness is None
                    and not by[l.strip()].meta.get("train_failed") and not by[l.strip()].meta.get("train_submitted")]
    lab = Lab(run.cfg["lab_thread"]) if run.cfg.get("lab_thread") else None
    args = dict(run.cfg.get("lab_train_args", {"neuron_type": "sltt", "epochs": 3, "seed": 0}))
    if a.seed is not None:
        args["seed"] = a.seed
    from feeder import enqueue
    jobs = {}
    if getattr(a, "reattach", False):  # container restart: wait again on jobs already submitted
        jobs = {c.meta["train_submitted"]: c.id for c in pop if c.fitness is None and c.meta.get("train_submitted")
                and not c.meta.get("train_failed") and not c.measured}
        ids = []
        run.log("train_reattach", jobs=jobs)
    for cid in ids:
        jid = f"tr-{cid}-{uuid.uuid4().hex[:6]}"
        enqueue({"job_id": jid, "kind": "train", "program": f"programs/{cid}.py", "args": args,
                 "protocol": run.cfg["protocol_train"], "cand_id": cid, "timeout_s": int(a.timeout)},
                by[cid].code, feed=run.cfg.get("lab_pool") or run.cfg.get("lab_feed", ""))
        jobs[jid] = cid
        by[cid].meta["train_submitted"] = jid
        run.log("train_submitted", id=cid, job=jid, args=args, protocol=run.cfg["protocol_train"])
    run.save("population.jsonl", pop)
    if a.no_wait:
        print("submitted", jobs)
        return
    if run.cfg.get("lab_pool"):  # jobs may land on any lab: wait on the local store (kept fresh by the collector daemon)
        t0, res = time.time(), {}
        while time.time() - t0 < a.timeout + 600:
            try:
                store = json.loads((ROOT / "runs" / "lab_store.json").read_text())
            except Exception:
                store = {}
            res = {j: store[j] for j in jobs if j in store}
            if len(res) == len(jobs):
                break
            time.sleep(45)
    else:
        res = lab.wait(list(jobs), timeout_s=a.timeout + 600, poll_s=60)
    lines = []
    oaf_mode = run.cfg.get("fitness") == "oaf"
    if oaf_mode:  # E40: overfitting-aware fitness from the job log (epoch-2 train/val PPL)
        import re as _re
        _labs = json.loads((Path(os.environ.get("OAE_LABS_JSON", ROOT / "config" / "labs.json"))).read_text())
        def _oaf(jid, r, val):
            for _ in range(4):
                try:
                    txt = Lab(_labs[r["lab"]]).read(f"/workspace/lab/logs/{jid}.log").decode(errors="replace")
                    ep = {int(mm.group(1)): (float(mm.group(2)), float(mm.group(3))) for mm in
                          _re.finditer(r"epoch (\d+) \| train ppl ([\d.]+) \| val ppl ([\d.]+)", txt)}
                    t2, v2 = ep[max(ep)]
                    return v2 + 0.5 * (v2 - t2), t2, v2
                except Exception:
                    time.sleep(20)
            return None
    for jid, cid in jobs.items():
        r = res.get(jid)
        if r and r.get("ok"):
            m = r["metrics"]
            val = m.get("wikitext2_ppl") if isinstance(m, dict) else None
            if val is None and isinstance(m, dict):
                val = (m.get("metrics") or {}).get("wikitext2_ppl")
            if oaf_mode and val is not None:
                o = _oaf(jid, r, val)
                if o is None:
                    run.log("oaf_failed", id=cid, job=jid)
                    lines.append({"id": cid, "ok": False, "error": "oaf: log unavailable"})
                    continue
                run.log("oaf", id=cid, job=jid, val2=o[2], train2=o[1], oaf=o[0], raw_val=val)
                val = o[0]
            lines.append({"id": cid, "ok": val is not None, "wikitext2_ppl": val, "protocol": run.cfg["protocol_train"],
                          "seconds": r.get("seconds"), "seed": args.get("seed"), "raw": m})
        else:
            lines.append({"id": cid, "ok": False, "error": (r or {}).get("error", "timeout"), "log_tail": (r or {}).get("log_tail")})
    rp = run.state / f"results_{time.strftime('%Y%m%dT%H%M%S')}.jsonl"
    rp.write_text("".join(json.dumps(l) + "\n" for l in lines))
    a.results = str(rp)
    cmd_ingest(run, a)
    print(json.dumps([{k: l.get(k) for k in ("id", "ok", "wikitext2_ppl", "seconds", "error")} for l in lines], indent=1))


def cmd_report(run: Run, a):
    pop = run.load("population.jsonl")
    ev = [json.loads(l) for l in (run.state / "events.jsonl").read_text().splitlines()]
    from collections import Counter
    print("events", Counter(e["event"] for e in ev))
    print("rejections", Counter(e.get("reason") for e in ev if e["event"] == "proposal_rejected"))
    for c in sorted(pop, key=lambda c: (c.fitness is None, -(c.fitness or 0))):
        print(f"{c.id:16s} r{c.round} isl={c.island} role={c.meta.get('role','seed'):9s} parents={c.parent_ids} "
              f"measured={c.measured} pred={c.predicted_fitness} alive={c.meta.get('alive')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path, required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init"); p.add_argument("--seeds", nargs="+", required=True); p.add_argument("--cfg")
    p = sub.add_parser("propose"); p.add_argument("--n", type=int, default=3); p.add_argument("--round", type=int, required=True)
    p = sub.add_parser("propose_batch"); p.add_argument("--n", type=int, default=9); p.add_argument("--round", type=int, required=True)
    p.add_argument("--directives")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--workers", type=int, default=4); p.add_argument("--fp-timeout", type=float, default=7200)
    p = sub.add_parser("select"); p.add_argument("--k", type=int, default=3)
    p = sub.add_parser("ingest"); p.add_argument("--results", required=True)
    p = sub.add_parser("train_lab"); p.add_argument("--ids", nargs="*"); p.add_argument("--seed", type=int)
    p.add_argument("--timeout", type=float, default=3 * 3600); p.add_argument("--no-wait", action="store_true"); p.add_argument("--reattach", action="store_true")
    sub.add_parser("report")
    a = ap.parse_args()
    run = Run(a.state)
    {"init": cmd_init, "propose": cmd_propose, "select": cmd_select, "ingest": cmd_ingest, "report": cmd_report,
     "train_lab": cmd_train_lab, "propose_batch": cmd_propose_batch}[a.cmd](run, a)


if __name__ == "__main__":
    main()
