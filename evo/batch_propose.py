#!/usr/bin/env python3
"""Proposal-level diagnostic batch: many DeepSeek proposals from a fixed seed set, no training.

Every attempt is logged (generation, parse, CPU check per repair round). Accepted candidates are
written as <out>/cands/<id>.json (same schema as the Skill's candidate.json + controller fields).

  python evo/batch_propose.py --out runs/diag1 --seeds i09-s0557 i03-s0335 ... --per-role 6 --workers 4
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
from llm_propose import LIB, build_messages, chat, cpu_check, parse_json  # noqa: E402

LOCK = threading.Lock()


def log(out: Path, **kw):
    with LOCK, open(out / "events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.time(), **kw}, ensure_ascii=False) + "\n")


def one(out: Path, role: str, parents: list[str], repair: int, idx: dict, temperature: float):
    pid = uuid.uuid4().hex[:12]
    cid = f"d-{role[:3]}-{pid[:6]}"
    work = out / "attempts" / cid
    work.mkdir(parents=True, exist_ok=True)
    sig_ref = {k: idx[k]["cpu_demo_v1"] for k in parents}
    msgs = build_messages(role, parents, pid, sig_ref)
    log(out, event="start", id=cid, role=role, parents=parents)
    cand, check, usage = None, None, []
    for att in range(1 + repair):
        try:
            text, u, secs = chat(msgs, temperature=temperature)
        except Exception as exc:
            log(out, event="llm_error", id=cid, attempt=att, error=str(exc)[:300])
            return
        usage.append({**u, "seconds": secs})
        (work / f"raw_{att}.txt").write_text(text)
        try:
            cand = parse_json(text)
        except Exception as exc:
            log(out, event="parse_error", id=cid, attempt=att, error=str(exc)[:200])
            msgs += [{"role": "assistant", "content": text}, {"role": "user", "content": "Return only the JSON object."}]
            cand = None
            continue
        check = cpu_check(cand.get("code", ""), work)
        log(out, event="cpu_check", id=cid, attempt=att, passed=check.get("passed"), stage=check.get("stage"),
            error_type=check.get("error_type"), error=(check.get("error") or "")[:300])
        if check.get("passed"):
            break
        msgs += [{"role": "assistant", "content": text},
                 {"role": "user", "content": f"The controller's CPU check failed at stage `{check.get('stage')}`: "
                  f"{check.get('error')}\nTraceback tail: {check.get('traceback_tail')}\nFix the code with minimal changes, "
                  "keep hypothesis and predictions unchanged, return the full JSON object again."}]
    ok = bool(cand and check and check.get("passed"))
    rec = {"id": cid, "role": role, "parents": parents, "accepted": ok, "attempts": len(usage), "usage": usage,
           "final_stage": (check or {}).get("stage"), "generator": "CS/DeepSeek-V4-Flash-FP8"}
    if ok:
        rec.update(candidate=cand, cpu_demo={"params_per_block": check.get("params_per_block"),
                                              "fire_rate": check.get("fire_rate")},
                   code_sha256=hashlib.sha256(cand["code"].encode()).hexdigest())
        (out / "cands").mkdir(exist_ok=True)
        (out / "cands" / f"{cid}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    log(out, event="done", id=cid, accepted=ok, attempts=len(usage), final_stage=rec["final_stage"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", nargs="+", required=True)
    ap.add_argument("--per-role", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--repair", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    idx = {p["key"]: p for p in json.loads((LIB / "index.json").read_text())["programs"]}
    rng = random.Random(a.seed)
    jobs = []
    pairs = list(itertools.permutations(a.seeds, 2))
    for i in range(a.per_role):
        jobs.append(("explore", [a.seeds[i % len(a.seeds)]]))
        jobs.append(("simplify", [a.seeds[(i + 2) % len(a.seeds)]]))
        jobs.append(("recombine", list(rng.choice(pairs))))
    (a.out / "plan.json").write_text(json.dumps({"seeds": a.seeds, "jobs": jobs, "args": vars(a)}, default=str, indent=1))
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(lambda j: one(a.out, j[0], j[1], a.repair, idx, a.temperature), jobs))
    print("done", len(jobs))


if __name__ == "__main__":
    main()
