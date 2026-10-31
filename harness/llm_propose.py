#!/usr/bin/env python3
"""Generate one candidate with an OpenAI-compatible chat endpoint (DeepSeek-V4-Flash in our runs) and check it on CPU.

  python llm_propose.py --role explore --parents i09-s0557 [--out runs/llm/<id>] [--repair 2]

Writes candidate.json (same schema as the SciencePal Skill), check.json and an events list.
The CPU check runs in a subprocess with a timeout and without API credentials in its env.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
SKILL = Path(os.environ.get("OAE_SKILL_DIR", ROOT / "skills" / "runtime"))  # runtime bundle: scripts/ + references/library
SKILL_MD_DEFAULT = ROOT / "skills" / "v0.7.0" / "SKILL.md"  # override with SKILL_MD_PATH
LIB = SKILL / "references" / "library"
MODEL = os.environ.get("OAE_LLM_MODEL", "deepseek-v4-flash")


def chat(messages, temperature=0.8, max_tokens=16000, timeout=600):
    base, key = os.environ["OAE_LLM_BASE_URL"], os.environ["OAE_LLM_API_KEY"]
    t0 = time.time()
    r = httpx.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {key}"},
                   json={"model": MODEL, "messages": messages, "temperature": temperature,
                         "max_tokens": max_tokens}, timeout=timeout)
    r.raise_for_status()
    d = r.json()
    return d["choices"][0]["message"]["content"], d.get("usage", {}), round(time.time() - t0, 1)


def parent_block(key: str) -> str:
    code = (LIB / "code" / f"{key}.py").read_text()
    rp = LIB / "rationale" / f"{key}.md"
    rat = rp.read_text()[:3000] if rp.exists() else "(no rationale)"
    return f"### Parent {key}\n#### Design rationale (historical, intent only)\n{rat}\n#### Code\n```python\n{code}\n```"


def build_messages(role: str, parents: list[str], pid: str, sig_ref: dict) -> list[dict]:
    skill = Path(os.environ.get("SKILL_MD_PATH", SKILL_MD_DEFAULT)).read_text()
    system = ("You are a research operator in an evolutionary search over native spiking sequence-model "
              "blocks. Follow the skill below exactly, but you cannot run commands: you output the final "
              "candidate JSON directly; the controller runs the CPU check.\n\n" + skill)
    ref = "\n".join(f"- {k}: params_per_block={v['params_per_block']}, fire_rate={v['fire_rate']:.4f} (cpu-demo-v1)"
                    for k, v in sig_ref.items())
    user = (f"ROLE: {role}\nPARENTS: {', '.join(parents)}\nPROPOSAL_ID: {pid}\nPROTOCOL: cpu-demo-v1\n\n"
            f"Reference measurements of the parents under cpu-demo-v1:\n{ref}\n\n"
            + "\n\n".join(parent_block(k) for k in parents) +
            "\n\nReturn ONLY one JSON object (no prose, no markdown fences) with the fields of SKILL.md §5. "
            "Set self_check to {\"passed\": null, \"attempts\": 0} and repair_log to []. "
            "The `code` field must contain the complete child program.")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def cpu_check(code: str, workdir: Path, timeout=240) -> dict:
    p = workdir / "child.py"
    p.write_text(code)
    env = {k: v for k, v in os.environ.items() if not any(s in k for s in ("KEY", "TOKEN", "SECRET"))}
    env["CPU_CHECK_THREADS"] = "1"
    try:
        subprocess.run([sys.executable, str(SKILL / "scripts" / "cpu_check.py"), str(p), "--json",
                        str(workdir / "check.json")], env=env, timeout=timeout, capture_output=True)
        return json.loads((workdir / "check.json").read_text())
    except subprocess.TimeoutExpired:
        return {"passed": False, "stage": "timeout"}
    except Exception as exc:
        return {"passed": False, "stage": "harness", "error": str(exc)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["explore", "recombine", "simplify"])
    ap.add_argument("--parents", nargs="+", required=True)
    ap.add_argument("--repair", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--out")
    a = ap.parse_args()
    pid = uuid.uuid4().hex[:12]
    out = Path(a.out or ROOT / "runs" / "llm" / f"{a.role}-{pid}")
    out.mkdir(parents=True, exist_ok=True)
    idx = {p["key"]: p for p in json.loads((LIB / "index.json").read_text())["programs"]}
    sig_ref = {k: idx[k]["cpu_demo_v1"] for k in a.parents}
    msgs = build_messages(a.role, a.parents, pid, sig_ref)
    events, cand, check = [], None, None
    for attempt in range(1 + a.repair):
        try:
            text, usage, secs = chat(msgs, temperature=a.temperature)
        except Exception as exc:
            events.append({"attempt": attempt, "event": "llm_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
            break
        (out / f"raw_{attempt}.txt").write_text(text)
        ev = {"attempt": attempt, "event": "llm_ok", "usage": usage, "seconds": secs}
        try:
            cand = parse_json(text)
        except Exception as exc:
            ev.update(event="parse_error", error=str(exc)[:200])
            events.append(ev)
            msgs += [{"role": "assistant", "content": text},
                     {"role": "user", "content": "That was not valid JSON. Return only the JSON object."}]
            continue
        if cand.get("proposal_id") != pid:
            ev["warning"] = "proposal_id mismatch; overwritten by controller"
            cand["proposal_id"] = pid
        check = cpu_check(cand.get("code", ""), out)
        ev.update(check_passed=check.get("passed"), check_stage=check.get("stage"),
                  check_error=(check.get("error") or "")[:300])
        events.append(ev)
        if check.get("passed"):
            break
        msgs += [{"role": "assistant", "content": text},
                 {"role": "user", "content": "The controller's CPU check failed at stage "
                  f"`{check.get('stage')}`: {check.get('error')}\nTraceback tail: {check.get('traceback_tail')}\n"
                  "Fix the code with minimal changes, keep the hypothesis and predictions unchanged, "
                  "and return the full JSON object again."}]
    result = {"proposal_id": pid, "generator": MODEL, "role": a.role, "parents": a.parents,
              "events": events, "final_check": check, "candidate": cand,
              "code_sha256": hashlib.sha256((cand or {}).get("code", "").encode()).hexdigest() if cand else None}
    (out / "result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps({"out": str(out), "passed": bool(check and check.get("passed")),
                      "attempts": len(events), "events": events}, ensure_ascii=False)[:1500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
