#!/usr/bin/env python3
"""Sequential 'lane' of SciencePal tasks that yields slots to the GPU lab sessions.

The account allows 5 concurrent runs. Lab supervisor sessions have priority: a task is only started when
no lab session has ended while its worker still has work (the keeper needs that slot), and fewer than 5
runs are active. Each task is executed with run_task.py (one reservation, ledger-gated).

  python lane.py tasks.json      # [{"name":..., "prompt":..., "attach":[...], "expect":[...], "web_search":true}, ...]
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "harness"), str(ROOT / "lab")]
from sp_client import Client, ledger_entries  # noqa: E402

LABS = json.loads((Path(os.environ.get("OAE_LABS_JSON", ROOT / "config" / "labs.json"))).read_text())


def active_threads(c: Client) -> tuple[int, list[str]]:
    threads = []
    for e in ledger_entries():
        t = e.get("thread_id")
        if t and t not in threads:
            threads.append(t)
    active, lab_waiting = 0, []
    for t in threads[-40:]:
        try:
            runs = c.thread_runs(t)["agent_runs"]
        except Exception:
            continue
        st = runs[0]["status"] if runs else None
        if st == "running":
            active += 1
        elif t in LABS.values():
            try:
                from lab_client import Lab
                hb = Lab(t, c).heartbeat() or {}
                if hb.get("running") or hb.get("queued"):
                    lab_waiting.append(t)
            except Exception:
                pass
    return active, lab_waiting


def main():
    tasks = json.loads(Path(sys.argv[1]).read_text())
    c = Client()
    for task in tasks:
        while True:
            active, waiting = active_threads(c)
            if active < 5:
                break
            print(time.strftime("%H:%M:%S"), "waiting for slot: active", active, "labs waiting", len(waiting), flush=True)
            time.sleep(90)
        cmd = [sys.executable, str(ROOT / "harness" / "run_task.py"), "--name", task["name"],
               "--prompt", task["prompt"], "--deadline", str(task.get("deadline", 3600)), "--slot-wait", "1800"]
        if task.get("attach"):
            cmd += ["--attach", *task["attach"]]
        if task.get("expect"):
            cmd += ["--expect", *task["expect"]]
        if task.get("agent_id"):
            cmd += ["--agent-id", task["agent_id"]]
        if task.get("web_search"):
            cmd += ["--web-search"]
        print(time.strftime("%H:%M:%S"), "start", task["name"], flush=True)
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        print(time.strftime("%H:%M:%S"), "done", task["name"], r.returncode, r.stdout.strip().splitlines()[-3:], flush=True)


if __name__ == "__main__":
    main()
