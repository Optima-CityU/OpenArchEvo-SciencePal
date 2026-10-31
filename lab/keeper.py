#!/usr/bin/env python3
"""Keep lab sessions alive: when a lab thread's supervising run has ended and there is pending work,
start one new supervising turn (billable, budget-gated by the ledger). Max --max-restarts per lab."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "lab"), str(ROOT / "harness")]
from sp_client import Client, BudgetExceeded, NeedsAttention, record  # noqa: E402
from feeder import feed_dir  # noqa: E402

POINTER = (ROOT / "harness" / "prompts" / "lab_turn_pointer.md").read_text()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", action="append", required=True, help="name=thread_id[:feed]")
    ap.add_argument("--max-restarts", type=int, default=3)
    ap.add_argument("--model", default="codex/gpt-6-astra")
    a = ap.parse_args()
    labs = {}
    for spec in a.lab:
        name, rest = spec.split("=", 1)
        thread, _, feed = rest.partition(":")
        labs[name] = {"thread": thread, "feed": feed, "restarts": 0}
    c = Client()
    while True:
        for name, L in labs.items():
            try:
                runs = c.thread_runs(L["thread"])["agent_runs"]
                status = runs[0]["status"] if runs else None
            except Exception as exc:
                print(time.strftime("%H:%M:%S"), name, "poll error", exc, flush=True)
                continue
            pending = list((feed_dir(L["feed"]) / "pending").glob("*.json"))
            busy = False
            if status in ("completed", "stopped", "failed") and not pending:
                try:
                    from lab_client import Lab
                    hb = Lab(L["thread"], c).heartbeat() or {}
                    busy = bool(hb.get("running")) or bool(hb.get("queued"))
                except Exception:
                    busy = False
            if status in ("completed", "stopped", "failed") and (pending or busy) and L["restarts"] < a.max_restarts:
                try:
                    out = c.continue_thread(L["thread"], POINTER, meta={"name": f"keeper-{name}"}, model_name=a.model)
                    L["restarts"] += 1
                    print(time.strftime("%H:%M:%S"), name, "restarted session", out.get("agent_run_id"), flush=True)
                except NeedsAttention as exc:
                    if "429" in str(exc):
                        print(time.strftime("%H:%M:%S"), name, "no free slot, will retry", flush=True)
                    else:
                        print(time.strftime("%H:%M:%S"), name, "cannot restart:", exc, exc.record, flush=True)
                        L["restarts"] = a.max_restarts
                except BudgetExceeded as exc:
                    print(time.strftime("%H:%M:%S"), name, "cannot restart:", exc, flush=True)
                    L["restarts"] = a.max_restarts
        if all(L["restarts"] >= a.max_restarts for L in labs.values()):
            print("restart budget used up")
        time.sleep(60)


if __name__ == "__main__":
    main()
