#!/usr/bin/env python3
"""Continue an existing thread with one new billable turn and wait for it (records everything)."""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sp_client import Client, NeedsAttention, record
ap = argparse.ArgumentParser(); ap.add_argument("--thread", required=True); ap.add_argument("--prompt", required=True)
ap.add_argument("--name", required=True); ap.add_argument("--model"); ap.add_argument("--deadline", type=float, default=12000)
a = ap.parse_args()
out = Path(__file__).resolve().parent.parent / "runs" / "sp" / f"{a.name}-{time.strftime('%Y%m%dT%H%M%S')}"
out.mkdir(parents=True, exist_ok=True)
msg = Path(a.prompt).read_text() if Path(a.prompt).exists() else a.prompt
(out / "prompt.md").write_text(msg)
c = Client()
try:
    run = c.continue_thread(a.thread, msg, meta={"name": a.name}, model_name=a.model)
except NeedsAttention as e:
    print("NEEDS ATTENTION", e, e.record); sys.exit(3)
(out / "meta.json").write_text(json.dumps(run, indent=1)); print("run", run.get("agent_run_id"), flush=True)
st = c.wait(run["agent_run_id"], deadline_s=a.deadline, max_interval_s=120)
runs = c.thread_runs(a.thread)["agent_runs"]; r = [x for x in runs if x["id"] == run["agent_run_id"]]
(out / "status.json").write_text(json.dumps({"wait": st, "run": r[-1] if r else None}, indent=1))
record("run_finished", run_id=run["agent_run_id"], status=(r[-1] if r else st).get("status"),
       tokens_used=(r[-1] if r else {}).get("tokens_used"))
print("status", (r[-1] if r else st).get("status"), (r[-1] if r else {}).get("tokens_used"), "OUT", out)
