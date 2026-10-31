#!/usr/bin/env python3
"""Run GPU lab sandboxes whose worker only lives while an agent session is running.
For each lab: upload the lab files, start a supervising turn (TASK_supervise.md); whenever that turn has ended, start
another (billable, ledger-gated). Threads: existing thread ids, or 'new' (creates a sandbox via initiate).
  python harness/lab_supervise.py --lab lab5=7939e11b-... --lab lab7=new --max-turns 30
State in runs/labs_sup.json."""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "harness"), str(ROOT / "lab")]
from sp_client import Client  # noqa: E402
from lab_client import Lab  # noqa: E402
from lane import active_threads  # noqa: E402

FILES = {"TASK_supervise.md": ROOT / "prompts/lab/TASK_supervise.md", "TASK_start.md": ROOT / "prompts/lab/TASK_start.md", "lab_setup.sh": ROOT / "lab/lab_setup.sh",
         "lab_worker.py": ROOT / "lab/lab_worker.py", "prep_data.py": ROOT / "lab/prep_data.py",
         "wikitext2_lab.yaml": ROOT / "lab/wikitext2_lab.yaml"}
MSG = ("Lab start turn. Ignore earlier tasks in this conversation. Read /workspace/lab/TASK_start.md and follow it "
       "exactly, unattended, without asking questions.")
STATE = ROOT / "runs/labs_sup.json"


def upload(c, thread):
    box = Lab(thread, c)
    for n, p in FILES.items():
        box._write(f"/workspace/lab/{n}", p.read_bytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", action="append", required=True)
    ap.add_argument("--max-turns", type=int, default=30)
    a = ap.parse_args()
    c = Client()
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    for spec in a.lab:
        name, t = spec.split("=", 1)
        st.setdefault(name, {"thread": None if t == "new" else t, "turns": 0, "uploaded": False})
    STATE.write_text(json.dumps(st, indent=1))
    while True:
        for name, L in st.items():
            try:
                if not L["thread"]:
                    files = [(n, p.read_bytes()) for n, p in FILES.items()]
                    run = c.initiate("New lab session. The attached files belong in /workspace/lab/ (copy them there if "
                                     "they are elsewhere; find them under /workspace, excluding /proc). Then read "
                                     "/workspace/lab/TASK_start.md and follow it exactly, unattended.",
                                     agent_id=None, files=files, meta={"name": f"{name}-sup"})
                    L.update(thread=run["thread_id"], turns=1, uploaded=True, last_start=time.time())
                    print(time.strftime("%H:%M:%S"), name, "created", L["thread"], flush=True)
                    continue
                runs = c.thread_runs(L["thread"])["agent_runs"]
                act, _w = active_threads(c)
                if act >= 4 and not any(r.get('status') == 'running' for r in runs):
                    continue  # leave >= 2 slots for experiment agents (research leads)
                active = any(r.get("status") in ("running", "pending", "queued", "in_progress") for r in runs)
                if active or L["turns"] >= a.max_turns:
                    continue
                # 10-29: only start a turn when the worker heartbeat is stale (> 12 min)
                try:
                    import datetime as _dt
                    _hb = Lab(L["thread"], c).heartbeat() or {}
                    _age = time.time() - _dt.datetime.strptime(_hb.get("ts", "2000-01-01T00:00:00Z"), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=_dt.timezone.utc).timestamp()
                except Exception:
                    _age = 1e9
                if _age < 720:
                    continue
                if time.time() - L.get("last_start", 0) < 1800:
                    continue  # at most one turn per 20 min per lab (some turns end within minutes)
                if not L["uploaded"]:
                    upload(c, L["thread"]); L["uploaded"] = True
                c.continue_thread(L["thread"], MSG, meta={"name": f"{name}-sup{L['turns'] + 1}"})
                L["turns"] += 1; L["last_start"] = time.time()
                print(time.strftime("%H:%M:%S"), name, "turn", L["turns"], flush=True)
            except Exception as exc:
                print(time.strftime("%H:%M:%S"), name, "error", str(exc)[:200], flush=True)
            finally:
                STATE.write_text(json.dumps(st, indent=1))
        time.sleep(120)


if __name__ == "__main__":
    main()
