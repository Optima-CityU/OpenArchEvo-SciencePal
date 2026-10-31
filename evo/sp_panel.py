#!/usr/bin/env python3
"""Run a panel of SciencePal agents (different models, optional repeats) on the same task in their own sandboxes.

Each member gets the same files in /workspace/<exp>/<tag>/, the same prompt, and writes out.json (+ optional extra
files listed in `fetch`). Records: runs/panel/<exp>/<tag>.json (status, model actually reported by the platform,
tokens, output, fetched files). Used by E21 (hypothesis panel) and E22 (independent re-analysis).
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "harness"), str(ROOT / "lab"), str(ROOT / "evo")]

MODELS = {"gpt61": "codex/gpt-6.1-sol-high", "kimik3": "kimi-k3", "glm53": "zai/glm-5.3", "dsv4pro": "deepseek/deepseek-v4-pro",
          "dsv4flash": None}


def run_panel(exp, files, prompt, members, fetch=(), par=3, deadline_s=3600):
    from sp_client import Client
    from e8_models import threads_pool, mark_used
    from lab_client import Lab
    c = Client()
    out = ROOT / "runs/panel" / exp
    out.mkdir(parents=True, exist_ok=True)
    todo = [(tag, m) for tag, m in members if not (out / f"{tag}.json").exists()]
    import fcntl

    def take():  # cross-process: pick an unused thread under a file lock
        with open(ROOT / "runs/panel/.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            t = threads_pool()[0]
            mark_used(t)
            return t

    def go(j):
        tag, model = j
        t = take()
        wd = f"/workspace/{exp}/{tag}"
        box = Lab(t, c)
        for name, data in files.items():
            box._write(f"{wd}/{name}", data)
        run = None
        for attempt in range(60):
            try:
                run = c.continue_thread(t, prompt.format(wd=wd), meta={"name": f"{exp}_{tag}"}, model_name=model)
                break
            except Exception as exc:
                print(tag, "start failed", attempt, str(exc)[:120], flush=True)
                if "409" in str(exc) and attempt % 3 == 2:  # thread busy: move to a fresh one
                    t = take(); box = Lab(t, c)
                    for name, data in files.items():
                        box._write(f"{wd}/{name}", data)
                time.sleep(90)
        if not run:
            return
        st = c.wait(run["agent_run_id"], deadline_s=deadline_s)
        rec = {"tag": tag, "model_requested": model, "thread": t, "run_id": run["agent_run_id"], "status": st.get("status"),
               "elapsed": st.get("elapsed_s")}
        try:
            r = [x for x in c.thread_runs(t)["agent_runs"] if x["id"] == run["agent_run_id"]][0]
            rec.update(model_reported=r.get("model_name"), tokens=r.get("tokens_used"))
        except Exception:
            pass
        for name in ("out.json",) + tuple(fetch):
            for k in range(4):
                try:
                    data = box.read(f"{wd}/{name}").decode(errors="replace")
                    if name == "out.json":
                        rec["output"] = json.loads(data)
                    else:
                        rec.setdefault("files", {})[name] = data
                    break
                except Exception as exc:
                    rec.setdefault("fetch_errors", {})[name] = str(exc)[:120]
                    time.sleep(15)
        (out / f"{tag}.json").write_text(json.dumps(rec, indent=1))
        print(tag, rec.get("status"), rec.get("model_reported"), rec.get("tokens"), "out" if "output" in rec else "NO-OUT", flush=True)

    with ThreadPoolExecutor(par) as ex:
        list(ex.map(go, todo))
