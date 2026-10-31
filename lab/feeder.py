#!/usr/bin/env python3
"""Controller-side feeder: keeps the sandbox queue small so jobs respect the 4 GiB RAM limit.

Rules: a training job runs alone; fingerprint jobs run at most FP_PAR at a time.
Pending jobs live locally in runs/feed/pending/*.json (written by enqueue()); the feeder pushes
one job at a time into /workspace/lab/queue/ when capacity allows. Runs until --until-empty or killed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab_client import Lab  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FEED = ROOT / "runs" / "feed"


def feed_dir(name: str = "") -> Path:
    d = ROOT / "runs" / (f"feed_{name}" if name else "feed")
    (d / "pending").mkdir(parents=True, exist_ok=True)
    (d / "pushed").mkdir(parents=True, exist_ok=True)
    return d


def enqueue(job: dict, code: str | None = None, feed: str = "") -> None:
    """Add a job locally; `code` (program source) is uploaded when the job is pushed."""
    rec = {"job": job, "code": code, "t": time.time()}
    (feed_dir(feed) / "pending" / f"{time.time():.6f}_{job['job_id']}.json").write_text(json.dumps(rec))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thread", required=True)
    ap.add_argument("--fp-par", type=int, default=2)
    ap.add_argument("--until-empty", action="store_true")
    ap.add_argument("--feed", default="")
    ap.add_argument("--max-train", type=int, default=1, help="trainings allowed at once in this sandbox (multi-GPU sandboxes)")
    ap.add_argument("--pool", default="", help="shared pending pool (feed name) that every lab draws from when idle")
    a = ap.parse_args()
    FEED = feed_dir(a.feed)
    lab = Lab(a.thread)
    while True:
        # 10-27: a lab whose worker heartbeat is stale gets no new jobs; its unpushed jobs go back to the pool
        try:
            _hb = lab.heartbeat() or {}
            _age = time.time() - time.mktime(time.strptime(_hb.get("ts", "1970-01-01T00:00:00Z"), "%Y-%m-%dT%H:%M:%SZ")) + time.timezone
        except Exception:
            _hb, _age = {}, 0
        if a.pool and _age > 600:
            for pf in (FEED / "pending").glob("*.json"):
                try:
                    pf.rename(feed_dir(a.pool) / "pending" / pf.name)
                except Exception:
                    pass
            if time.time() - globals().get("_last_stale_note", 0) > 1800:
                print(time.strftime("%H:%M:%S"), f"worker heartbeat stale ({_age/60:.0f} min): not taking jobs", flush=True)
                globals()["_last_stale_note"] = time.time()
            time.sleep(120)
            continue
        pend = sorted((FEED / "pending").glob("*.json"))
        if a.pool and [r for r in (_hb.get("running") or []) if r.startswith("tr-")]:
            # 10-27: busy with a training -> do not hold claimed jobs (30-epoch jobs block a lab for hours)
            for pf in pend:
                try:
                    pf.rename(feed_dir(a.pool) / "pending" / pf.name)
                except Exception:
                    pass
            time.sleep(60)
            continue
        if a.pool and not pend:
            import os
            for pf in sorted((feed_dir(a.pool) / "pending").glob("*.json")):
                try:  # atomic claim: move the pool job into this lab's own pending dir
                    dst = FEED / "pending" / pf.name
                    os.rename(pf, dst)
                    pend = [dst]
                    break
                except FileNotFoundError:
                    continue
        if not pend and a.until_empty:
            break
        try:
            hb = lab.heartbeat() or {}
            queued = [i["name"] for i in lab.ls("/workspace/lab/queue") if i["name"].endswith(".json")]
        except Exception as exc:
            print("poll error", exc, flush=True)
            time.sleep(30)
            continue
        running = hb.get("running", [])
        # recover jobs orphaned by a worker restart (job file left in running/ without a result)
        try:
            hb_age = time.time() - time.mktime(time.strptime(hb.get("ts", "1970-01-01T00:00:00Z"), "%Y-%m-%dT%H:%M:%SZ")) + time.timezone
            if hb_age < 90:
                results = {i["name"][:-5] for i in lab.ls("/workspace/lab/results")}
                for i in lab.ls("/workspace/lab/running"):
                    n = i["name"]
                    try:
                        mt = time.mktime(time.strptime(i["mod_time"][:19], "%Y-%m-%d %H:%M:%S")) - time.timezone
                    except Exception:
                        mt = time.time()
                    if time.time() - mt < 120:
                        continue
                    if n.endswith(".json") and ".metrics" not in n and n[:-5] not in running and n[:-5] not in results:
                        job = json.loads(lab.read(i["path"]))
                        lab.c._c.delete(f"/sandboxes/{lab.sid}/files", params={"path": i["path"]})
                        lab._write(f"/workspace/lab/queue/{n}", json.dumps(job))
                        print(time.strftime("%H:%M:%S"), "requeued orphan", n, flush=True)
        except Exception as exc:
            print("orphan check error", exc, flush=True)
        try:
            res_names = {i["name"][:-5] for i in lab.ls("/workspace/lab/results")}
            in_running_dir = [i["name"][:-5] for i in lab.ls("/workspace/lab/running")
                              if i["name"].endswith(".json") and ".metrics" not in i["name"] and i["name"][:-5] not in res_names]
        except Exception:
            in_running_dir = []
        busy = sorted(set(running + [q[:-5] for q in queued] + in_running_dir))
        trains_busy = [j for j in busy if j.startswith("tr-")]
        if pend and len(trains_busy) < a.max_train:
            rec = json.loads(pend[0].read_text())
            job = rec["job"]
            ok = (len(trains_busy) < a.max_train and len(busy) - len(trains_busy) == 0) if job["kind"] == "train" else (len(busy) < a.fp_par)
            if ok:
                if rec.get("code") is not None:
                    lab._write(f"/workspace/lab/{job['program']}", rec["code"])
                lab._write(f"/workspace/lab/queue/{job['job_id']}.json", json.dumps(job))
                pend[0].rename(FEED / "pushed" / pend[0].name)
                print(time.strftime("%H:%M:%S"), "pushed", job["job_id"], "busy", busy, flush=True)
                time.sleep(40 if job["kind"] == "train" else 5)  # let the worker pick it up and refresh its heartbeat
                continue
        time.sleep(20)


if __name__ == "__main__":
    while True:  # 10-25: survive transient platform errors (503 / connection reset) instead of dying
        try:
            main()
        except Exception as exc:
            print(time.strftime("%H:%M:%S"), "feeder crashed, restarting in 60 s:", str(exc)[:200], flush=True)
            time.sleep(60)
