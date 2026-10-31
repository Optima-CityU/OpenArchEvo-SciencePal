#!/usr/bin/env python3
"""SciencePal sandbox lab worker: a small file-queue runner for GPU jobs.

Runs inside the SciencePal sandbox (started once by the Lab agent). The external
controller drops job files into QUEUE via the sandbox file API and reads RESULTS
back the same way, so long GPU jobs do not need an agent turn each.

Job file  queue/<job_id>.json : {"job_id", "kind": "train"|"fingerprint", "program": "programs/<id>.py",
                                 "args": {...}, "protocol": "..."}
Result    results/<job_id>.json: {"job_id", "ok", "kind", "protocol", "seconds", ...metrics or error}
Heartbeat heartbeat.json updated every loop. Stop by creating a file named STOP.

Jobs never receive platform or API credentials: the child environment is filtered.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

LAB = Path(os.environ.get("LAB_DIR", "/workspace/lab"))
REPO = Path(os.environ.get("OAE_REPO", "/workspace/lab/OpenArchEvo"))
MAX_PAR = int(os.environ.get("LAB_MAX_PARALLEL", "2"))
for d in ("queue", "running", "results", "logs", "programs"):
    (LAB / d).mkdir(parents=True, exist_ok=True)

FP_SCRIPT = r'''
import sys, json, time
sys.path.insert(0, sys.argv[3])
from openarchevo.config import FingerprintConfig, ProbeConfig, WrapperConfig
from openarchevo.fingerprint.extract import extract_fingerprint
from openarchevo.fingerprint.schema import FEATURE_KEYS
a = json.loads(sys.argv[2])
cfg = FingerprintConfig(seed=a.get("seed", 42), neuron_type=a.get("neuron_type", "sltt"), devices=["cuda:0"],
    probe=ProbeConfig(batch_size=a.get("batch", 32), seq_len=a.get("seq_len", 2000)), wrapper=WrapperConfig(),
    structure_seq_len=a.get("structure_seq_len", 5767))
t = time.time()
fp = extract_fingerprint(open(sys.argv[1]).read(), cfg, device="cuda:0")
print("FP_JSON " + json.dumps({"seconds": round(time.time() - t, 1), "fingerprint": fp, "keys": FEATURE_KEYS}))
'''


def clean_env() -> dict:
    env = {k: v for k, v in os.environ.items()
           if not any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET", "PASSWORD", "SUPABASE", "AUTH"))}
    env["PYTHONPATH"] = str(REPO)
    # the sandbox sets HF_ENDPOINT to a mirror that was unreachable on 2026-10-21; huggingface.co is reachable
    env["HF_ENDPOINT"] = os.environ.get("LAB_HF_ENDPOINT", "https://huggingface.co")
    return env


def launch(job: dict) -> subprocess.Popen:
    jid, prog = job["job_id"], LAB / job["program"]
    log = open(LAB / "logs" / f"{jid}.log", "w")
    a = job.get("args", {})
    if job["kind"] == "train":
        cmd = [sys.executable, str(REPO / "training" / "train_wikitext2.py"), "--program", str(prog),
               "--neuron_type", a.get("neuron_type", "sltt"), "--epochs", str(a.get("epochs", 3)),
               "--seed", str(a.get("seed", 0)), "--output_dir", str(LAB / "out" / jid),
               "--result_json", str(LAB / "running" / f"{jid}.metrics.json")]
        if a.get("config"):
            cmd += ["--config", a["config"]]
        if a.get("data_dir"):
            cmd += ["--data_dir", a["data_dir"]]
    elif job["kind"] == "fingerprint":
        cmd = [sys.executable, "-c", FP_SCRIPT, str(prog), json.dumps(a), str(REPO)]
    else:
        raise ValueError(f"unknown kind {job['kind']}")
    env = clean_env()
    ngpu = int(os.environ.get("LAB_GPUS", "1"))
    gpu = None
    if ngpu > 1:  # several GPUs in this sandbox: give each job the least-used device
        used = [getattr(r, "_gpu", None) for r in RUNNING]
        gpu = min(range(ngpu), key=lambda g: used.count(g))
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if os.environ.get("LAB_GPU_AUTO") == "1":  # several sandboxes may share a multi-GPU machine: pick the least-used card
        try:
            q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
            rows = [tuple(int(x) for x in l.split(",")) for l in q if l.strip()]
            if rows:
                gpu = min(rows, key=lambda r: (r[1], r[2]))[0]
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        except Exception:
            pass
    p = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=log, stderr=subprocess.STDOUT)
    p._job, p._t0, p._log, p._gpu = job, time.time(), log, gpu  # type: ignore[attr-defined]
    return p


def finish(p: subprocess.Popen) -> None:
    job, jid = p._job, p._job["job_id"]  # type: ignore[attr-defined]
    p._log.close()  # type: ignore[attr-defined]
    if p.returncode == -9 and not job.get("oom_retry"):  # killed (cgroup OOM): requeue once
        job["oom_retry"] = True
        (LAB / "queue" / f"{jid}.json").write_text(json.dumps(job))
        (LAB / "running" / f"{jid}.json").unlink(missing_ok=True)
        return
    res = {"job_id": jid, "kind": job["kind"], "protocol": job.get("protocol"), "args": job.get("args", {}),
           "returncode": p.returncode, "seconds": round(time.time() - p._t0, 1), "ok": False}  # type: ignore[attr-defined]
    logtxt = (LAB / "logs" / f"{jid}.log").read_text(errors="replace")
    try:
        if job["kind"] == "train":
            m = json.loads((LAB / "running" / f"{jid}.metrics.json").read_text())
            res.update(metrics=m, ok=p.returncode == 0)
        else:
            line = [l for l in logtxt.splitlines() if l.startswith("FP_JSON ")][-1]
            res.update(json.loads(line[8:]), ok=True)
    except Exception as exc:
        res["error"] = f"{type(exc).__name__}: {exc}"
        res["log_tail"] = logtxt[-1500:]
    (LAB / "results" / f"{jid}.json").write_text(json.dumps(res, indent=1))
    (LAB / "running" / f"{jid}.json").unlink(missing_ok=True)


RUNNING: list = []


def main() -> None:
    running: list[subprocess.Popen] = RUNNING
    done = 0
    while not (LAB / "STOP").exists() or running:
        for p in list(running):
            if p.poll() is not None:
                finish(p)
                running.remove(p)
                done += 1
            elif time.time() - p._t0 > p._job.get("timeout_s", 4 * 3600):  # type: ignore[attr-defined]
                p.kill()
        try:  # parallelism can be changed at runtime by the controller
            max_par = int((LAB / "MAX_PARALLEL").read_text().strip())
        except Exception:
            max_par = MAX_PAR
        while len(running) < max_par and not (LAB / "STOP").exists():
            jobs = sorted((LAB / "queue").glob("*.json"), key=lambda f: f.stat().st_mtime)
            if not jobs:
                break
            jf = jobs[0]
            target = LAB / "running" / jf.name
            jf.replace(target)
            try:
                running.append(launch(json.loads(target.read_text())))
            except Exception as exc:
                jid = jf.stem
                (LAB / "results" / f"{jid}.json").write_text(json.dumps({"job_id": jid, "ok": False, "error": str(exc)}))
                target.unlink(missing_ok=True)
        hb = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "running": [p._job["job_id"] for p in running],
              "queued": len(list((LAB / "queue").glob("*.json"))), "done": done, "pid": os.getpid()}
        try:
            hb["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                                       capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            pass
        (LAB / "heartbeat.json").write_text(json.dumps(hb))
        time.sleep(15)


if __name__ == "__main__":
    main()
