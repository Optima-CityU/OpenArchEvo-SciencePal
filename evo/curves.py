#!/usr/bin/env python3
"""Read per-epoch validation curves of running/finished 30-epoch jobs from all lab sandboxes.

Writes runs/rank_validation/curves.json {job_id: {"cand": .., "lab": .., "epochs": [[ep, train_ppl, val_ppl], ...]}}
and prints the ranking at each epoch.
"""
import json
import re
import sys
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "lab"), str(ROOT / "harness")]
from lab_client import Lab  # noqa: E402

PAT = re.compile(r"epoch (\d+) \| train ppl ([\d.]+) \| val ppl ([\d.]+)")
OUT = ROOT / "runs" / "rank_validation"
OUT.mkdir(exist_ok=True)


def main(prefix="tr-"):
    labs = json.loads((Path(os.environ.get("OAE_LABS_JSON", ROOT / "config" / "labs.json"))).read_text())
    curves = json.loads((OUT / "curves.json").read_text()) if (OUT / "curves.json").exists() else {}
    for name, t in labs.items():
        try:
            lab = Lab(t)
            logs = [i for i in lab.ls("/workspace/lab/logs") if i["name"].endswith(".log")]
        except Exception as exc:
            print(name, "ERR", exc)
            continue
        for i in logs:
            jid = i["name"][:-4]
            if not jid.startswith(prefix):
                continue
            # only 30-epoch rank jobs (submitted tonight) — identified via the feed records
            try:
                txt = lab.read(i["path"]).decode(errors="replace")
            except Exception:
                continue
            ep = [[int(m.group(1)), float(m.group(2)), float(m.group(3))] for m in PAT.finditer(txt)]
            if not ep:
                continue
            cand = jid[3:-7]
            curves[jid] = {"cand": cand, "lab": name, "epochs": ep, "finished": "best val ppl" in txt}
    (OUT / "curves.json").write_text(json.dumps(curves, indent=1))
    rank_jobs = {j: c for j, c in curves.items() if len(c["epochs"]) >= 1 and j in RANK_JOBS()}
    if not rank_jobs:
        print("no rank jobs yet")
        return
    max_ep = max(len(c["epochs"]) for c in rank_jobs.values())
    for e in range(1, max_ep + 1):
        row = sorted(((c["epochs"][e - 1][2], c["cand"]) for c in rank_jobs.values() if len(c["epochs"]) >= e))
        print(f"ep{e:02d}", "  ".join(f"{n}:{v:.1f}" for v, n in row))


def RANK_JOBS():
    ids = set()
    for feed in ROOT.glob("runs/feed*/pushed/*.json"):
        r = json.loads(feed.read_text())
        if r["job"].get("protocol") == "sp-wt2-sltt-s128-e30-rank":
            ids.add(r["job"]["job_id"])
    for feed in ROOT.glob("runs/feed*/pending/*.json"):
        r = json.loads(feed.read_text())
        if r["job"].get("protocol") == "sp-wt2-sltt-s128-e30-rank":
            ids.add(r["job"]["job_id"])
    return ids


if __name__ == "__main__":
    main()
