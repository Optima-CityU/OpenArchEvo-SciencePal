#!/usr/bin/env python3
"""If a lab's worker heartbeat is older than 40 min, move its feeder's pending jobs and its pushed-but-unfinished jobs
back to the shared pool so other labs pick them up (10-26: dead workers held jobs that runs were waiting on)."""
import datetime, json, shutil, sys
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "harness"), str(ROOT / "lab")]
from lab_client import Lab  # noqa: E402
R = ROOT / "runs"
labs = json.loads((Path(os.environ.get("OAE_LABS_JSON", ROOT / "config" / "labs.json"))).read_text())
store = json.loads((R / "lab_store.json").read_text())
now = datetime.datetime.now(datetime.timezone.utc)
for name, t in labs.items():
    feed = R / ("feed" if name == "lab1" else f"feed_{name}")
    if not feed.exists():
        continue
    try:
        hb = Lab(t).heartbeat()
        age = (now - datetime.datetime.fromisoformat(hb["ts"].replace("Z", "+00:00"))).total_seconds() / 60 if hb else 1e9
    except Exception:
        continue
    if age < 40:
        continue
    for sub in ("pending", "pushed"):
        for f in (feed / sub).glob("*.json"):
            try:
                jid = json.loads(f.read_text())["job"]["job_id"]
            except Exception:
                continue
            if jid in store:
                continue
            try:  # only jobs pushed in the last 24 h (old night-2 leftovers must not be revived)
                import re as _re; ts = float(_re.search(r"(\d{10}\.\d+)", f.name).group(1))
                if now.timestamp() - ts > 86400:
                    continue
            except Exception:
                continue
            shutil.move(str(f), str(R / "feed_pool/pending" / ("0" + f.name)))
            print(datetime.datetime.now().strftime("%H:%M:%S"), name, f"stale {age:.0f} min: requeued", jid, flush=True)
