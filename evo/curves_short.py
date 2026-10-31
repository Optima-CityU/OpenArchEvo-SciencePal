#!/usr/bin/env python3
"""Fetch per-epoch curves only for 30-epoch jobs (job id contains 'e30'), in parallel over labs.
Merges into runs/rank_validation/curves.json."""
import json, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "lab"), str(ROOT / "harness")]
from lab_client import Lab
PAT = re.compile(r"epoch (\d+) \| train ppl ([\d.]+) \| val ppl ([\d.]+)")
OUT = ROOT / "runs/rank_validation/curves_short.json"
FILT = sys.argv[1] if len(sys.argv) > 1 else r"e6|vale6|e26|e29"

def one(item):
    name, t = item
    res = {}
    try:
        lab = Lab(t)
        logs = [i for i in lab.ls("/workspace/lab/logs") if i["name"].endswith(".log") and re.search(FILT, i["name"])]
    except Exception as exc:
        print(name, "ERR", str(exc)[:80], flush=True); return res
    for i in logs:
        try:
            txt = lab.read(i["path"]).decode(errors="replace")
        except Exception:
            continue
        ep = [[int(m.group(1)), float(m.group(2)), float(m.group(3))] for m in PAT.finditer(txt)]
        if ep:
            res[i["name"][:-4]] = {"lab": name, "epochs": ep, "finished": "best val ppl" in txt}
    print(name, len(logs), len(res), flush=True)
    return res

labs = json.loads((Path(os.environ.get("OAE_LABS_JSON", ROOT / "config" / "labs.json"))).read_text())
cur = json.loads(OUT.read_text()) if OUT.exists() else {}
with ThreadPoolExecutor(6) as ex:
    for r in ex.map(one, labs.items()):
        for j, v in r.items():
            v["cand"] = j[3:]
            cur[j] = v
OUT.write_text(json.dumps(cur, indent=1))
print("total", len(cur))
