#!/usr/bin/env python3
"""Ingest training results for a protocol from the lab store into a controller state (by cand_id)."""
import json, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lab"))
from collect import collect
state, protocol = Path(sys.argv[1]), sys.argv[2]
store = collect()
pop = {json.loads(l)["id"] for l in (state / "population.jsonl").read_text().splitlines() if l.strip()}
done = set()
if (state / "ingested_jobs.txt").exists():
    done = set((state / "ingested_jobs.txt").read_text().split())
lines, new = [], []
for jid, r in store.items():
    if r.get("kind") != "train" or r.get("protocol") != protocol or jid in done:
        continue
    if (r.get("args") or {}).get("seed", 0) != 0:  # replicate seeds are analysed separately, never fed to search
        continue
    cid = jid[3:-7]
    if cid not in pop:
        continue
    m = (r.get("metrics") or {}).get("metrics", {}) if isinstance(r.get("metrics"), dict) else {}
    val = m.get("wikitext2_ppl")
    lines.append({"id": cid, "ok": bool(r.get("ok") and val is not None), "wikitext2_ppl": val, "protocol": protocol,
                  "seconds": r.get("seconds"), "seed": (r.get("args") or {}).get("seed"), "job": jid,
                  "test_ppl": ((r.get("metrics") or {}).get("details") or {}).get("test_ppl"),
                  "error": r.get("error")})
    new.append(jid)
if lines:
    p = state / f"results_store_{len(done)}.jsonl"
    p.write_text("".join(json.dumps(l) + "\n" for l in lines))
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "controller.py"), "--state", str(state), "ingest", "--results", str(p)], check=True)
    with open(state / "ingested_jobs.txt", "a") as f:
        f.write("\n".join(new) + "\n")
print(json.dumps([{k: l[k] for k in ("id", "ok", "wikitext2_ppl", "seconds")} for l in lines], indent=0))
