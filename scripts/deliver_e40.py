#!/usr/bin/env python3
"""E40 delivery: when a run has finished generation 4, queue its top-2 children (own fitness) for 30 epochs x seeds 0,1."""
import argparse, glob, json, os, subprocess, time, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="runs/e40")
ap.add_argument("--pool", default="runs/feed_pool/pending", help="feeder pool directory (lab/feeder.py)")
a = ap.parse_args()
os.chdir(ROOT)
out, pool = Path(a.out), Path(a.pool)
pool.mkdir(parents=True, exist_ok=True)
dfile = out / "delivered.json"
done = set(json.loads(dfile.read_text())) if dfile.exists() else set()
runs = sorted(glob.glob(str(out / "*_s?")))
log = lambda *x: print(time.strftime("%H:%M"), *x, flush=True)
while len(done) < len(runs):
    for d in runs:
        name = os.path.basename(d)
        if name in done:
            continue
        if subprocess.run(["pgrep", "-f", f"run_arm2.py --state {d}"], capture_output=True).stdout.strip():
            continue
        pop = [json.loads(l) for l in open(f"{d}/population.jsonl") if l.strip()]
        ch = [c for c in pop if not c["meta"].get("seed") and (c.get("measured") or {}).get("wikitext2_ppl") is not None]
        top = sorted(ch, key=lambda c: c["measured"]["wikitext2_ppl"])[:2]
        log(name, "children trained", len(ch), "deliver", [(c["id"], round(c["measured"]["wikitext2_ppl"], 2)) for c in top])
        for c in top:
            for sd in (0, 1):
                jid = f"tr-{c['id']}-e40e30s{sd}-{uuid.uuid4().hex[:4]}"
                job = {"job_id": jid, "kind": "train", "program": f"programs/{c['id']}.py",
                       "args": {"neuron_type": "sltt", "epochs": 30, "seed": sd, "config": "/workspace/lab/wikitext2_lab.yaml"},
                       "protocol": "sp-wt2-sltt-s128-e30-rank", "cand_id": c["id"], "timeout_s": 36000}
                (pool / f"1_40_{name}_{jid}.json").write_text(json.dumps({"job": job, "code": c["code"], "t": time.time()}))
        done.add(name)
        dfile.write_text(json.dumps(sorted(done)))
        (out / f"deliver_{name}.json").write_text(json.dumps({name: [c["id"] for c in top]}))
    time.sleep(300)
log("all delivered")
