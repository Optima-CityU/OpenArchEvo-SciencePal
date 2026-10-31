#!/usr/bin/env python3
"""Collect all lab results (both sandboxes) into a local store: runs/lab_store.json {job_id: result+lab}."""
import json, sys
from pathlib import Path
import os
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab_client import Lab
LABS = json.loads((Path(os.environ.get("OAE_LABS_JSON", Path(__file__).resolve().parent.parent / "config" / "labs.json"))).read_text())
STORE = Path(__file__).resolve().parent.parent / "runs" / "lab_store.json"

def collect():
    store = json.loads(STORE.read_text()) if STORE.exists() else {}
    for name, t in LABS.items():
        try:
            for jid, r in Lab(t).results(skip=set(store)).items():
                if jid not in store:
                    r["lab"] = name
                    store[jid] = r
        except Exception as exc:
            print("collect error", name, exc)
        tmp = STORE.with_suffix(f".tmp{__import__('os').getpid()}")
        tmp.write_text(json.dumps(store))
        tmp.replace(STORE)
        print("collected", name, len(store), flush=True)
    tmp = STORE.with_suffix(f".tmp{__import__('os').getpid()}")
    tmp.write_text(json.dumps(store))
    tmp.replace(STORE)
    return store

def find(store, cand_id, kind, protocol, args):
    for jid, r in store.items():
        if r.get("ok") and r.get("kind") == kind and r.get("protocol") == protocol and r.get("args") == args \
                and jid.startswith(f"{kind[:2]}-{cand_id}-"):
            return {"job_id": jid, **r}
    return None

if __name__ == "__main__":
    s = collect()
    print(len(s), "results;", sum(1 for r in s.values() if r.get("ok")), "ok")
