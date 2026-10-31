#!/usr/bin/env python3
"""Run one budget-gated SciencePal task and collect its artifacts.

  python run_task.py --name probe --prompt prompts/probe.md --attach build/a.zip build/b.zip \
      [--agent-id ID] [--expect probe.json] [--deadline 1800]

Writes runs/sp/<name>-<ts>/{meta.json,status.json,files/...}. Exactly one creation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sp_client import Client, NeedsAttention, record  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def find_files(c: Client, sandbox_id: str, names: set[str], roots=("/workspace", "/home", "/root", "/tmp", "/"),
               max_dirs: int = 400) -> dict[str, str]:
    found, seen, queue = {}, set(), list(roots)
    while queue and len(seen) < max_dirs and len(found) < len(names):
        d = queue.pop(0)
        if d in seen:
            continue
        seen.add(d)
        try:
            items = c.list_files(sandbox_id, d)
        except Exception:
            continue
        items = items.get("files", items) if isinstance(items, dict) else items
        for it in items or []:
            p, nm = it.get("path"), it.get("name")
            if it.get("is_dir"):
                if not any(x in (p or "") for x in ("/proc", "/sys", "/usr", "/lib", "node_modules", "/.cache",
                                                    "site-packages", "/dev", "/etc", "/var", "/bin", "/sbin", "/opt")):
                    queue.append(p)
            elif nm in names and nm not in found:
                found[nm] = p
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--attach", nargs="*", default=[])
    ap.add_argument("--agent-id")
    ap.add_argument("--expect", nargs="*", default=[])
    ap.add_argument("--fetch-dir", help="sandbox dir (relative names resolved by search) to list and download")
    ap.add_argument("--deadline", type=float, default=1800)
    ap.add_argument("--model")
    ap.add_argument("--meta", default="{}")
    ap.add_argument("--slot-wait", type=float, default=0, help="seconds to keep retrying on the concurrency 429")
    ap.add_argument("--web-search", action="store_true")
    a = ap.parse_args()

    out = ROOT / "runs" / "sp" / f"{a.name}-{time.strftime('%Y%m%dT%H%M%S')}"
    (out / "files").mkdir(parents=True, exist_ok=True)
    prompt = Path(a.prompt).read_text(encoding="utf-8") if Path(a.prompt).exists() else a.prompt
    files = [(Path(p).name, Path(p).read_bytes()) for p in a.attach]
    meta = {"name": a.name, "prompt_file": a.prompt, "attachments": [f[0] for f in files], **json.loads(a.meta)}
    (out / "prompt.md").write_text(prompt, encoding="utf-8")
    c = Client()
    t0 = time.time()
    while True:
        try:
            run = c.initiate(prompt, agent_id=a.agent_id, files=files, meta=meta, model_name=a.model,
                             web_search_on=a.web_search)
            break
        except NeedsAttention as exc:
            body = str(exc.record.get("body", ""))
            if "Concurrent task limit" in body and time.time() - t0 < a.slot_wait:
                print(time.strftime("%H:%M:%S"), "no free slot; retry in 60 s", flush=True)
                time.sleep(60)
                continue
            (out / "meta.json").write_text(json.dumps({"error": str(exc), **exc.record, **meta}, indent=1))
            print("NEEDS ATTENTION:", exc)
            return 3
    meta.update(thread_id=run.get("thread_id"), run_id=run.get("agent_run_id"), reservation=run["reservation"])
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("thread", meta["thread_id"], "run", meta["run_id"], flush=True)
    if not meta["run_id"]:
        print("no run id; inspect thread")
        return 3
    st = c.wait(meta["run_id"], deadline_s=a.deadline)
    st["wall_s"] = round(time.time() - t0, 1)
    (out / "status.json").write_text(json.dumps(st, indent=1))
    record("run_finished", run_id=meta["run_id"], status=st.get("status"), elapsed=st.get("wall_s"),
           deadline_reached=st.get("deadline_reached", False))
    print("status", st.get("status"), "elapsed", st["wall_s"], "error", st.get("error"), flush=True)
    try:
        sb = c.thread_sandbox(meta["thread_id"])
        (out / "sandbox.json").write_text(json.dumps(sb, indent=1))
        sid = sb.get("sandbox_id")
        if a.expect and sid:
            found = find_files(c, sid, set(a.expect))
            (out / "found.json").write_text(json.dumps(found, indent=1))
            for nm, p in found.items():
                (out / "files" / nm).write_bytes(c.read_file(sid, p))
            print("fetched", sorted(found))
        if a.fetch_dir and sid:
            listing = c.list_files(sid, a.fetch_dir)
            (out / "listing.json").write_text(json.dumps(listing, indent=1))
    except Exception as exc:
        print("artifact fetch failed:", type(exc).__name__, exc)
    print("OUT", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
