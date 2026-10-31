"""Controller side of the SciencePal sandbox lab queue (file API only; never creates agent runs)."""
from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))
from sp_client import Client, record  # noqa: E402

LAB = "/workspace/lab"


class Lab:
    def __init__(self, thread_id: str, client: Client | None = None):
        self.c = client or Client()
        self.thread_id = thread_id
        self.sid = self.c.thread_sandbox(thread_id)["sandbox_id"]

    def _write(self, path: str, content: bytes | str) -> None:
        if isinstance(content, str):
            content = content.encode()
        files = {"file": (path.split("/")[-1], content, "application/octet-stream")}
        r = self.c._c.post(f"/sandboxes/{self.sid}/files", data={"path": path}, files=files)
        if r.status_code == 409:  # exists: the API does not overwrite, so delete and write again
            self.c._c.delete(f"/sandboxes/{self.sid}/files", params={"path": path}).raise_for_status()
            r = self.c._c.post(f"/sandboxes/{self.sid}/files", data={"path": path}, files=files)
        r.raise_for_status()

    def read(self, path: str) -> bytes:
        return self.c.read_file(self.sid, path)

    def ls(self, d: str) -> list[dict]:
        x = self.c.list_files(self.sid, d)
        return x.get("files", x) if isinstance(x, dict) else x

    def heartbeat(self) -> dict | None:
        try:
            return json.loads(self.read(f"{LAB}/heartbeat.json"))
        except Exception:
            return None

    def submit(self, kind: str, cand_id: str, code: str, args: dict, protocol: str, timeout_s: int = 4 * 3600) -> str:
        job_id = f"{kind[:2]}-{cand_id}-{uuid.uuid4().hex[:6]}"
        self._write(f"{LAB}/programs/{cand_id}.py", code)
        job = {"job_id": job_id, "kind": kind, "program": f"programs/{cand_id}.py", "args": args,
               "protocol": protocol, "cand_id": cand_id, "timeout_s": timeout_s}
        self._write(f"{LAB}/queue/{job_id}.json", json.dumps(job))
        record("lab_job_submitted", job_id=job_id, kind=kind, cand_id=cand_id, protocol=protocol, args=args)
        return job_id

    def results(self, skip=()) -> dict[str, dict]:
        out = {}
        for it in self.ls(f"{LAB}/results"):
            if it["name"].endswith(".json") and it["name"][:-5] not in skip:
                try:
                    out[it["name"][:-5]] = json.loads(self.read(it["path"]))
                except Exception:
                    pass
        return out

    def find_result(self, cand_id: str, kind: str, protocol: str, args: dict) -> dict | None:
        """Reuse an existing successful result for the same candidate, kind, protocol and args."""
        for jid, r in self.results().items():
            if r.get("ok") and r.get("kind") == kind and r.get("protocol") == protocol and r.get("args") == args \
                    and jid.startswith(f"{kind[:2]}-{cand_id}-"):
                return {"job_id": jid, **r}
        return None

    def wait(self, job_ids: list[str], timeout_s: float = 6 * 3600, poll_s: float = 60) -> dict[str, dict]:
        t0, got = time.time(), {}
        while time.time() - t0 < timeout_s:
            res = self.results()
            got = {j: res[j] for j in job_ids if j in res}
            if len(got) == len(job_ids):
                break
            time.sleep(poll_s)
        return got
