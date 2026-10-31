"""Minimal SciencePal HTTP client with a persistent, atomic call budget.

Endpoints mirror the official SDK 0.2.1 (statically audited, sha256
9b732dbd...559e). We use httpx directly so that NO request is ever retried
automatically: a POST that creates a run is attempted at most once, and an
unknown outcome is recorded and stops the caller.

Budget ledger: one JSON-lines file. Every billable creation (initiate/start)
first appends a `reserve` entry under an exclusive file lock; the reservation
counts against the limit whether or not the request succeeds. The limits file
is written by the human operator only.
"""
from __future__ import annotations

import fcntl
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import httpx

BASE_URL = os.environ.get("SCIENCEPAL_BASE_URL", "https://sciencepal.ai/api")
STATE = Path(os.environ.get("SP_STATE_DIR", Path(__file__).resolve().parent.parent / "ledger"))
LEDGER = STATE / "ledger.jsonl"
LIMITS = STATE / "limits.json"  # {"max_creations": N, "max_concurrent": 1, "note": "..."}


class BudgetExceeded(RuntimeError):
    pass


class NeedsAttention(RuntimeError):
    def __init__(self, reason: str, record: dict):
        super().__init__(reason)
        self.record = record


@contextmanager
def _locked():
    STATE.mkdir(parents=True, exist_ok=True)
    with open(STATE / ".lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _append(entry: dict) -> None:
    entry = {"ts": time.time(), **entry}
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def ledger_entries() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(l) for l in LEDGER.read_text(encoding="utf-8").splitlines() if l.strip()]


def limits() -> dict:
    if not LIMITS.exists():
        return {"max_creations": 0, "max_concurrent": 1}
    return json.loads(LIMITS.read_text())


def reserve(kind: str, meta: dict) -> str:
    """Atomically reserve one billable creation; raises BudgetExceeded."""
    with _locked():
        L = ledger_entries()
        # a 429 means the run was never created (concurrency limit), so that reservation is released
        released = {e.get("reservation") for e in L
                    if e.get("event") in ("initiate_http_error", "start_http_error") and e.get("status") in (400, 404, 409, 422, 429, 503)}
        used = sum(1 for e in L if e.get("event") == "reserve" and e.get("reservation") not in released)
        lim = limits()
        if used >= int(lim.get("max_creations", 0)):
            raise BudgetExceeded(f"creation budget exhausted: used {used} / {lim.get('max_creations', 0)}")
        day = time.strftime("%Y-%m-%d", time.localtime())
        used_day = sum(1 for e in L if e.get("event") == "reserve" and e.get("reservation") not in released
                       and time.strftime("%Y-%m-%d", time.localtime(e.get("ts", 0))) == day)
        if used_day >= int(lim.get("max_per_day", 10**9)):
            raise BudgetExceeded(f"daily budget exhausted: {used_day} / {lim.get('max_per_day')} on {day}")
        rid = uuid.uuid4().hex
        _append({"event": "reserve", "reservation": rid, "kind": kind, **meta})
        return rid


def record(event: str, **kw: Any) -> None:
    with _locked():
        _append({"event": event, **kw})


class Client:
    def __init__(self, api_key: Optional[str] = None, timeout: float = 60.0):
        key = api_key or os.environ.get("SCIENCEPAL_API_KEY")
        if not key:
            raise RuntimeError("SCIENCEPAL_API_KEY not set")
        self._c = httpx.Client(base_url=BASE_URL, timeout=httpx.Timeout(timeout, connect=15.0),
                               headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})

    def close(self):
        self._c.close()

    # ---- read-only (safe to repeat) --------------------------------------------
    def _get(self, endpoint: str, params: Optional[dict] = None, **kw) -> httpx.Response:
        params = {**(params or {}), **kw}
        r = self._c.get(endpoint, params=params or None)
        r.raise_for_status()
        return r

    def list_agents(self, search: Optional[str] = None, limit: int = 100) -> Any:
        return self._get("/agents", page=1, limit=limit, **({"search": search} if search else {})).json()

    def get_agent(self, agent_id: str) -> dict:
        return self._get(f"/agents/{agent_id}").json()

    def run_status(self, run_id: str) -> dict:
        return self._get(f"/agent-run/{run_id}").json()

    def thread_runs(self, thread_id: str) -> Any:
        return self._get(f"/thread/{thread_id}/agent-runs").json()

    def thread_sandbox(self, thread_id: str) -> dict:
        return self._get(f"/thread/{thread_id}/sandbox").json()

    def list_files(self, sandbox_id: str, directory: str = "/") -> Any:
        return self._get(f"/sandboxes/{sandbox_id}/files", {"path": directory}).json()

    def read_file(self, sandbox_id: str, path: str) -> bytes:
        return self._get(f"/sandboxes/{sandbox_id}/files/content", {"path": path}).content

    # ---- state-changing, NOT billable runs -------------------------------------
    def create_agent(self, name: str, system_prompt: str, description: str = "", **extra) -> dict:
        body = {"name": name, "system_prompt": system_prompt, "description": description, **extra}
        r = self._c.post("/agents", json=body)
        r.raise_for_status()
        return r.json()

    def update_agent(self, agent_id: str, **fields) -> dict:
        r = self._c.put(f"/agents/{agent_id}", json=fields)
        r.raise_for_status()
        return r.json()

    def stop_run(self, run_id: str) -> None:
        self._c.post(f"/agent-run/{run_id}/stop", json={}).raise_for_status()

    # ---- billable: creates an agent run (exactly one attempt, budget-gated) -----
    def initiate(self, prompt: str, *, agent_id: Optional[str], files: list[tuple[str, bytes]] = (),
                 meta: Optional[dict] = None, model_name: Optional[str] = None,
                 web_search_on: bool = False, enable_thinking: bool = False,
                 reasoning_effort: str = "low") -> dict:
        meta = dict(meta or {})
        rid = reserve("initiate", {"agent_id": agent_id, **meta})
        data = {"prompt": prompt, "stream": "true", "agent_select_type": "manual",
                "web_search_on": "true" if web_search_on else "false",
                "enable_thinking": "true" if enable_thinking else "false",
                "reasoning_effort": reasoning_effort, "enable_context_manager": "false",
                "is_agent_builder": "false"}
        if agent_id:
            data["agent_id"] = agent_id
        if model_name:
            data["model_name"] = model_name
        up = [("files", (n, b, "application/octet-stream")) for n, b in files] or None
        try:
            r = self._c.post("/agent/initiate", data=data, files=up, timeout=httpx.Timeout(900.0, connect=15.0))
        except Exception as exc:  # outcome unknown: do NOT retry
            record("initiate_unknown", reservation=rid, error=f"{type(exc).__name__}: {exc}")
            raise NeedsAttention("initiate outcome unknown; inspect account before any new run",
                                 {"reservation": rid}) from exc
        if r.status_code >= 400:
            record("initiate_http_error", reservation=rid, status=r.status_code, body=r.text[:500])
            raise NeedsAttention(f"initiate HTTP {r.status_code}", {"reservation": rid, "body": r.text[:500]})
        out = r.json()
        record("initiated", reservation=rid, thread_id=out.get("thread_id"), run_id=out.get("agent_run_id"))
        return {"reservation": rid, **out}

    def continue_thread(self, thread_id: str, message: str, *, meta: Optional[dict] = None,
                        model_name: Optional[str] = None, agent_id: Optional[str] = None) -> dict:
        """Billable: add a user message to an existing thread and start one new run (one attempt each)."""
        meta = dict(meta or {})
        rid = reserve("start", {"thread_id": thread_id, **meta})
        mid = str(uuid.uuid4())
        try:
            r = self._c.post(f"/thread/{thread_id}/messages", json={"message_id": mid, "content": message},
                             timeout=httpx.Timeout(120.0, connect=15.0))
            r.raise_for_status()
            body = {"stream": True}
            if model_name:
                body["model_name"] = model_name
            if agent_id:
                body["agent_id"] = agent_id
            r = self._c.post(f"/thread/{thread_id}/agent/start", json=body, timeout=httpx.Timeout(300.0, connect=15.0))
        except Exception as exc:
            record("start_unknown", reservation=rid, thread_id=thread_id, message_id=mid, error=f"{type(exc).__name__}: {exc}")
            raise NeedsAttention("start outcome unknown; check list_thread_runs before retrying", {"reservation": rid}) from exc
        if r.status_code >= 400:
            record("start_http_error", reservation=rid, status=r.status_code, body=r.text[:500])
            raise NeedsAttention(f"start HTTP {r.status_code}", {"reservation": rid, "body": r.text[:500]})
        out = r.json()
        record("started", reservation=rid, thread_id=thread_id, run_id=out.get("agent_run_id"), message_id=mid)
        return {"reservation": rid, "thread_id": thread_id, **out}

    def wait(self, run_id: str, *, deadline_s: float = 1800, first_poll_s: float = 10,
             max_interval_s: float = 60) -> dict:
        """Poll until a non-running status or the deadline. Never stops the run itself."""
        t0, interval, last = time.time(), first_poll_s, None
        while time.time() - t0 < deadline_s:
            try:
                last = self.run_status(run_id)
            except httpx.HTTPError as exc:
                last = {"status": "poll_error", "error": str(exc)}
            st = (last or {}).get("status")
            if st not in (None, "running", "queued", "pending", "poll_error", "in_progress", "starting"):
                return {"elapsed_s": round(time.time() - t0, 1), **last}
            time.sleep(interval)
            interval = min(max_interval_s, interval * 1.5)
        return {"elapsed_s": round(time.time() - t0, 1), "deadline_reached": True, **(last or {})}
