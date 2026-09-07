"""Tiny in-process job registry.

Good enough for a single box / a demo. In production swap this for Celery +
Redis + S3 (see docs/05-production-roadmap.md) - the rest of the code does not
change because every worker only needs `pipeline.process_*`.
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Callable, Dict, Optional

from .types import JobState

_LOCK = threading.Lock()
_JOBS: Dict[str, JobState] = {}


def create() -> JobState:
    jid = uuid.uuid4().hex[:12]
    st = JobState(id=jid)
    with _LOCK:
        _JOBS[jid] = st
    return st


def get(jid: str) -> Optional[JobState]:
    with _LOCK:
        return _JOBS.get(jid)


def update(jid: str, **kw) -> None:
    with _LOCK:
        st = _JOBS.get(jid)
        if st:
            for k, v in kw.items():
                setattr(st, k, v)


def run(jid: str, fn: Callable[[Callable[[float, str], None]], None]) -> None:
    """Execute `fn` in a worker thread; exceptions are recorded on the job."""

    def wrapper() -> None:
        update(jid, status="running", progress=0.0, stage="starting")
        t0 = time.time()

        def progress(v: float, stage: str) -> None:
            update(jid, progress=round(float(v), 3), stage=stage)

        try:
            fn(progress)
            update(jid, status="done", progress=1.0, stage="done",
                   message=f"finished in {time.time() - t0:.1f}s")
        except Exception as exc:                       # noqa: BLE001
            update(jid, status="error", message=f"{type(exc).__name__}: {exc}")

    threading.Thread(target=wrapper, daemon=True).start()


def gc(max_age_s: float = 3600.0) -> int:
    """Convenience for a cron/background task - not strictly needed."""
    now = time.time()
    drop = []
    with _LOCK:
        for k, v in _JOBS.items():
            if getattr(v, "_ts", now) < now - max_age_s:
                drop.append(k)
        for k in drop:
            _JOBS.pop(k, None)
    return len(drop)
