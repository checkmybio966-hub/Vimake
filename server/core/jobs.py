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

from . import config as cfgmod
from .types import JobState

_LOCK = threading.Lock()
_JOBS: Dict[str, JobState] = {}

# ---------------- pluggable store (memory | redis) ----------------------
class _MemoryStore:
    name = "memory"

    def set(self, jid: str, st: JobState) -> None:
        _JOBS[jid] = st

    def get(self, jid: str):
        return _JOBS.get(jid)

    def update(self, jid: str, **kw) -> None:
        st = _JOBS.get(jid)
        if st:
            for k, v in kw.items():
                setattr(st, k, v)


class _RedisStore:
    """Serverless (Vercel) ke liye: state Redis/Upstash me, kyunki function
    stateless hota hai aur har request alag instance par ja sakti hai."""

    name = "redis"

    def __init__(self, url: str, ttl: int = 86400):
        import json

        import redis  # type: ignore

        self._json = json
        self.ttl = ttl
        self.r = redis.from_url(url)

    def _key(self, jid: str) -> str:
        return f"wm:job:{jid}"

    def set(self, jid: str, st: JobState) -> None:
        self.r.setex(self._key(jid), self.ttl, self._json.dumps(st.__dict__))

    def get(self, jid: str):
        raw = self.r.get(self._key(jid))
        if not raw:
            return None
        st = JobState(id=jid)
        st.__dict__.update(self._json.loads(raw))
        return st

    def update(self, jid: str, **kw) -> None:
        st = self.get(jid) or JobState(id=jid)
        for k, v in kw.items():
            setattr(st, k, v)
        self.set(jid, st)


_STORE = None


def _store():
    global _STORE
    if _STORE is None:
        url = cfgmod.cfg.redis_url if hasattr(cfgmod, "cfg") else ""
        if not url:
            import os
            url = os.environ.get("REDIS_URL", "")
        if url:
            try:
                _STORE = _RedisStore(url)
            except Exception as exc:                      # noqa: BLE001
                print(f"[jobs] redis unavailable ({exc}) -> memory")
                _STORE = _MemoryStore()
        else:
            _STORE = _MemoryStore()
    return _STORE


def create() -> JobState:
    jid = uuid.uuid4().hex[:12]
    st = JobState(id=jid)
    with _LOCK:
        _JOBS[jid] = st
    if _store().name != "memory":
        _store().set(jid, st)
    return st


def get(jid: str) -> Optional[JobState]:
    if _store().name != "memory":
        try:
            st = _store().get(jid)
            if st:
                return st
        except Exception:                                  # noqa: BLE001
            pass
    with _LOCK:
        return _JOBS.get(jid)


def update(jid: str, **kw) -> None:
    with _LOCK:
        st = _JOBS.get(jid)
        if st:
            for k, v in kw.items():
                setattr(st, k, v)
        else:
            st = JobState(id=jid)
            _JOBS[jid] = st
            for k, v in kw.items():
                setattr(st, k, v)
    if _store().name != "memory":
        try:
            _store().update(jid, **kw)
        except Exception:                                  # noqa: BLE001
            pass


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
