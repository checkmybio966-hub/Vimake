"""Forward heavy jobs to a long-running worker.

Vercel (ya koi bhi serverless) par 10-60 s ki limit hoti hai, jabki ek 30 s ki
video ko inpainting me minute lag sakte hain. Isliye heavy jobs ek alag
long-running service (Docker / Railway / Render / Fly / apna GPU box) par
bhej diye jaate hain - **wo bhi yahi FastAPI app hai**, bas full mode me.

Config:
  WM_WORKER_URL=https://worker.up.railway.app
  WORKER_TOKEN=shared-secret        (optional but recommended)
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Optional

from . import config as cfgmod

cfg = cfgmod.cfg


def enabled() -> bool:
    return bool(cfg.worker_url)


def submit(source_url: str, kind: str, mode: str = "auto", remove_text: bool = True,
           backend: str = "auto", mask_key: Optional[str] = None) -> Optional[str]:
    """Ask the worker to process an asset; returns its job id."""
    if not enabled():
        return None
    payload = {"source_url": source_url, "kind": kind, "mode": mode,
               "remove_text": "1" if remove_text else "0", "backend": backend}
    if mask_key:
        payload["mask_key"] = mask_key
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if cfg.worker_token:
        headers["X-Worker-Token"] = cfg.worker_token
    req = urllib.request.Request(cfg.worker_url.rstrip("/") + "/api/remote/process",
                                 data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode()).get("job_id")


def poll(remote_id: str) -> Optional[dict]:
    """Current state of a remote job (same shape as our local JobState)."""
    if not enabled():
        return None
    headers = {"X-Worker-Token": cfg.worker_token} if cfg.worker_token else {}
    req = urllib.request.Request(
        cfg.worker_url.rstrip("/") + f"/api/remote/jobs/{remote_id}",
        headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def download_result(result_url: str, dest: str, timeout: int = 900) -> str:
    """Pull the finished file back (so results can be served from our storage)."""
    import urllib.request

    headers = {"X-Worker-Token": cfg.worker_token} if cfg.worker_token else {}
    if result_url.startswith("/"):
        result_url = cfg.worker_url.rstrip("/") + result_url
    req = urllib.request.Request(result_url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest


def check_token(headers) -> bool:
    """Worker-side auth check."""
    if not cfg.worker_token:
        return True
    return headers.get("x-worker-token") == cfg.worker_token


def submit_file(path: str, kind: str, mode: str = "auto", remove_text: bool = True,
                backend: str = "auto", mask_path: Optional[str] = None) -> Optional[str]:
    """Upload the asset to the worker (used when there is no shared storage)."""
    if not enabled():
        return None
    import mimetypes
    import uuid

    boundary = uuid.uuid4().hex
    fields = [("kind", kind), ("mode", mode),
              ("remove_text", "1" if remove_text else "0"), ("backend", backend)]
    body = b""
    for name, value in fields:
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                 f"{value}\r\n").encode()
    for field, file_path in (("file", path),
                             *([("mask", mask_path)] if mask_path else [])):
        ctype = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        with open(file_path, "rb") as f:
            data = f.read()
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; "
                 f"filename=\"{Path(file_path).name}\"\r\n"
                 f"Content-Type: {ctype}\r\n\r\n").encode() + data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if cfg.worker_token:
        headers["X-Worker-Token"] = cfg.worker_token
    import urllib.request

    req = urllib.request.Request(cfg.worker_url.rstrip("/") + "/api/remote/process",
                                 data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        import json
        return json.loads(r.read().decode()).get("job_id")


class _P:
    pass


try:
    from pathlib import Path  # noqa: F401
except Exception:  # pragma: no cover
    pass
