"""Vercel serverless entry (ASGI).

Vercel's Python runtime natively serves ASGI apps, so `app` is all it needs.
`vercel.json` rewrites every path here ("/(.*)" -> "/api/index"), and Vercel
hands the ASGI app the ORIGINAL request path - isliye FastAPI ke apne routes
(/api/upload, /api/detect, ...) waise hi kaam karte hain.

Agar kabhi path "/api/index" ke roop me aa jaaye (kuch proxy setups), to neeche
ka wrapper use "/" se replace kar deta hai taaki kam se kam UI to load ho.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# serverless par ffmpeg/opencv heavy pipeline nahi chalani - video worker par
os.environ.setdefault("WM_LIGHT", "1")
os.environ.setdefault("WM_STORAGE", os.environ.get("WM_STORAGE", "local"))

from server.app import app as _fastapi_app  # noqa: E402


class _PathFix:
    """Defensive: /api/index -> / (UI) if a proxy rewrites the path literally."""

    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            path = scope.get("path", "")
            if path == "/api/index" or path.startswith("/api/index/"):
                scope = dict(scope)
                rest = path.split("/api/index", 1)[1]
                scope["path"] = rest or "/"
                scope["raw_path"] = scope["path"].encode()
        await self.inner(scope, receive, send)


app = _PathFix(_fastapi_app)
handler = app          # WSGI/ASGI fallback name some runtimes look for
