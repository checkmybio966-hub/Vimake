"""Vercel entrypoint (FastAPI).

Vercel ka FastAPI preset `api/index.py` ko entrypoint maanta hai: top-level me
`app` (ek FastAPI instance) milte hi SAARE requests isi app par aa jaate hain -
koi `builds`/`rewrites` ki jarurat nahi (aur `functions` ke saath `builds`
rakha to build fail ho jaata hai).

Serverless = light mode: ffmpeg nahi, /tmp writable, images yahin ban jaati
hain, videos `WM_WORKER_URL` wali service par jaate hain.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Vercel par hamesha light mode (ffmpeg/OpenCV-heavy video pipeline nahi chalani)
os.environ.setdefault("WM_LIGHT", "1")
os.environ.setdefault("WM_STORAGE", "local")

from server.app import app  # noqa: E402  <- FastAPI instance (Vercel isi ko dhoondhta hai)
