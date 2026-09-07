"""Remote (GPU) backends.

Local CPU inpainting is fine for a demo, but for a real product you either run
your own GPU or you rent one per inference. This module talks to hosted
providers over plain HTTP (urllib only - no SDK dependency).

Providers
---------
replicate   image : `zylim0702/remove-object` (LaMa)  - set WM_API_MODEL to override
            video : `jd7h/xmem-propainter-inpainting`  (propagates a FIRST-FRAME mask
                    through the clip with XMem, then inpaints with ProPainter) - this
                    is the closest open equivalent of what Vmake does server-side.
fal         image : set WM_API_MODEL (e.g. an eraser/inpainting model on fal)
custom      any endpoint that accepts {"image": b64, "mask": b64} -> image bytes

Env vars: WM_API_PROVIDER, WM_API_KEY, WM_API_MODEL, WM_API_BASE
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from typing import Optional

import cv2
import numpy as np

from .. import config as cfgmod
from .base import InpaintBackend


def _b64_png(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("png encode failed")
    return base64.b64encode(buf.tobytes()).decode()


def _b64_jpg(img: np.ndarray, q: int = 95) -> str:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        raise RuntimeError("jpg encode failed")
    return base64.b64encode(buf.tobytes()).decode()


def _http(url: str, data: Optional[bytes] = None, headers: Optional[dict] = None,
          timeout: int = 120, method: Optional[str] = None):
    req = urllib.request.Request(url, data=data, headers=headers or {},
                                 method=method or ("POST" if data is not None else "GET"))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "ignore"))


class ReplicateBackend(InpaintBackend):
    name = "replicate"
    supports_video = True
    quality = 0.88
    cost = 1.0

    API = "https://api.replicate.com/v1"

    def __init__(self, token: str = "", image_model: str = "", video_model: str = ""):
        self.token = token or cfgmod.cfg.api_key or os.environ.get("REPLICATE_API_TOKEN", "")
        self.image_model = image_model or cfgmod.cfg.api_model or "zylim0702/remove-object"
        self.video_model = video_model or "jd7h/xmem-propainter-inpainting"

    # ------------------------------------------------------------------ #
    def _create(self, model: str, payload: dict) -> dict:
        if not self.token:
            raise RuntimeError("REPLICATE_API_TOKEN / WM_API_KEY is not set")
        url = f"{self.API}/models/{model}/predictions"
        body = json.dumps({"input": payload}).encode()
        return _http(url, body, {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Prefer": "wait",
        }, timeout=180)

    def _poll(self, result: dict, timeout: float = 900.0) -> dict:
        url = result.get("urls", {}).get("get") or f"{self.API}/predictions/{result['id']}"
        t0 = time.time()
        while result.get("status") in ("starting", "processing", "queued"):
            if time.time() - t0 > timeout:
                raise RuntimeError("remote prediction timed out")
            time.sleep(2.0)
            result = _http(url, None, {"Authorization": f"Bearer {self.token}"}, timeout=60)
        if result.get("status") != "succeeded":
            raise RuntimeError(f"remote prediction failed: {result.get('error')}")
        return result

    @staticmethod
    def _first_url(out) -> str:
        if isinstance(out, str):
            return out
        if isinstance(out, list) and out:
            return out[0] if isinstance(out[0], str) else str(out[0])
        return str(out)

    # ------------------------------------------------------------------ #
    def inpaint(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        payload = {"image": f"data:image/png;base64,{_b64_png(frame)}",
                   "mask": f"data:image/png;base64,{_b64_png(mask)}"}
        res = self._poll(self._create(self.image_model, payload))
        url = self._first_url(res.get("output"))
        with urllib.request.urlopen(url, timeout=300) as r:
            data = r.read()
        out = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if out is None:
            raise RuntimeError("remote returned undecodable image")
        return cv2.resize(out, (frame.shape[1], frame.shape[0]))

    def inpaint_video(self, video_path: str, mask_path: str, out_path: str) -> Optional[str]:
        with open(video_path, "rb") as f:
            vurl = "data:video/mp4;base64," + base64.b64encode(f.read()).decode()
        with open(mask_path, "rb") as f:
            murl = "data:image/png;base64," + base64.b64encode(f.read()).decode()
        res = self._poll(self._create(self.video_model, {"video": vurl, "mask": murl}),
                         timeout=1800)
        url = self._first_url(res.get("output"))
        with urllib.request.urlopen(url, timeout=900) as r, open(out_path, "wb") as o:
            o.write(r.read())
        return out_path


class FalBackend(InpaintBackend):
    name = "fal"
    supports_video = True
    quality = 0.9
    cost = 1.0

    def __init__(self, key: str = "", model: str = ""):
        self.key = key or cfgmod.cfg.api_key or os.environ.get("FAL_KEY", "")
        self.model = model or cfgmod.cfg.api_model
        self.base = cfgmod.cfg.api_base or "https://queue.fal.run"

    def _run(self, payload: dict, timeout: float = 900.0):
        if not self.key:
            raise RuntimeError("FAL_KEY / WM_API_KEY is not set")
        if not self.model:
            raise RuntimeError("set WM_API_MODEL to a fal model id")
        res = _http(f"{self.base.rstrip('/')}/{self.model.strip('/')}",
                    json.dumps(payload).encode(),
                    {"Authorization": f"Key {self.key}", "Content-Type": "application/json"},
                    timeout=180)
        status_url = res.get("status_url")
        if not status_url:
            return res
        t0 = time.time()
        while True:
            st = _http(status_url, None, {"Authorization": f"Key {self.key}"}, timeout=60)
            s = st.get("status")
            if s in ("COMPLETED", "OK", "SUCCESS"):
                return _http(st["response_url"], None,
                             {"Authorization": f"Key {self.key}"}, timeout=120)
            if s in ("FAILED", "ERROR"):
                raise RuntimeError(f"fal job failed: {st}")
            if time.time() - t0 > timeout:
                raise RuntimeError("fal job timed out")
            time.sleep(2.0)

    def inpaint(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        res = self._run({"image_url": f"data:image/png;base64,{_b64_png(frame)}",
                         "mask_url": f"data:image/png;base64,{_b64_png(mask)}"})
        url = res.get("image", {}).get("url") or res.get("image_url")
        if not url:
            raise RuntimeError(f"unexpected fal response: {list(res)[:5]}")
        with urllib.request.urlopen(url, timeout=300) as r:
            data = r.read()
        out = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        return cv2.resize(out, (frame.shape[1], frame.shape[0]))


class CustomBackend(InpaintBackend):
    """POST {"image": b64, "mask": b64} -> raw image bytes (or {"url": ...}).

    Point WM_API_BASE at anything you self-host (a RunPod/A100 box running
    IOPaint, a Triton deployment, your own FastAPI microservice...).
    """

    name = "custom"
    quality = 0.85
    cost = 1.0

    def __init__(self, url: str = "", key: str = ""):
        self.url = url or cfgmod.cfg.api_base
        self.key = key or cfgmod.cfg.api_key

    def inpaint(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not self.url:
            raise RuntimeError("WM_API_BASE not set")
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        res = _http(self.url, json.dumps({"image": _b64_png(frame),
                                          "mask": _b64_png(mask)}).encode(),
                    headers, timeout=600)
        if isinstance(res, dict) and ("url" in res or "image" in res):
            url = res.get("url") or res.get("image")
            with urllib.request.urlopen(url, timeout=300) as r:
                data = r.read()
        elif isinstance(res, dict) and "image_base64" in res:
            data = base64.b64decode(res["image_base64"])
        else:
            raise RuntimeError("custom backend: unsupported response shape")
        out = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        return cv2.resize(out, (frame.shape[1], frame.shape[0]))
