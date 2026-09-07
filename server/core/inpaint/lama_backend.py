"""LaMa (Large Mask Inpainting, WACV 2022) - the open model most "AI eraser"
tools use under the hood. Loads a big-lama ONNX export through onnxruntime.

Setup
-----
    pip install onnxruntime-gpu          # or onnxruntime for CPU
    # get an ONNX export, e.g. from the IOPaint / lama-cleaner release, or:
    #   https://huggingface.co/Carve/LaMa-ONNX  (big-lama.pt -> onnx)
    export LAMA_ONNX_PATH=/models/big-lama.onnx
    export WM_BACKEND=lama

If the model or onnxruntime is missing the backend simply reports itself
unavailable and the pipeline falls back to OpenCV - the app never crashes.
"""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

from .. import config as cfgmod
from .base import InpaintBackend


class LamaBackend(InpaintBackend):
    name = "lama"
    supports_video = False
    quality = 0.9
    cost = 0.0

    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        self.model_path = model_path or cfgmod.cfg.lama_onnx or os.environ.get("LAMA_ONNX_PATH", "")
        self.device = device
        self._sess = None
        self._input_name: Optional[str] = None
        self.available = False
        self._load()

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        if not self.model_path or not os.path.exists(self.model_path):
            return
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError:
            return
        providers = ["CPUExecutionProvider"]
        if self.device in ("auto", "cuda"):
            if "CUDAExecutionProvider" in ort.get_available_providers() and self.device != "cpu":
                providers.insert(0, "CUDAExecutionProvider")
        try:
            self._sess = ort.InferenceSession(self.model_path, providers=providers)
            self._input_name = self._sess.get_inputs()[0].name
            self.available = True
        except Exception:
            self._sess = None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _prep(frame: np.ndarray, mask: np.ndarray):
        """Pad to a multiple of 8, normalise to [-1, 1] - the big-lama contract."""
        h, w = frame.shape[:2]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        img = cv2.copyMakeBorder(frame, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        msk = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

        img = img.astype(np.float32) / 255.0
        msk = (msk > 127).astype(np.float32)
        img = img[:, :, ::-1]                       # BGR -> RGB
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        img = (img - mean) / std
        x = np.concatenate([img, msk[:, :, None]], axis=2)     # H W 4
        x = np.transpose(x, (2, 0, 1))[None].astype(np.float32)
        return x, (h, w)

    def inpaint(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not self.available:
            raise RuntimeError("LaMa backend not available (missing model or onnxruntime)")
        x, (h, w) = self._prep(frame, mask)
        out = self._sess.run(None, {self._input_name: x})[0]
        out = np.squeeze(out)
        if out.ndim == 3 and out.shape[0] in (1, 3, 4):
            out = np.transpose(out, (1, 2, 0))
        out = out[:h, :w]
        out = np.clip(out, -1, 1)
        out = (out * 0.5 + 0.5) * 255.0 if out.min() < 0 else out * 255.0
        out = out[:, :, ::-1].astype(np.uint8)                # RGB -> BGR
        hole = (mask > 127)
        res = frame.copy()
        res[hole] = out[hole]
        return res


class IOPaintBackend(InpaintBackend):
    """Alternative local path: run IOPaint (lama_cleaner) as a service.

        pip install iopaint
        iopaint start --model=lama --device=cuda --port=8080
        export WM_IOPAINT_URL=http://127.0.0.1:8080

    IOPaint exposes LaMa / MAT / ZITS / SD / Anything and even a browser UI, so
    it is the fastest way to get production-grade removal without writing
    inference code. Kept here as a thin HTTP client.
    """

    name = "iopaint"
    quality = 0.92
    cost = 0.0

    def __init__(self, url: str = ""):
        import urllib.request

        self.url = url or os.environ.get("WM_IOPAINT_URL", "http://127.0.0.1:8080")
        self._urllib = urllib.request

    def inpaint(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        import base64
        import json

        ok, ib = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        ok2, mb = cv2.imencode(".png", mask)
        if not (ok and ok2):
            raise RuntimeError("encode failed")
        payload = json.dumps({
            "image": base64.b64encode(ib.tobytes()).decode(),
            "mask": base64.b64encode(mb.tobytes()).decode(),
            "ldm_steps": 25,
        }).encode()
        req = self._urllib.Request(
            self.url.rstrip("/") + "/api/v1/inpaint",
            data=payload, headers={"Content-Type": "application/json"},
        )
        with self._urllib.urlopen(req, timeout=300) as r:
            data = r.read()
        arr = np.frombuffer(data, np.uint8)
        out = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if out is None:
            raise RuntimeError("iopaint returned undecodable image")
        return cv2.resize(out, (frame.shape[1], frame.shape[0]))
