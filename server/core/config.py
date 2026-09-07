"""Central configuration.

Everything the pipeline needs lives here so you can swap models/backends without
touching the algorithm code.

Environment overrides (all optional):
  WM_BACKEND          auto | opencv | lama | api          (default: auto)
  WM_API_PROVIDER     replicate | fal | dewatermark | unwatermark | vmodel | custom
  WM_API_KEY          API token for the remote provider
  WM_API_MODEL        provider specific model id / version
  LAMA_ONNX_PATH      path to big-lama ONNX model (used by the `lama` backend)
  WM_MAX_SIDE         processing resolution cap (default 960)
  WM_MAX_FRAMES       max frames processed per video (default 600)
  WM_WORKDIR          where uploads/results are stored (default ./runtime)
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT / "web"


def _is_serverless() -> bool:
    """Vercel / Lambda: repo read-only hota hai, sirf /tmp writable hai."""
    if os.environ.get("WM_LIGHT", "0") in ("1", "true", "yes"):
        return True
    return any(os.environ.get(k) for k in ("VERCEL", "AWS_LAMBDA_FUNCTION_NAME",
                                           "LAMBDA_TASK_ROOT"))


def _default_workdir() -> Path:
    env = os.environ.get("WM_WORKDIR")
    if env:
        return Path(env)
    if _is_serverless():
        return Path(tempfile.gettempdir()) / "wm-runtime"
    return ROOT / "runtime"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Config:
    # ---- storage -------------------------------------------------------
    workdir: Path = field(default_factory=_default_workdir)
    upload_dir: Path = field(init=False)
    result_dir: Path = field(init=False)

    # ---- quality / cost knobs ------------------------------------------
    max_side: int = field(default_factory=lambda: _env_int("WM_MAX_SIDE", 960))
    max_frames: int = field(default_factory=lambda: _env_int("WM_MAX_FRAMES", 600))
    max_upload_mb: int = field(default_factory=lambda: _env_int("WM_MAX_UPLOAD_MB", 500))
    crf: int = field(default_factory=lambda: _env_int("WM_CRQ", 18))

    # ---- detection ------------------------------------------------------
    detect_samples: int = field(default_factory=lambda: _env_int("WM_DETECT_SAMPLES", 48))
    detect_width: int = field(default_factory=lambda: _env_int("WM_DETECT_WIDTH", 640))
    mask_dilate_px: int = field(default_factory=lambda: _env_int("WM_MASK_DILATE", 6))
    mask_feather_px: int = field(default_factory=lambda: _env_int("WM_MASK_FEATHER", 3))
    min_area_ratio: float = field(default_factory=lambda: _env_float("WM_MIN_AREA", 0.0004))   # 0.04% of frame
    max_area_ratio: float = field(default_factory=lambda: _env_float("WM_MAX_AREA", 0.12))     # 12% of frame

    # ---- video removal ---------------------------------------------------
    # "clean-plate" = temporal median (best for MOVING watermarks: Sora/Veo/Kling)
    clean_plate_radius: int = field(default_factory=lambda: _env_int("WM_CP_RADIUS", 10))
    # robust z-score above which a pixel is declared "mark" (higher = safer)
    clean_plate_z: float = field(default_factory=lambda: _env_float("WM_CP_Z", 5.0))
    # "inpaint" = flow-guided propagation + spatial fill (for FIXED watermarks)
    inpaint_window: int = field(default_factory=lambda: _env_int("WM_INPAINT_WINDOW", 8))
    # moving marks: the window must outlast the mark's dwell time on a pixel
    moving_window: int = field(default_factory=lambda: _env_int("WM_MOVING_WINDOW", 24))
    moving_stride: int = field(default_factory=lambda: _env_int("WM_MOVING_STRIDE", 2))
    moving_plate: str = os.environ.get("WM_MOVING_PLATE", "median")   # median | percentile

    # ---- backends ---------------------------------------------------------
    backend: str = os.environ.get("WM_BACKEND", "auto")
    api_provider: str = os.environ.get("WM_API_PROVIDER", "replicate")
    api_key: str = os.environ.get("WM_API_KEY", "")
    api_model: str = os.environ.get("WM_API_MODEL", "")
    api_base: str = os.environ.get("WM_API_BASE", "")
    lama_onnx: str = os.environ.get("LAMA_ONNX_PATH", "")
    device: str = os.environ.get("WM_DEVICE", "auto")   # auto | cpu | cuda

    # ---- deployment ---------------------------------------------------
    # serverless (Vercel): bhaari jobs is worker ko bhej do
    worker_url: str = os.environ.get("WM_WORKER_URL", "")
    worker_token: str = os.environ.get("WORKER_TOKEN", "")
    # light mode = no ffmpeg / no local video pipeline (Vercel par yahi chalega)
    light: bool = os.environ.get("WM_LIGHT", "0") in ("1", "true", "yes")
    storage_mode: str = os.environ.get("WM_STORAGE", "local")
    redis_url: str = os.environ.get("REDIS_URL", "")

    def __post_init__(self) -> None:
        self.upload_dir = self.workdir / "uploads"
        self.result_dir = self.workdir / "results"
        for d in (self.upload_dir, self.result_dir):
            try:
                d.mkdir(parents=True, exist_ok=True)
            except OSError as exc:               # read-only FS (serverless)
                print(f"[config] cannot create {d} ({exc}) -> using temp dir")
                self.workdir = Path(tempfile.mkdtemp(prefix="wm-"))
                self.upload_dir = self.workdir / "uploads"
                self.result_dir = self.workdir / "results"
                self.upload_dir.mkdir(parents=True, exist_ok=True)
                self.result_dir.mkdir(parents=True, exist_ok=True)
                break
        if self.light:                 # serverless: sirf videos worker par
            self.max_frames = min(self.max_frames, 1)


cfg = Config()
