"""Backend registry. `get_backend("auto")` picks the best thing that actually works."""
from __future__ import annotations

from typing import Optional

from .. import config as cfgmod
from .base import InpaintBackend


def get_backend(name: Optional[str] = None, **kw) -> InpaintBackend:
    name = (name or cfgmod.cfg.backend or "auto").lower()

    if name == "opencv":
        from .opencv_backend import OpenCVBackend

        return OpenCVBackend(**kw)
    if name == "lama":
        from .lama_backend import LamaBackend

        return LamaBackend(**kw)
    if name == "iopaint":
        from .lama_backend import IOPaintBackend

        return IOPaintBackend(**kw)
    if name == "replicate":
        from .api_backend import ReplicateBackend

        return ReplicateBackend(**kw)
    if name == "fal":
        from .api_backend import FalBackend

        return FalBackend(**kw)
    if name == "custom":
        from .api_backend import CustomBackend

        return CustomBackend(**kw)

    # ---- auto -----------------------------------------------------------
    import os

    from .opencv_backend import OpenCVBackend
    fallback = OpenCVBackend()

    try:
        from .lama_backend import LamaBackend

        b = LamaBackend(**kw)
        if b.available:
            return b
    except Exception:
        pass

    if cfgmod.cfg.api_key or os.environ.get("REPLICATE_API_TOKEN") or os.environ.get("FAL_KEY"):
        if cfgmod.cfg.api_provider == "fal":
            from .api_backend import FalBackend

            return FalBackend(**kw)
        if cfgmod.cfg.api_provider == "custom":
            from .api_backend import CustomBackend

            return CustomBackend(**kw)
        from .api_backend import ReplicateBackend

        return ReplicateBackend(**kw)

    if os.environ.get("WM_IOPAINT_URL"):
        from .lama_backend import IOPaintBackend

        return IOPaintBackend(**kw)

    return fallback


def describe() -> dict:
    """What is configured / available - surfaced by GET /api/backends."""
    import os

    from .opencv_backend import OpenCVBackend
    info: dict = {"active": None, "available": [{"name": "opencv", "quality": 0.35, "local": True}]}
    try:
        from .lama_backend import LamaBackend

        b = LamaBackend()
        info["available"].append({"name": "lama", "quality": 0.9, "local": True,
                                  "ready": b.available, "model_path": b.model_path})
    except Exception:
        pass
    has_key = bool(cfgmod.cfg.api_key or os.environ.get("REPLICATE_API_TOKEN")
                   or os.environ.get("FAL_KEY"))
    info["available"].append({"name": "replicate", "quality": 0.88, "local": False,
                              "ready": has_key})
    info["available"].append({"name": "fal", "quality": 0.9, "local": False,
                              "ready": bool(os.environ.get("FAL_KEY"))})
    info["available"].append({"name": "iopaint", "quality": 0.92, "local": True,
                              "ready": bool(os.environ.get("WM_IOPAINT_URL"))})
    info["configured"] = cfgmod.cfg.backend
    info["active"] = get_backend().name
    return info
