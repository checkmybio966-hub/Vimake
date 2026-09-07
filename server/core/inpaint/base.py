from __future__ import annotations

import abc

import numpy as np


class InpaintBackend(abc.ABC):
    """Contract used by the pipeline.

    Implementations receive a BGR uint8 frame and a uint8 mask (255 = hole) and
    must return a BGR uint8 frame where the hole is filled.
    """

    name = "base"
    supports_video = False
    quality = 0.5          # 0..1, used by the "auto" backend picker
    cost = 0.0             # 0 = free/local, >0 = paid per call (arbitrary units)

    @abc.abstractmethod
    def inpaint(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        ...

    # ---- optional: whole-video removal (used when backend is a remote API) --
    def inpaint_video(self, video_path: str, mask_path: str, out_path: str) -> str | None:
        raise NotImplementedError(f"{self.name} backend has no native video mode")
