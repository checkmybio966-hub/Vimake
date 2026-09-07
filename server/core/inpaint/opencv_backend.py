"""Zero-dependency baseline: Navier-Stokes / Telea diffusion inpainting.

Runs anywhere, instantly, needs no model. Quality is "smudge the background in"
- good enough for small logos on simple backgrounds, weak on textured scenes.
It is the fallback the pipeline uses when no GPU/model/API is configured, and it
is also the final hole-closer after flow-guided propagation.
"""
from __future__ import annotations

import cv2
import numpy as np

from .base import InpaintBackend


class OpenCVBackend(InpaintBackend):
    name = "opencv"
    quality = 0.35
    cost = 0.0

    def __init__(self, method: str = "telea", radius: int = 5, pyramid: bool = True):
        self.method = method
        self.radius = radius
        self.pyramid = pyramid
        self.flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS

    def inpaint(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        hole = (mask > 127).astype(np.uint8) * 255
        if not hole.any():
            return frame
        # Sirf zaroori jagah chhuo: har mask blob ke around chhota ROI.
        # Poore frame par inpaint karna 10-50x slow hai aur result same dikhta hai.
        out = self._inpaint_rois(frame, hole)
        if out is not None:
            return out
        if not self.pyramid or max(frame.shape[:2]) <= 512:
            return cv2.inpaint(frame, hole, self.radius, self.flag)

        # Large holes diffuse badly at full res: solve coarse, then refine fine.
        h, w = frame.shape[:2]
        out = frame.copy()
        for scale in (0.25, 0.5, 1.0):
            sw, sh = max(16, int(w * scale)), max(16, int(h * scale))
            fi = cv2.resize(out, (sw, sh), interpolation=cv2.INTER_AREA)
            mi = cv2.resize(hole, (sw, sh), interpolation=cv2.INTER_NEAREST)
            r = max(2, int(self.radius * scale))
            fi = cv2.inpaint(fi, mi, r, self.flag)
            if scale < 1.0:
                fi = cv2.resize(fi, (w, h), interpolation=cv2.INTER_LINEAR)
                out = np.where(cv2.cvtColor(hole, cv2.COLOR_GRAY2BGR) > 0, fi, out).astype(np.uint8)
            else:
                out = fi
        return out

    @staticmethod
    def close_holes(frame: np.ndarray, hole: np.ndarray, radius: int = 3) -> np.ndarray:
        """Fill whatever the temporal propagation could not reach."""
        if not hole.any():
            return frame
        return cv2.inpaint(frame, hole, radius, cv2.INPAINT_TELEA)

    # ------------------------------------------------------------------ #
    def _inpaint_rois(self, frame: np.ndarray, hole: np.ndarray, pad: int = 28,
                      max_rois: int = 48) -> np.ndarray | None:
        """Inpaint each blob in a small padded crop, then paste it back."""
        n, _, stats, _ = cv2.connectedComponentsWithStats(hole, 8)
        if n <= 1 or n - 1 > max_rois:
            return None                       # too many blobs -> full-frame is faster
        H, W = frame.shape[:2]
        out = frame.copy()
        covered = 0
        for i in range(1, n):
            x = int(stats[i, cv2.CC_STAT_LEFT]); y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH]); h = int(stats[i, cv2.CC_STAT_HEIGHT])
            covered += w * h
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
            crop = out[y0:y1, x0:x1]
            cmask = hole[y0:y1, x0:x1]
            if crop.size == 0 or not cmask.any():
                continue
            r = max(2, int(self.radius * min(1.0, max(crop.shape[:2]) / 512.0) + 1))
            filled = cv2.inpaint(crop, cmask, r, self.flag)
            out[y0:y1, x0:x1] = np.where(cmask[..., None] > 0, filled, crop)
        # blobs poore frame me phail gaye to ROI ka fayda nahi
        if covered > 0.5 * H * W:
            return None
        return out
