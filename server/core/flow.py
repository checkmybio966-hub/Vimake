"""Optical-flow utilities + global motion estimation.

Two things live here:

1. `FlowField` - DIS/Farneback flow (contrib DIS if available, Farneback as
   fallback so plain opencv-python also works).

2. `MotionModel` - a global (camera) motion estimator. Needed because the
   "temporal median" trick only works if the frames are aligned; with a panning
   phone shot every pixel moves and the median would smear. We estimate an
   affine between consecutive frames with ECC and compose them.
"""
from __future__ import annotations

import cv2
import numpy as np


def _has_dis() -> bool:
    return hasattr(cv2, "DISOpticalFlow_create") or hasattr(cv2, "optflow")


def compute_flow(prev: np.ndarray, cur: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Forward flow prev -> cur returned at FULL resolution (float32, 2ch)."""
    pg = _gray(prev)
    cg = _gray(cur)
    h, w = pg.shape
    if scale != 1.0:
        sw, sh = max(16, int(w * scale)), max(16, int(h * scale))
        ps = cv2.resize(pg, (sw, sh), interpolation=cv2.INTER_AREA)
        cs = cv2.resize(cg, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        ps, cs, sw, sh = pg, cg, w, h

    try:
        if hasattr(cv2, "DISOpticalFlow_create"):
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            f = dis.calc(ps, cs, None)
        else:  # pragma: no cover
            raise AttributeError
    except Exception:
        f = cv2.calcOpticalFlowFarneback(ps, cs, None, 0.5, 3, 25, 3, 5, 1.2, 0)

    if (sw, sh) != (w, h):
        f = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)
        f = f.astype(np.float32)
        f[..., 0] *= w / float(sw)
        f[..., 1] *= h / float(sh)
    return f.astype(np.float32)


def _gray(f: np.ndarray) -> np.ndarray:
    if f.ndim == 3:
        return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    return f


def warp_image(img: np.ndarray, flow: np.ndarray, flags=cv2.INTER_LINEAR) -> np.ndarray:
    """Warp img by a forward displacement field (sample at x + flow)."""
    h, w = flow.shape[:2]
    if img.shape[:2] != (h, w):
        flow = cv2.resize(flow, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        fh, fw = img.shape[:2]
        sx, sy = fw / float(w), fh / float(h)
        flow = flow.copy()
        flow[..., 0] *= sx
        flow[..., 1] *= sy
    grid_x, grid_y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    return cv2.remap(img, map_x, map_y, flags, borderMode=cv2.BORDER_REPLICATE)


def compose_step(acc: np.ndarray, f: np.ndarray) -> np.ndarray:
    """acc_{k+1}(x) = acc_k(x) + f(x + acc_k(x))  (chain forward displacements)."""
    moved = warp_image(f, acc, flags=cv2.INTER_LINEAR)
    return acc + moved


class MotionModel:
    """Global (camera) motion between consecutive frames.

    Why not plain ECC?  ECC minimises a pixel-wise error over the WHOLE frame,
    so a locally moving subject (a person walking, a car) drags the estimate
    away from the true camera motion - and a wrong global transform smears the
    temporal median, which is fatal for the clean-plate stage.

    Instead we track sparse features with Lucas-Kanade and fit the affine with
    RANSAC: features on independently moving objects become outliers and are
    rejected, so a static camera really returns the identity transform.

    Two extra guards, both important in practice:
      * temporal smoothing - camera motion is smooth, estimates are jittery;
      * a dead zone - if the motion is sub-pixel we return the identity rather
        than resampling every frame (resampling = blur = worse median).
    """

    FEAT = dict(maxCorners=400, qualityLevel=0.01, minDistance=9, blockSize=7)
    LK = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 24, 0.03))

    def __init__(self, smooth: float = 0.55) -> None:
        self.smooth = smooth
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None
        self.ema = np.zeros(2, np.float32)          # smoothed translation

    @staticmethod
    def _as_3x3(m23: np.ndarray) -> np.ndarray:
        m = np.eye(3, dtype=np.float64)
        m[:2] = m23
        return m

    def push(self, frame_gray: np.ndarray) -> np.ndarray:
        """Feed the next frame; returns the affine mapping THIS frame -> previous."""
        ident = np.eye(2, 3, dtype=np.float32)
        # LK / goodFeatures need 8-bit images
        g = frame_gray if frame_gray.dtype == np.uint8 else np.clip(frame_gray, 0, 255).astype(np.uint8)
        g = np.ascontiguousarray(g)
        if self.prev_gray is None:
            self.prev_gray = g
            self.prev_pts = cv2.goodFeaturesToTrack(g, mask=None, **self.FEAT)
            return ident

        prev_pts = self.prev_pts
        if prev_pts is None or len(prev_pts) < 12:
            prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.FEAT)
        M = ident
        if prev_pts is not None and len(prev_pts) >= 12:
            cur, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, g, prev_pts, None, **self.LK)
            if cur is not None and status is not None:
                ok = status.ravel().astype(bool)
                if ok.sum() >= 12:
                    src = prev_pts[ok].reshape(-1, 2).astype(np.float32)
                    dst = cur[ok].reshape(-1, 2).astype(np.float32)
                    est, inliers = cv2.estimateAffinePartial2D(
                        dst, src, method=cv2.RANSAC, ransacReprojThreshold=2.0,
                        maxIters=400, confidence=0.995)
                    if est is not None and inliers is not None and float(inliers.mean()) > 0.35:
                        M = est.astype(np.float32)

        # temporal smoothing of the translation part, then the dead zone
        self.ema = self.smooth * self.ema + (1 - self.smooth) * M[:, 2]
        M = M.copy()
        M[:, 2] = self.ema
        scale = max(g.shape[:2]) / 720.0
        if np.linalg.norm(M[:, 2]) < 0.45 * max(1.0, scale):
            M = ident

        self.prev_gray = g
        self.prev_pts = cv2.goodFeaturesToTrack(g, mask=None, **self.FEAT)
        return M

    @staticmethod
    def compose(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
        """outer(inner(x)) for 2x3 affines -> returns 2x3."""
        return (MotionModel._as_3x3(outer) @ MotionModel._as_3x3(inner))[:2].astype(np.float32)

    @staticmethod
    def invert(m23: np.ndarray) -> np.ndarray:
        return cv2.invertAffineTransform(m23.astype(np.float32))
