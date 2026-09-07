"""AUTO WATERMARK DETECTION FOR VIDEO - the part that makes it feel "AI magic".

A video gives us something a photo cannot: TIME. Two completely different
signals come out of the time axis, and between them they cover both kinds of
watermark that exist in the wild.

--------------------------------------------------------------------------
CASE A - FIXED watermark (corner logo, TV bug, timestamp, TikTok export)
--------------------------------------------------------------------------
The mark never moves, so it does NOT behave like the scene around it:

  * variance attenuation A: a constant layer laid over moving content FLATTENS
    the motion it covers, so the temporal variance inside the mark is far lower
    than the variance of its surroundings;
  * edge persistence P:     the logo outline sits on the exact same pixels in
    ~100% of frames, while scene edges come and go.

      static_score = 0.55 * A_norm + 0.45 * P_norm

  These two are then verified with a CONTRAST test (how much stronger the
  signal is inside the candidate than outside) - that is what separates a real
  overlay from an ordinary flat region such as the inside of a moving object.

--------------------------------------------------------------------------
CASE B - MOVING / FLOATING mark (Sora, Veo, Kling, animated CapCut badge)
--------------------------------------------------------------------------
The mark is somewhere else in every frame, so for any pixel it is present only
in a few frames and ABSENT in most:

      clean plate  = median over time of that pixel   (== background, no mark)
      residual r   = |frame - clean plate|

Where the mark passes, r is an OUTLIER relative to that pixel's own history. So
instead of a fixed threshold (which a busy scene destroys) we use an adaptive
per-pixel robust z-score:

      z = (r - median_t r) / (1.4826 * MAD_t r)

A moving mark produces big sporadic z; a fixed mark produces z ≈ 0 because it
is constant. That single statistic both finds the mark and tells the two cases
apart - and the very same z-map is reused by the removal stage.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .maskops import cleanup, dilate, ensure_mask
from .types import Detection

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def _gray(f: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f


def stack_gray(frames: Sequence[np.ndarray], width: int = 640) -> np.ndarray:
    """Bring a sampled frame list to a common (small) size -> (T, H, W) float32."""
    g0 = _gray(frames[0])
    h, w = g0.shape[:2]
    if w > width:
        s = width / float(w)
        h, w = int(round(h * s)), width
    out = np.empty((len(frames), h, w), np.float32)
    for i, f in enumerate(frames):
        g = _gray(f)
        if g.shape[:2] != (h, w):
            g = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
        out[i] = g.astype(np.float32)
    return out


def _norm(x: np.ndarray, p: float = 99.0) -> np.ndarray:
    return np.clip(x / (np.percentile(x, p) + 1e-6), 0, 1)


def temporal_signals(stack: np.ndarray) -> dict:
    """Every map the detector needs. stack: (T,H,W) float32 gray."""
    med = np.median(stack, axis=0)                     # clean plate (moving marks vanish)
    dev = np.abs(stack - med[None])
    D = np.median(dev, axis=0)                         # robust "always different" map
    S = stack.std(axis=0)                              # temporal variance
    Sblur = cv2.blur(S, (25, 25))
    A = np.clip((Sblur - S) / (Sblur + 1e-6), 0, 1)    # variance attenuation

    P = np.zeros(stack.shape[1:], np.float32)          # edge persistence
    for i in range(stack.shape[0]):
        e = cv2.Laplacian(stack[i], cv2.CV_32F, ksize=3)
        P += (np.abs(e) > 12).astype(np.float32)
    P /= float(stack.shape[0])

    # adaptive outlier score of the residual (the MOVING-mark signal)
    med_r = np.median(dev, axis=0)
    mad_r = np.median(np.abs(dev - med_r[None]), axis=0) * 1.4826 + 2.0
    z = (dev - med_r[None]) / mad_r[None]

    return {"median": med, "D": D, "S": S, "A": A, "P": P,
            "z": z, "zp90": np.percentile(z, 90, axis=0)}


def _threshold_mask(score: np.ndarray, q: float, shape,
                    min_ratio: float, max_ratio: float) -> np.ndarray:
    """Top-q% of the score map -> cleaned binary mask."""
    thr = np.percentile(score, q)
    m = ((score >= thr) * 255).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, KERNEL, 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), 2)
    h, w = shape
    return cleanup(m, min_area=int(min_ratio * h * w), max_area=int(max_ratio * h * w))


def grow_from_seeds(score: np.ndarray, q_seed: float, q_grow: float, shape,
                    min_ratio: float, max_ratio: float) -> np.ndarray:
    """Seed at a strict threshold, then keep whole blobs down to a looser one.

    Detection almost always finds the *core* of a watermark first; its fainter
    parts (antialiased edges, the low-contrast half of a logo) sit a little
    below the threshold. Growing from the core recovers them.
    """
    core = _threshold_mask(score, q_seed, shape, min_ratio, max_ratio)
    if not core.any():
        return core
    cand = ((score >= np.percentile(score, q_grow)) * 255).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, KERNEL, 1)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    out = np.zeros(shape, np.uint8)
    for i in range(1, n):
        comp = lab == i
        if (comp & (core > 127)).sum() >= max(3, 0.02 * int(comp.sum())):
            out[comp] = 255
    h, w = shape
    return cleanup(out, min_area=int(min_ratio * h * w), max_area=int(max_ratio * h * w))


def detect_watermark_video(frames: Sequence[np.ndarray], target_shape=None,
                           min_ratio: float = 0.0004, max_ratio: float = 0.12) -> Detection:
    if len(frames) < 4:
        return Detection(mask=np.zeros(_gray(frames[0]).shape, np.uint8), score=0.0,
                         method="not-enough-frames", notes="need >= 4 sampled frames")

    stack = stack_gray(frames, width=640)
    sig = temporal_signals(stack)
    An = _norm(sig["A"], 99)
    Pn = _norm(sig["P"], 99)
    zp = sig["zp90"]
    Z = _norm(cv2.blur(np.clip(zp, 0, None), (5, 5)), 99.5)

    static_score = 0.55 * An + 0.45 * Pn
    moving_score = Z * (0.5 + 0.5 * (1.0 - Pn))

    s_mask = _threshold_mask(static_score, 96.0, stack.shape[1:], min_ratio, max_ratio)
    m_mask = grow_from_seeds(moving_score, 98.0, 94.0, stack.shape[1:], min_ratio, max_ratio)

    # ---- evaluate the "fixed overlay" hypothesis -------------------------
    static_ok, s_conf, s_notes = False, 0.0, ""
    if s_mask.any():
        inside = s_mask > 127
        p_in, p_out = float(Pn[inside].mean()), float(Pn[~inside].mean())
        a_in, a_out = float(An[inside].mean()), float(An[~inside].mean())
        c_p = p_in / (p_out + 0.05)
        c_a = a_in / (a_out + 0.05)
        # a FIXED overlay is constant -> it is never an outlier (z ~ 0)
        static_ok = p_in > 0.40 and c_p > 2.0 and a_in > 0.30 and float(zp[inside].mean()) < 2.0
        s_conf = float(np.clip(0.45 * min(c_p / 4.0, 1.0) + 0.30 * min(c_a / 8.0, 1.0)
                               + 0.25 * p_in, 0, 1))
        s_notes = (f"edge-persistence inside={p_in:.2f} vs outside={p_out:.2f} "
                   f"(x{c_p:.1f}), variance-attenuation x{c_a:.1f}")

    # ---- evaluate the "moving mark" hypothesis ---------------------------
    moving_ok, m_conf, m_notes = False, 0.0, ""
    if m_mask.any():
        inside = m_mask > 127
        z_in = float(zp[inside].mean())
        z_out = float(zp[~inside].mean())
        # a MOVING mark is a real shape seen again and again as it drifts, so its
        # edges persist (p_in > 0.25) even though its pixels are outliers.
        moving_ok = (z_in > 3.0 and z_in > z_out * 2.0
                     and float(Pn[inside].mean()) > 0.25)
        m_conf = float(np.clip(0.6 * min(z_in / 8.0, 1.0) +
                               0.4 * min(z_in / (z_out * 4.0 + 1e-6), 1.0), 0, 1))
        m_notes = f"outlier z inside={z_in:.2f} vs outside={z_out:.2f}"

    # ---- decide -----------------------------------------------------------
    if static_ok and (not moving_ok or s_conf >= m_conf):
        mask, kind, conf, notes = s_mask, "static", s_conf, s_notes
        method = "variance-attenuation+edge-persistence"
    elif moving_ok:
        mask, kind, conf, notes = m_mask, "moving", m_conf, m_notes
        method = "temporal-outlier (clean plate)"
    elif s_mask.any():
        mask, kind, conf, notes = s_mask, "static", s_conf * 0.6, s_notes + " (weak)"
        method = "variance-attenuation+edge-persistence"
    else:
        return Detection(mask=np.zeros(stack.shape[1:], np.uint8), score=0.0,
                         method="temporal", notes="no consistent overlay found")

    if target_shape is not None and mask.shape[:2] != target_shape[:2]:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

    return Detection(mask=ensure_mask(mask), boxes=_boxes(ensure_mask(mask)),
                     score=round(float(conf), 3), method=method, kind=kind, notes=notes)


def _boxes(mask: np.ndarray) -> list[list[int]]:
    from .maskops import boxes_from_mask

    return boxes_from_mask(mask)


def per_frame_moving_mask(frame_gray: np.ndarray, clean_plate: np.ndarray,
                          thresh: float = 12.0) -> np.ndarray:
    """Where is the (moving) mark in THIS frame? -> |frame - clean plate|."""
    d = cv2.absdiff(frame_gray.astype(np.float32), clean_plate.astype(np.float32))
    d = cv2.blur(d, (3, 3))
    m = (d > thresh).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, KERNEL, 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, KERNEL, 1)
    return dilate(m, 1)


def adaptive_alpha(diff_map: np.ndarray, med_d: np.ndarray, mad_d: np.ndarray,
                   k: float = 4.0) -> np.ndarray:
    """Robust per-pixel z-score -> 0..1 'this pixel is the mark' alpha.

    This is what makes the clean-plate stage safe on busy footage: a pixel is
    only replaced when its residual is an OUTLIER for that pixel's own history,
    not merely because the scene moved.
    """
    z = (diff_map - med_d) / (1.4826 * mad_d + 2.0)
    return np.clip((z - k) / (k * 0.8), 0, 1).astype(np.float32)
