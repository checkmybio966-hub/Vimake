"""AUTO WATERMARK DETECTION FOR A SINGLE IMAGE (no temporal signal available).

A still image has no time axis, so we fuse several cheap cues - the same family
of cues a human uses when spotting a watermark. They are tried in order of how
reliable they are:

 1. REPEATED SHAPE  (strongest, ~90% precise)
    Shutterstock-style grids and repeated @handles print the same glyphs many
    times. We build a binary "ink" mask of the picture, pull out connected
    blobs, cluster them by SHAPE (same size + high mask-IoU after alignment),
    and if a cluster has 2+ members we template-match that shape over the whole
    ink mask. Comparing binary masks - not pixels - is what makes this work:
    the varying background underneath simply is not in the data.

 2. STROKE + PRIOR   (the general case: one logo, one timestamp)
    Score every stroke blob with five weak signals:
      corner/edge prior · size prior · stroke thinness/consistency (textness)
      translucency (low saturation, mid luminance) · fill ratio

 3. Nothing? -> report "not found" and the UI asks the user to brush it.

For production, plug a learned detector into `ExternalDetector` (bottom).
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .maskops import cleanup, dilate, ensure_mask
from .types import Detection

MAX_WORK_WIDTH = 1024
ELL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
ELL7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


# --------------------------------------------------------------------------- #
#  preprocessing
# --------------------------------------------------------------------------- #
def _prep(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    h, w = bgr.shape[:2]
    scale = 1.0
    if w > MAX_WORK_WIDTH:
        scale = MAX_WORK_WIDTH / float(w)
        bgr = cv2.resize(bgr, (MAX_WORK_WIDTH, int(round(h * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return bgr, gray, scale


def _stroke_mask(gray: np.ndarray) -> np.ndarray:
    """Thin edges of everything in the picture (used only to propose regions)."""
    med = float(np.median(gray))
    lo, hi = int(max(0, 0.66 * med)), int(min(255, 1.33 * med))
    e = cv2.Canny(gray, lo, hi)
    return cv2.morphologyEx(e, cv2.MORPH_CLOSE, ELL, 1)


def _ink_mask(gray: np.ndarray, box=None, k: float = 2.6) -> np.ndarray:
    """Filled "ink" of the watermark strokes.

    A watermark is a locally-bright (or locally-dark) layer on top of a smooth
    background, so comparing each pixel with a LARGE local median isolates it -
    this fills the glyphs instead of only tracing their outlines, which is what
    the inpainting model actually needs.
    """
    g = gray
    off = (0, 0)
    if box is not None:
        x, y, bw, bh = box
        pad = 6
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(g.shape[1], x + bw + pad), min(g.shape[0], y + bh + pad)
        g = g[y0:y1, x0:x1]
        off = (x0, y0)
    if min(g.shape) < 8:
        return np.zeros(gray.shape, np.uint8)
    ksz = 31 if min(g.shape) > 40 else (min(g.shape) // 2 * 2 + 1)
    loc = cv2.medianBlur(g, ksz)
    res = g.astype(np.float32) - loc.astype(np.float32)
    mad = float(np.median(np.abs(res - np.median(res)))) * 1.4826 + 1e-6
    thr = max(5.0, k * mad)
    ink = (np.abs(res) > thr).astype(np.uint8) * 255
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, ELL, 1)
    if box is None:
        return ink
    full = np.zeros(gray.shape, np.uint8)
    full[off[1]:off[1] + ink.shape[0], off[0]:off[0] + ink.shape[1]] = ink
    return full


# --------------------------------------------------------------------------- #
#  priors used by the stroke+prior path
# --------------------------------------------------------------------------- #
def _position_prior(box, shape) -> float:
    h, w = shape[:2]
    x, y, bw, bh = box
    cx, cy = x + bw / 2.0, y + bh / 2.0
    d_border = min(cx, cy, w - cx, h - cy) / float(max(w, h))
    pos = float(np.exp(-d_border / 0.10))
    dc = min(np.hypot(cx, cy), np.hypot(w - cx, cy),
             np.hypot(cx, h - cy), np.hypot(w - cx, h - cy)) / float(np.hypot(w, h))
    return float(np.clip(0.65 * pos + 0.35 * float(np.exp(-dc / 0.22)), 0, 1))


def _size_prior(box, shape, min_ratio=0.0005, max_ratio=0.08) -> float:
    h, w = shape[:2]
    r = (box[2] * box[3]) / float(w * h)
    if r < min_ratio * 0.2 or r > max_ratio * 3:
        return 0.0
    if r < min_ratio:
        return r / min_ratio
    if r > max_ratio:
        return max(0.0, 1.0 - (r - max_ratio) / (max_ratio * 2))
    return 1.0


def _textness(gray: np.ndarray, binmask: np.ndarray) -> float:
    sub = cv2.bitwise_and(gray, gray, mask=binmask)
    area = int((binmask > 0).sum())
    if area < 12:
        return 0.0
    e = cv2.Canny(sub, 40, 140)
    peri = float(cv2.countNonZero(e)) + 1e-6
    stroke = 2.0 * area / peri
    if stroke <= 0 or stroke > 25:
        return 0.0
    try:
        dist = cv2.distanceTransform(binmask, cv2.DIST_L2, 3)
        vals = dist[binmask > 0]
        cons = float(np.clip(1.0 - (vals.std() / (vals.mean() + 1e-6)), 0, 1))
    except Exception:
        cons = 0.5
    return float(np.clip(0.5 * float(np.clip(1.0 - stroke / 25.0, 0, 1)) + 0.5 * cons, 0, 1))


def _blend_prior(bgr: np.ndarray, binmask: np.ndarray) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1][binmask > 0], hsv[:, :, 2][binmask > 0]
    if sat.size == 0:
        return 0.0
    low_sat = float(np.clip(1.0 - sat.mean() / 140.0, 0, 1))
    mid = float(np.exp(-((val.mean() - 170.0) ** 2) / (2 * 70.0 ** 2)))
    return float(np.clip(0.55 * low_sat + 0.45 * mid, 0, 1))


# --------------------------------------------------------------------------- #
#  candidate blobs
# --------------------------------------------------------------------------- #
def _candidate_blobs(gray: np.ndarray) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    """Connected stroke blobs + MSER text groups -> [(box, tight_ink_mask)]."""
    strokes = _stroke_mask(gray)
    joined = cv2.dilate(cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, ELL7, 1), ELL7, 1)

    boxes: list[tuple[int, int, int, int]] = []
    n, _, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    for i in range(1, n):
        x, y, w, h = (int(stats[i, j]) for j in
                      (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        boxes.append((x, y, w, h))
    boxes += _group_boxes(_mser_candidates(gray))

    H, W = gray.shape[:2]
    blobs = []
    pad = 6
    for (x, y, bw, bh) in boxes:
        x, y = max(0, int(x)), max(0, int(y))
        bw, bh = min(int(bw), W - x), min(int(bh), H - y)
        if bw < 6 or bh < 4:
            continue
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + bw + pad), min(H, y + bh + pad)

        crop = _ink_mask(gray, (x, y, bw, bh))[y0:y1, x0:x1]     # ink, box-local
        edges = strokes[y0:y1, x0:x1]                            # edges, same window
        if crop.any():
            # keep only ink blobs that really contain strokes (kills flat areas)
            lab_n, lab, lstats, _ = cv2.connectedComponentsWithStats(crop, 8)
            keep = np.zeros_like(crop)
            for i in range(1, lab_n):
                yy = int(lstats[i, cv2.CC_STAT_TOP]); xx = int(lstats[i, cv2.CC_STAT_LEFT])
                hh = int(lstats[i, cv2.CC_STAT_HEIGHT]); ww = int(lstats[i, cv2.CC_STAT_WIDTH])
                if edges[yy:yy + hh, xx:xx + ww].any():
                    keep[lab == i] = 255
            crop = keep
        if not crop.any():
            continue
        full = np.zeros((H, W), np.uint8)
        full[y0:y1, x0:x1] = crop
        blobs.append(((x, y, bw, bh), full))
    return blobs


def _mser_candidates(gray: np.ndarray) -> list[np.ndarray]:
    try:
        mser = cv2.MSER_create(delta=4, min_area=24, max_area=int(0.02 * gray.size))
        regs, boxes = mser.detectRegions(gray.astype(np.uint8))
    except Exception:
        return []
    out, h, w = [], gray.shape[0], gray.shape[1]
    for (x, y, bw, bh) in ([] if boxes is None else list(boxes)):
        if bw < 6 or bh < 5 or bw > w * 0.6 or bh > h * 0.4:
            continue
        ar = bw / float(bh)
        if ar > 14 or ar < 0.07:
            continue
        out.append(np.array([x, y, bw, bh], int))
    return out


def _group_boxes(boxes: list[np.ndarray], gap: int = 18) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []
    canvas = np.zeros((int(max(b[1] + b[3] for b in boxes)) + gap * 2,
                       int(max(b[0] + b[2] for b in boxes)) + gap * 2), np.uint8)
    for (x, y, bw, bh) in boxes:
        cv2.rectangle(canvas, (int(x), int(y)), (int(x + bw), int(y + bh)), 255, -1)
    canvas = cv2.dilate(canvas, cv2.getStructuringElement(cv2.MORPH_RECT, (gap, max(3, gap // 3))), 1)
    n, _, stats, _ = cv2.connectedComponentsWithStats(canvas, 8)
    out = []
    for i in range(1, n):
        x, y, w, h = (int(stats[i, j]) for j in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                                 cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        if w >= 8 and h >= 6:
            out.append((max(0, x), max(0, y), w, h))
    return out


# --------------------------------------------------------------------------- #
#  1) repeated-shape detection
# --------------------------------------------------------------------------- #
def _shape_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two binary crops after resizing to a common canvas."""
    h = max(a.shape[0], b.shape[0]); w = max(a.shape[1], b.shape[1])
    if h == 0 or w == 0:
        return 0.0
    A = cv2.resize(a, (w, h), interpolation=cv2.INTER_NEAREST) > 127
    B = cv2.resize(b, (w, h), interpolation=cv2.INTER_NEAREST) > 127
    inter = float(np.logical_and(A, B).sum())
    uni = float(np.logical_or(A, B).sum())
    return inter / uni if uni else 0.0


def cluster_repeats(blobs, min_members: int = 2, iou_thr: float = 0.55,
                    size_tol: float = 0.35) -> list[list[int]]:
    """Group blobs that look like the same printed element."""
    n = len(blobs)
    used = [False] * n
    clusters: list[list[int]] = []
    for i in range(n):
        if used[i]:
            continue
        (xi, yi, wi, hi), mi = blobs[i]
        grp = [i]
        for j in range(i + 1, n):
            if used[j]:
                continue
            (xj, yj, wj, hj), mj = blobs[j]
            if abs(wi - wj) / float(max(wi, wj, 1)) > size_tol or \
               abs(hi - hj) / float(max(hi, hj, 1)) > size_tol:
                continue
            if _shape_iou(mi, mj) >= iou_thr:
                grp.append(j)
        if len(grp) >= min_members:
            used_grp = True
            for g in grp:
                used[g] = True
            clusters.append(grp)
    return clusters


def expand_repeat(strokes_or_ink: np.ndarray, template: np.ndarray,
                  threshold: float = 0.55) -> Optional[np.ndarray]:
    """Stamp `template` everywhere it matches in the binary ink mask."""
    tmpl = (template > 127).astype(np.uint8) * 255
    if tmpl.sum() < 20:
        return None
    th, tw = tmpl.shape[:2]
    H, W = strokes_or_ink.shape[:2]
    if th >= H or tw >= W:
        return None
    hay = (strokes_or_ink > 127).astype(np.float32)
    t = tmpl.astype(np.float32)
    res = cv2.matchTemplate(hay, t, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(res >= threshold)
    if len(xs) == 0:
        return None
    pts = sorted(zip(xs.tolist(), ys.tolist()), key=lambda p: -res[p[1], p[0]])
    keep: list[tuple[int, int, float]] = []
    for (x, y) in pts:
        if all(abs(x - kx) > tw * 0.5 or abs(y - ky) > th * 0.5 for (kx, ky, _) in keep):
            keep.append((int(x), int(y), float(res[y, x])))
        if len(keep) >= 40:
            break
    if len(keep) < 2:
        return None
    out = np.zeros((H, W), np.uint8)
    for (x, y, _) in keep:
        patch = out[y:y + th, x:x + tw]
        if patch.shape[:2] == tmpl.shape[:2]:
            out[y:y + th, x:x + tw] = cv2.bitwise_or(patch, tmpl)
    return out


# --------------------------------------------------------------------------- #
#  main entry point
# --------------------------------------------------------------------------- #
def detect_watermark_image(bgr: np.ndarray, min_ratio: float = 0.0004,
                           max_ratio: float = 0.12) -> Detection:
    bgr_s, gray, scale = _prep(bgr)
    blobs = _candidate_blobs(gray)
    if not blobs:
        return Detection(mask=np.zeros(bgr.shape[:2], np.uint8), score=0.0,
                         method="none", notes="no candidate region found")

    strokes = _stroke_mask(gray)
    ink_full = _ink_mask(gray)

    # ---- path 1: repeated shapes ------------------------------------------
    clusters = cluster_repeats(blobs)
    rep_mask: Optional[np.ndarray] = None
    hits = 0
    for grp in clusters:
        rep = max(grp, key=lambda i: int(blobs[i][1].sum()))     # most complete instance
        rm = blobs[rep][1]
        ys, xs = np.nonzero(rm)
        if xs.size == 0:
            continue
        crop = rm[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        exp = expand_repeat(ink_full, crop, 0.60)
        if exp is None:
            exp = expand_repeat(strokes, crop, 0.60)
        if exp is None or float((exp > 127).mean()) >= max_ratio * 2:
            continue                                             # silly coverage -> reject
        rep_mask = exp if rep_mask is None else cv2.bitwise_or(rep_mask, exp)
        hits += len(grp)
    if rep_mask is not None:
        score = 0.90 * float(np.clip(hits / 3.0, 0.72, 1.0))
        mask = _rescale_back(dilate(cleanup(rep_mask, min_area=16), 1), bgr.shape)
        return Detection(mask=mask, boxes=_boxes(mask), score=round(score, 3),
                         method="repeat-shape", kind="static",
                         notes=f"{len(clusters)} repeated shape cluster(s), {hits} instances")

    # ---- path 2: stroke + priors -------------------------------------------
    best: Optional[tuple[float, tuple, np.ndarray]] = None
    for box, tight in blobs:
        x, y, bw, bh = box
        sp = _size_prior(box, gray.shape, min_ratio, max_ratio)
        if sp <= 0:
            continue
        pp = _position_prior(box, gray.shape)
        tx = _textness(gray, tight)
        bl = _blend_prior(bgr_s, tight)
        fill = float((tight > 0).sum()) / float(max(1, bw * bh))
        fill_p = float(np.clip(1.0 - abs(fill - 0.30) / 0.45, 0, 1))
        score = 0.34 * pp + 0.18 * sp + 0.20 * tx + 0.16 * bl + 0.12 * fill_p
        if best is None or score > best[0]:
            best = (score, box, tight)

    if best is None:
        return Detection(mask=np.zeros(bgr.shape[:2], np.uint8), score=0.0,
                         method="none", notes="all candidates rejected by size prior")

    score, box, tight = best
    mask = _rescale_back(dilate(cleanup(tight, min_area=20), 2), bgr.shape)
    return Detection(mask=mask, boxes=_boxes(mask), score=round(float(np.clip(score, 0, 1)), 3),
                     method="stroke+prior", kind="static", notes=f"box={box}")


def _rescale_back(mask: np.ndarray, shape) -> np.ndarray:
    if mask.shape[:2] == shape[:2]:
        return ensure_mask(mask)
    return ensure_mask(cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST))


def _boxes(mask: np.ndarray) -> list[list[int]]:
    from .maskops import boxes_from_mask

    return boxes_from_mask(mask)


# --------------------------------------------------------------------------- #
#  optional: plug a real learned detector here
# --------------------------------------------------------------------------- #
class ExternalDetector:
    """Drop-in slot for a learned detector (recommended in production).

        detector = ExternalDetector(YoloWatermark("weights.pt"))   # -> list[[x,y,w,h]]
        det = detector(bgr)

    Good options:
      * YOLOv8/YOLO11 fine-tuned on watermark/logo boxes  (best accuracy)
      * DBNet / PaddleOCR det  (text overlays, timestamps, @handles)
      * Grounding-DINO prompted with "watermark, logo, text overlay" (zero-shot)
    """

    def __init__(self, impl=None):
        self.impl = impl

    def __call__(self, bgr: np.ndarray) -> Optional[Detection]:
        if self.impl is None:
            return None
        boxes = self.impl(bgr)
        if not boxes:
            return None
        mask = np.zeros(bgr.shape[:2], np.uint8)
        for (x, y, w, h) in boxes:
            cv2.rectangle(mask, (int(x), int(y)), (int(x + w), int(y + h)), 255, -1)
        return Detection(mask=mask, boxes=[list(map(int, b)) for b in boxes],
                         score=0.95, method="external-detector")
