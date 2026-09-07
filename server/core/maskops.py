"""Mask utilities: every mask in this project is a uint8 array, 255 = remove."""
from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


def ensure_mask(mask: np.ndarray, shape=None) -> np.ndarray:
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m.max(axis=2)
    m = m.astype(np.uint8)
    if shape is not None and m.shape[:2] != shape[:2]:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return m


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask, k, iterations=1)


def erode(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.erode(mask, k, iterations=1)


def feather(mask: np.ndarray, px: int) -> np.ndarray:
    """Soft edges so the fill blends instead of showing a hard seam."""
    if px <= 0:
        return mask
    return cv2.GaussianBlur(mask, (0, 0), px / 2.0)


def cleanup(mask: np.ndarray, min_area: int = 40, max_area: Optional[int] = None) -> np.ndarray:
    mask = ensure_mask(mask)
    if not mask.any():
        return mask
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < min_area:
            continue
        if max_area and a > max_area:
            continue
        out[lab == i] = 255
    return out


def keep_largest(mask: np.ndarray, k: int = 1) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ensure_mask(mask), 8)
    if n <= 1:
        return mask
    order = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1
    out = np.zeros_like(mask)
    for i in order[:k]:
        out[lab == i] = 255
    return out


def bbox(mask: np.ndarray):
    ys, xs = np.nonzero(ensure_mask(mask))
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def boxes_from_mask(mask: np.ndarray, min_area: int = 40) -> list[list[int]]:
    m = cleanup(mask, min_area=min_area)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    boxes = []
    for i in range(1, n):
        x, y, w, h = (int(stats[i, k]) for k in
                      (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        boxes.append([x, y, w, h])
    return boxes


def union(masks: Iterable[np.ndarray]) -> np.ndarray:
    out = None
    for m in masks:
        m = ensure_mask(m)
        out = m if out is None else cv2.bitwise_or(out, m)
    return out if out is not None else np.zeros((1, 1), np.uint8)


def mask_from_boxes(shape, boxes: Iterable[Iterable[int]], fill: int = 2) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    for (x, y, w, h) in boxes:
        cv2.rectangle(m, (int(x - fill), int(y - fill)), (int(x + w + fill), int(y + h + fill)), 255, -1)
    return m


def to_png_bytes(mask: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", ensure_mask(mask))
    if not ok:
        raise RuntimeError("png encode failed")
    return buf.tobytes()


def from_png_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    m = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise ValueError("bad mask png")
    return ensure_mask(m)


def coverage(mask: np.ndarray) -> float:
    m = ensure_mask(mask)
    return float((m > 127).mean())


def overlay_preview(frame: np.ndarray, mask: np.ndarray, alpha: float = 0.45,
                    color=(0, 0, 255)) -> np.ndarray:
    """Red translucent overlay, exactly what the web editor shows."""
    m = ensure_mask(mask, frame.shape) > 127
    out = frame.copy()
    out[m] = (np.array(color) * alpha + out[m] * (1 - alpha)).astype(np.uint8)
    return out
