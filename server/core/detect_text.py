"""TEXT / CAPTION DETECTION FOR IMAGES (zero extra dependency).

"Image me koi bhi text nahi chahiye" - isliye watermark ke saath-saath
captions, timestamps, subtitles, @handles, sticker-text sab pakadne padte hain.

Yeh classic (non-deep-learning) text detector hai, jo bina koi model download
kiye kaam karta hai:

  1. MSER (Maximally Stable Extremal Regions) dono polarity par - chhota text
     bright-on-dark aur dark-on-bright dono hota hai.
  2. Har candidate region ko geometry se filter karo:
        - size          : 4px <= height <= 15% of image
        - aspect ratio  : 0.1 .. 12
        - fill ratio    : glyph andar kitna bhara hai (0.15 .. 0.9)
        - stroke width  : distance transform se - text ka stroke patla aur
                          CONSISTENT hota hai (std/mean chhota)
  3. Bachhe hue components ko line/word me group karo (horizontal dilate) -
     text akele glyph nahi, line hota hai.
  4. Har group ke andar "ink mask" le lo (filled glyphs, sirf outline nahi).

Agar aapke paas GPU/accuracy ki requirement ho to neeche `OcrTextDetector` me
EasyOCR / PaddleOCR / doctr plug kar dijiye - interface wahi hai (boxes out).
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
def _component_stats(binmask: np.ndarray, gray: np.ndarray):
    """Geometry + stroke-width consistency of one connected component."""
    area = int(np.count_nonzero(binmask))
    if area < 8:
        return None
    ys, xs = np.nonzero(binmask)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    w, h = x1 - x0, y1 - y0
    if w < 2 or h < 3:
        return None

    fill = area / float(w * h)
    aspect = w / float(h)

    dist = cv2.distanceTransform(binmask, cv2.DIST_L2, 3)
    vals = dist[binmask > 0]
    if vals.size < 5:
        return None
    sw = float(vals.mean())                      # mean stroke half-width
    cons = float(np.clip(1.0 - vals.std() / (sw + 1e-6), 0, 1))

    contrast = float(gray[binmask > 0].mean()) - float(gray[~binmask].mean())
    return {"box": (x0, y0, w, h), "area": area, "fill": fill, "aspect": aspect,
            "stroke": 2.0 * sw, "consistency": cons, "contrast": contrast}


def _keep(s: dict, h_img: int, w_img: int) -> bool:
    if not (4 <= s["box"][3] <= 0.15 * h_img):            # height
        return False
    if not (3 <= s["box"][2] <= 0.65 * w_img):            # width
        return False
    if not (0.12 <= s["fill"] <= 0.92):                   # glyph bhara hua
        return False
    if not (0.10 <= s["aspect"] <= 14.0):                 # na line, na square block
        return False
    if not (0.8 <= s["stroke"] <= 0.35 * s["box"][3]):    # patla stroke
        return False
    if s["consistency"] < 0.30:                           # stroke width consistent
        return False
    if abs(s["contrast"]) < 6:                            # background se alag dikhe
        return False
    return True


def detect_text_regions(gray: np.ndarray, min_chars: int = 3,
                        aggressive: bool = False) -> list[tuple[int, int, int, int]]:
    """Return list of (x, y, w, h) boxes that look like text lines / words."""
    H, W = gray.shape[:2]
    if min(H, W) < 24:
        return []
    g = gray if gray.dtype == np.uint8 else gray.astype(np.uint8)
    g = cv2.GaussianBlur(g, (3, 3), 0)

    accepted: list[tuple[int, int, int, int]] = []
    for polarity in (g, 255 - g):
        try:
            mser = cv2.MSER_create(delta=3, min_area=18,
                                   max_area=max(30, int(0.006 * H * W)),
                                   max_variation=0.35, min_diversity=0.2)
            regions, boxes = mser.detectRegions(polarity)
        except cv2.error:
            continue
        if regions is None:
            continue
        for region in regions:
            comp = np.zeros((H, W), np.uint8)
            pts = region.reshape(-1, 1, 2)
            cv2.fillPoly(comp, [pts], 255)
            st = _component_stats(comp, g)
            if st and _keep(st, H, W):
                accepted.append(st["box"])

    if len(accepted) < min_chars and not aggressive:
        return []

    # ---- group glyphs into words / lines ---------------------------------
    canvas = np.zeros((H + 4, W + 4), np.uint8)
    for (x, y, w, h) in accepted:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), 255, -1)

    med_h = float(np.median([b[3] for b in accepted])) if accepted else 12.0
    kx = max(5, int(med_h * 1.4))
    ky = max(3, int(med_h * 0.6))
    canvas = cv2.dilate(canvas, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)), 1)

    n, _, stats, _ = cv2.connectedComponentsWithStats(canvas, 8)
    out: list[tuple[int, int, int, int]] = []
    for i in range(1, n):
        x, y, w, h = (int(stats[i, j]) for j in
                      (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < med_h * med_h * 0.8:            # ek akshar se chhota
            continue
        if h > 0.35 * H or w > 0.95 * W:          # poora frame = noise
            continue
        # count how many glyphs went into this group
        inside = sum(1 for (bx, by, bw, bh) in accepted
                     if bx + bw / 2 >= x - 2 and bx + bw / 2 <= x + w + 2
                     and by + bh / 2 >= y - 2 and by + bh / 2 <= y + h + 2)
        if inside < (1 if aggressive else 2):
            continue
        out.append((max(0, x - 2), max(0, y - 2), min(w + 4, W), min(h + 4, H)))
    return out


def text_mask(gray: np.ndarray, boxes, pad: int = 2) -> np.ndarray:
    """Tight *filled* mask of the text inside those boxes (not just outlines)."""
    H, W = gray.shape[:2]
    mask = np.zeros((H, W), np.uint8)
    if not boxes:
        return mask
    g = gray if gray.dtype == np.uint8 else gray.astype(np.uint8)
    k = 31 if min(H, W) > 40 else (min(H, W) // 2 * 2 + 1)
    loc = cv2.medianBlur(g, k)
    res = g.astype(np.float32) - loc.astype(np.float32)
    mad = float(np.median(np.abs(res - np.median(res)))) * 1.4826 + 1e-6
    ink = (np.abs(res) > max(4.0, 2.2 * mad)).astype(np.uint8) * 255

    kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for (x, y, w, h) in boxes:
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        crop = ink[y0:y1, x0:x1]
        if not crop.any():
            continue
        # Poori ink lo (glyph filter mat lagao - text ke outline/antialiasing
        # bhi hatne chahiye), phir close karke solid bana do taaki inpainter ko
        # filled region mile. Recall > precision yahan sahi trade-off hai:
        # "koi bhi text nahi chahiye".
        crop = cv2.morphologyEx(crop, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1)
        crop = cv2.dilate(cv2.morphologyEx(crop, cv2.MORPH_CLOSE, kk, 1), kk, 1)
        mask[y0:y1, x0:x1] = cv2.bitwise_or(mask[y0:y1, x0:x1], crop)
    return mask


# --------------------------------------------------------------------------- #
class OcrTextDetector:
    """Optional learned text detector - better recall on fancy fonts.

        pip install easyocr          # ya paddleocr / python-doctr
        det = OcrTextDetector("easyocr")
        boxes = det(gray_or_bgr)      # [(x, y, w, h), ...]

    EasyOCR/PaddleOCR bhi confidence aur text string dete hain - unhe use karke
    aap sirf "real text" hatana vs "logo ke andar ka text" rakhna decide kar
    sakte ho.
    """

    def __init__(self, engine: str = "easyocr", langs=("en",), gpu: bool = False):
        self.engine = engine
        self._impl = None
        self._langs = list(langs)
        self._gpu = gpu

    def _load(self):
        if self._impl is not None:
            return
        if self.engine == "easyocr":
            import easyocr  # type: ignore

            self._impl = easyocr.Reader(self._langs, gpu=self._gpu)
        elif self.engine == "paddleocr":
            from paddleocr import PaddleOCR  # type: ignore

            self._impl = PaddleOCR(use_angle_cls=False, lang=self._langs[0])
        else:
            raise ValueError(f"unknown ocr engine {self.engine!r}")

    def __call__(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        self._load()
        bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        boxes: list[tuple[int, int, int, int]] = []
        if self.engine == "easyocr":
            for (pts, _text, _conf) in self._impl.readtext(bgr):
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                x0, y0 = int(min(xs)), int(min(ys))
                boxes.append((x0, y0, int(max(xs) - x0), int(max(ys) - y0)))
        else:
            res = self._impl.ocr(bgr, cls=False)
            for line in res or []:
                for item in line:
                    pts = item[0]
                    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                    x0, y0 = int(min(xs)), int(min(ys))
                    boxes.append((x0, y0, int(max(xs) - x0), int(max(ys) - y0)))
        return boxes


def remove_text_from_mask(mask: np.ndarray, gray: np.ndarray,
                          detector=None, aggressive: bool = True) -> np.ndarray:
    """Existing mask ∪ detected text."""
    boxes = detector(gray) if detector else detect_text_regions(gray, aggressive=aggressive)
    if not boxes:
        return mask
    tmask = text_mask(gray, boxes)
    return cv2.bitwise_or(mask, tmask)
