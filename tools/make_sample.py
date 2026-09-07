"""Generate synthetic test media (image + video) with known watermarks.

Everything is generated in pairs:
    xxx_wm.*     -> with the watermark
    xxx_clean.*  -> the same frames WITHOUT it
    xxx_mask.png -> ground truth: exactly where the watermark is

That lets `python tests/test_pipeline.py` measure real numbers (mask IoU and
PSNR) instead of "looks fine to me".

    python tools/make_sample.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

from core import media  # noqa: E402

OUT = ROOT / "samples"
W, H, FPS = 640, 360, 25
N = 100                       # 4 seconds


# --------------------------------------------------------------------------- #
def background(t: float) -> np.ndarray:
    """A realistic scene: smooth static gradient, a couple of slow subjects,
    mild film grain. Not a stress test - ordinary footage."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    f = np.zeros((H, W, 3), np.float32)
    f[..., 0] = 90 + 110 * (xx / W)
    f[..., 1] = 70 + 120 * (yy / H)
    f[..., 2] = 110 + 90 * ((xx + yy) / (W + H))

    for k, (sx, sy, rad, col) in enumerate((
            (0.30, 0.55, 46, (70, 190, 130)),
            (0.72, 0.40, 34, (210, 140, 90)),
            (0.55, 0.80, 26, (120, 120, 220)))):
        a = t * (0.35 + 0.08 * k) + k * 2.1
        cx = W * (sx + 0.10 * np.sin(a))
        cy = H * (sy + 0.07 * np.cos(a * 0.9 + k))
        cv2.circle(f, (int(cx), int(cy)), rad, col, -1, lineType=cv2.LINE_AA)

    f += np.random.normal(0, 1.5, f.shape).astype(np.float32)      # mild grain
    return np.clip(f, 0, 255).astype(np.uint8)


def draw_fixed_watermark(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Semi-transparent corner logo + timestamp (the classic 'fixed' case)."""
    lay = frame.copy()
    cv2.putText(lay, "@vmake-demo", (W - 250, H - 26), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(lay, "2026.09.07", (18, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (235, 235, 235), 2, cv2.LINE_AA)
    cv2.circle(lay, (W - 290, H - 36), 16, (255, 255, 255), -1, cv2.LINE_AA)
    mask = (cv2.absdiff(lay, frame).max(axis=2) > 8).astype(np.uint8) * 255
    out = cv2.addWeighted(lay, 0.55, frame, 0.45, 0)
    return out, mask


def draw_moving_watermark(frame: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Floating AI-generation mark - drifts slowly (like the real ones do)."""
    lay = frame.copy()
    x = int(W * (0.5 + 0.22 * np.sin(t * 0.55)))
    y = int(H * (0.5 + 0.18 * np.cos(t * 0.42)))
    cv2.putText(lay, "Sora", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                (255, 255, 255), 3, cv2.LINE_AA)
    mask = (cv2.absdiff(lay, frame).max(axis=2) > 8).astype(np.uint8) * 255
    out = cv2.addWeighted(lay, 0.55, frame, 0.45, 0)
    return out, mask


# --------------------------------------------------------------------------- #
def build_video(name: str, moving: bool) -> None:
    wm_path, cl_path = OUT / f"{name}_wm.mp4", OUT / f"{name}_clean.mp4"
    masks = []
    with media.VideoWriter(str(wm_path), W, H, FPS) as ww, \
            media.VideoWriter(str(cl_path), W, H, FPS) as wc:
        for i in range(N):
            t = i / FPS
            base = background(t)
            if moving:
                wm, m = draw_moving_watermark(base, t)
            else:
                wm, m = draw_fixed_watermark(base)
            ww.write(wm)
            wc.write(base)
            masks.append(m)
    gt = np.stack(masks).max(axis=0)
    cv2.imwrite(str(OUT / f"{name}_mask.png"), gt)
    print(f"  {wm_path.name:26s} {cl_path.name:26s} gt coverage={gt.mean():.4f}")


def draw_text_block(base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Caption + timestamp + @handle - the 'koi bhi text nahi chahiye' case."""
    lay = base.copy()
    # meme-style caption (bottom centre, white with dark outline)
    cv2.putText(lay, "MORNING VIBES", (90, H - 46), cv2.FONT_HERSHEY_DUPLEX,
                1.15, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(lay, "MORNING VIBES", (90, H - 46), cv2.FONT_HERSHEY_DUPLEX,
                1.15, (255, 255, 255), 2, cv2.LINE_AA)
    # timestamp (top left)
    cv2.putText(lay, "2026-09-07 10:42", (20, 36), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (250, 250, 250), 2, cv2.LINE_AA)
    # handle (bottom right)
    cv2.putText(lay, "@travel.diaries", (W - 232, H - 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (255, 255, 255), 2, cv2.LINE_AA)
    mask = (cv2.absdiff(lay, base).max(axis=2) > 8).astype(np.uint8) * 255
    out = cv2.addWeighted(lay, 0.88, base, 0.12, 0)
    return out, mask


def build_text_image() -> None:
    base = background(2.0)
    wm, mask = draw_text_block(base)
    cv2.imwrite(str(OUT / "text_wm.png"), wm)
    cv2.imwrite(str(OUT / "text_clean.png"), base)
    cv2.imwrite(str(OUT / "text_mask.png"), mask)
    print(f"  text_wm.png                  text_clean.png             gt coverage={mask.mean():.4f}")


def build_image() -> None:
    base = background(1.0)
    lay = base.copy()
    # tiled repeated text watermark: 3x3 grid, translucent
    for gy in range(3):
        for gx in range(3):
            p = (60 + gx * 240, 90 + gy * 110)
            cv2.putText(lay, "DEMO", p, cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (245, 245, 245), 2, cv2.LINE_AA)
    mask = (cv2.absdiff(lay, base).max(axis=2) > 8).astype(np.uint8) * 255
    wm = cv2.addWeighted(lay, 0.5, base, 0.5, 0)
    cv2.imwrite(str(OUT / "photo_wm.png"), wm)
    cv2.imwrite(str(OUT / "photo_clean.png"), base)
    cv2.imwrite(str(OUT / "photo_mask.png"), mask)
    print(f"  photo_wm.png                photo_clean.png            gt coverage={mask.mean():.4f}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("building samples in", OUT)
    build_image()
    build_text_image()
    build_video("video_static", moving=False)
    build_video("video_moving", moving=True)
    print("done")


if __name__ == "__main__":
    main()
