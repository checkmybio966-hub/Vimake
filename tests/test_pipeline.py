"""End-to-end check on the synthetic samples.

    python tools/make_sample.py && python tests/test_pipeline.py

Reports, for each asset:
  * detection IoU  - how well the auto-detector localises the watermark
  * PSNR before    - watermarked vs clean  (baseline)
  * PSNR after     - processed  vs clean   (should be clearly higher)
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

from core import media, pipeline  # noqa: E402
from core.detect_image import detect_watermark_image  # noqa: E402
from core.maskops import dilate, ensure_mask  # noqa: E402

SAMPLES = ROOT / "samples"
TMP = ROOT / "runtime" / "test"
TMP.mkdir(parents=True, exist_ok=True)


def reencode_baseline(src: Path, n: int = 40) -> float:
    """PSNR of the watermarked video after a plain re-encode.

    Comparing the processed output against the ORIGINAL file would silently
    charge the h264 generational loss to the algorithm, so every video number
    in this report is measured against a copy that went through the same
    encoder with no processing at all.
    """
    tmp = str(TMP / "reencoded.mp4")
    info = media.probe(str(src))
    media.remux_copy(str(src), str(src), tmp) if False else None
    with media.VideoWriter(tmp, info.width, info.height, info.fps,
                           audio_src=None, crf=18) as w:
        for f in media.decode_frames(str(src), max_side=960, fps=info.fps):
            w.write(f)
    cl = media.read_frames(str(src).replace("_wm", "_clean"), fps=25)[:n]
    rc = media.read_frames(tmp, fps=25)[:n]
    k = min(len(cl), len(rc))
    return float(np.mean([psnr(rc[i], cl[i]) for i in range(k)]))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32); b = b.astype(np.float32)
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse < 1e-6 else 10 * np.log10(255.0 ** 2 / mse)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = ensure_mask(a) > 127; b = ensure_mask(b) > 127
    inter = float(np.logical_and(a, b).sum()); uni = float(np.logical_or(a, b).sum())
    return inter / uni if uni else 0.0


def test_image() -> None:
    src, clean, gt = (SAMPLES / n for n in ("photo_wm.png", "photo_clean.png", "photo_mask.png"))
    frame = cv2.imread(str(src))
    det = detect_watermark_image(frame)
    print(f"[image]  detect={det.found} score={det.score:.2f} method={det.method} "
          f"mask-IoU={iou(det.mask if det.found else np.zeros_like(frame[:, :, 0]), cv2.imread(str(gt), 0)):.3f}")
    if not det.found:
        print("[image]  FAIL: nothing detected"); return
    out = str(TMP / "photo_out.png")
    pipeline.process_image(str(src), out, mask=det.mask)
    before = psnr(cv2.imread(str(src)), cv2.imread(str(clean)))
    after = psnr(cv2.imread(out), cv2.imread(str(clean)))
    print(f"[image]  PSNR {before:.2f} dB -> {after:.2f} dB  ({'OK' if after > before else 'FAIL'})")


def test_text_image() -> None:
    """Captions / timestamps / @handles - "image me koi bhi text nahi chahiye"."""
    src, clean, gt = (SAMPLES / n for n in ("text_wm.png", "text_clean.png", "text_mask.png"))
    if not src.exists():
        print("[text]   sample missing - run tools/make_sample.py")
        return
    frame = cv2.imread(str(src))
    det = detect_watermark_image(frame, remove_text=True)
    g = cv2.imread(str(gt), 0)
    print(f"[text]   detect={det.found} score={det.score:.2f} method={det.method} "
          f"mask-IoU={iou(det.mask if det.found else np.zeros_like(g), g):.3f} {det.notes}")
    if not det.found:
        print("[text]   FAIL: nothing detected")
        return
    out = str(TMP / "text_out.png")
    pipeline.process_image(str(src), out, mask=det.mask)
    before = psnr(cv2.imread(str(src)), cv2.imread(str(clean)))
    after = psnr(cv2.imread(out), cv2.imread(str(clean)))
    print(f"[text]   PSNR {before:.2f} dB -> {after:.2f} dB ({'OK' if after > before else 'FAIL'})")


def test_video(name: str, expect_kind: str) -> None:
    src = SAMPLES / f"{name}_wm.mp4"
    clean_p = SAMPLES / f"{name}_clean.mp4"
    gt = cv2.imread(str(SAMPLES / f"{name}_mask.png"), 0)

    det = pipeline.detect_video(str(src))
    kind_ok = det.kind == expect_kind
    print(f"[{name}] detect={det.found} score={det.score:.2f} kind={det.kind} "
          f"(expected {expect_kind} {'OK' if kind_ok else 'MISMATCH'}) method={det.method}")
    if det.found:
        print(f"[{name}] mask-IoU={iou(det.mask, cv2.resize(gt, (det.mask.shape[1], det.mask.shape[0]))):.3f}")

    out = str(TMP / f"{name}_out.mp4")
    pipeline.process_video(str(src), out, mode="auto")
    cl = media.read_frames(str(clean_p), fps=25)[:40]
    pr = media.read_frames(out, fps=25)[:40]
    k = min(len(cl), len(pr))
    b = reencode_baseline(src, n=40)
    a = float(np.mean([psnr(pr[i], cl[i]) for i in range(k)]))
    print(f"[{name}] PSNR {b:.2f} dB (re-encoded, unprocessed) -> {a:.2f} dB "
          f"over {k} frames ({'OK' if a > b else 'FAIL'})  gain={a - b:+.2f} dB")


if __name__ == "__main__":
    if not (SAMPLES / "photo_wm.png").exists():
        print("run tools/make_sample.py first"); sys.exit(1)
    test_image()
    test_text_image()
    test_video("video_static", "static")
    test_video("video_moving", "moving")
