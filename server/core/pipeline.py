"""The actual removal pipeline.

    image  ->  detect (or user mask)  ->  inpaint  ->  write

    video  ->  probe
           ->  sample frames
           ->  detect  ── kind == "moving" ──> CLEAN PLATE  (temporal median with
           |                                   camera-motion compensation)
           └─  kind == "static" ──> INPAINT VIDEO (optical-flow propagation of
                                    real pixels from neighbouring frames, then
                                    spatial diffusion for whatever is left)

Both video paths stream: only a small window of frames is ever in memory, so a
3 minute 1080p clip does not need 6 GB of RAM.
"""
from __future__ import annotations

import shutil
from collections import deque
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from . import config as cfgmod
from . import media
from .detect_image import detect_watermark_image
from .detect_video import adaptive_alpha, detect_watermark_video
from .flow import MotionModel, compute_flow, compose_step, warp_image
from .inpaint import get_backend
from .inpaint.opencv_backend import OpenCVBackend
from .maskops import dilate, ensure_mask, feather, overlay_preview, to_png_bytes, union
from .types import Detection

Progress = Callable[[float, str], None]
cfg = cfgmod.cfg


# --------------------------------------------------------------------------- #
#  images
# --------------------------------------------------------------------------- #
def process_image(src: str, dst: str, mask: Optional[np.ndarray] = None,
                  auto: bool = True, backend_name: Optional[str] = None,
                  remove_text: bool = False, text_detector=None,
                  progress: Optional[Progress] = None) -> Detection:
    _p(progress, 0.05, "loading")
    frame = cv2.imread(src, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"cannot read image: {src}")
    h, w = frame.shape[:2]
    if max(h, w) > cfg.max_side:
        s = cfg.max_side / float(max(h, w))
        frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    if mask is not None:
        det = Detection(mask=ensure_mask(mask, frame.shape), score=1.0,
                        method="manual-brush", kind="static")
    elif auto:
        _p(progress, 0.25, "detecting watermark" + (" + text" if remove_text else ""))
        det = detect_watermark_image(frame, remove_text=remove_text,
                                     text_detector=text_detector)
    else:
        raise ValueError("no mask supplied and auto-detection disabled")

    if not det.found:
        raise LookupError("WATERMARK_NOT_FOUND")

    _p(progress, 0.5, f"inpainting ({det.method})")
    m = dilate(det.mask, cfg.mask_dilate_px)
    m = feather(m, cfg.mask_feather_px)
    backend = get_backend(backend_name)
    out = backend.inpaint(frame, m)

    _p(progress, 0.9, "writing")
    cv2.imwrite(dst, out, [int(cv2.IMWRITE_JPEG_QUALITY), 95] if dst.lower().endswith(".jpg")
                else [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    _p(progress, 1.0, "done")
    return det


# --------------------------------------------------------------------------- #
#  video - planning
# --------------------------------------------------------------------------- #
def plan_video(path: str) -> tuple[media.VideoInfo, int, int, float]:
    info = media.probe(path)
    w, h = media.scaled_size(info.width, info.height, cfg.max_side)
    fps = info.fps
    dur = info.duration or (info.nb_frames / fps if info.nb_frames else 0.0)
    if dur > 0 and dur * fps > cfg.max_frames:
        fps = max(1.0, cfg.max_frames / dur)
    return info, w, h, fps


def sample_for_detection(path: str) -> list[np.ndarray]:
    info = media.probe(path)
    dur = info.duration or 10.0
    rate = max(0.2, min(info.fps, cfg.detect_samples / max(dur, 0.5)))
    frames = media.read_frames(path, max_side=cfg.detect_width, fps=rate)
    return frames[: cfg.detect_samples * 2]


def detect_video(path: str, target_shape=None) -> Detection:
    frames = sample_for_detection(path)
    if len(frames) < 4:
        return Detection(mask=np.zeros((1, 1), np.uint8), method="not-enough-frames")
    det = detect_watermark_video(frames, target_shape=target_shape)
    if det.found:
        return det
    # fallback: try the still-image detector on the temporal median frame
    stack = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f for f in frames])
    med = np.median(stack, axis=0).astype(np.uint8)
    bgr = cv2.cvtColor(med, cv2.COLOR_GRAY2BGR)
    img_det = detect_watermark_image(bgr)
    if img_det.found and img_det.score > 0.45:
        img_det.kind = "static"
        img_det.method += " (on median frame)"
        if target_shape is not None:
            img_det.mask = cv2.resize(img_det.mask, (target_shape[1], target_shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
        return img_det
    return det


# --------------------------------------------------------------------------- #
#  video removal - ONE propagation engine, two modes
# --------------------------------------------------------------------------- #
def _shape_gate(alpha: np.ndarray, area: int) -> np.ndarray:
    """Keep only blob-shaped outlier regions.

    A watermark is a compact, solid glyph/logo. The other thing that produces
    temporal outliers is a moving object's EDGE, which shows up as thin arcs
    and rings - those fail the solidity/extent test below and get dropped.
    """
    if not np.any(alpha > 0.35):
        return np.zeros_like(alpha)
    binary = ((alpha > 0.35) * 255).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    keep = np.zeros_like(binary)
    min_a, max_a = 0.0002 * area, 0.06 * area
    for i in range(1, n):
        a = float(stats[i, cv2.CC_STAT_AREA])
        if a < min_a or a > max_a:
            continue
        comp = (lab == i).astype(np.uint8) * 255
        cnt, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnt:
            continue
        c = max(cnt, key=cv2.contourArea)
        hull = cv2.convexHull(c)
        ha = float(cv2.contourArea(hull)) + 1e-6
        x, y, bw, bh = cv2.boundingRect(c)
        solidity = cv2.contourArea(c) / ha
        extent = a / float(bw * bh + 1e-6)
        aspect = bw / float(bh + 1e-6)
        if solidity >= 0.45 and extent >= 0.25 and 0.15 <= aspect <= 9.0:
            keep[lab == i] = 255
    if not keep.any():
        return np.zeros_like(alpha)
    return alpha * (cv2.blur(keep, (3, 3)) / 255.0)


def _plate_stats(stack: np.ndarray, hw: int, hh: int):
    """Robust background estimate + residual statistics for a neighbour stack."""
    m = stack.shape[0]
    g = np.empty((m, hh, hw), np.float32)
    for i in range(m):
        small = cv2.resize(stack[i], (hw, hh), interpolation=cv2.INTER_AREA)
        g[i] = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

    plate = np.median(g, axis=0)
    res = np.abs(g - plate[None])
    med = np.median(res, axis=0)
    mad = np.median(np.abs(res - med[None]), axis=0)
    keep = res <= (med[None] + 3.0 * 1.4826 * mad[None])          # clean samples

    with np.errstate(invalid="ignore"):
        plate = np.nanmedian(np.where(keep, g, np.nan), axis=0)
    plate = np.nan_to_num(plate, nan=float(np.median(g)))
    res2 = np.where(keep, np.abs(g - plate[None]), np.nan)
    with np.errstate(invalid="ignore"):
        med_d = np.nanmedian(res2, axis=0)
        mad_d = np.nanmedian(np.abs(res2 - med_d[None]), axis=0)
    return g, np.nan_to_num(med_d, nan=0.0), np.nan_to_num(mad_d, nan=1.0), keep, plate


def _moving_alpha(raw: np.ndarray, warped: Optional[np.ndarray], ref: np.ndarray,
                  hw: int, hh: int, z_thresh: float, prior: Optional[np.ndarray],
                  shape_filter: bool, plate_mode: str = "median"):
    """Where is the floating mark in THIS frame, and what should replace it?

    Two candidate background models are built and the TIGHTER one wins:

      * RAW     - the neighbours exactly as they are. Best when the camera is
                  steady: no resampling, no flow error, so the residual spread
                  is tiny and the mark stands out miles above it.
      * WARPED  - neighbours brought into this frame's geometry with optical
                  flow. Needed when the camera (or the subject) really moves,
                  but warping adds its own noise, which inflates the residual
                  spread and hides the mark.

    We measure the residual spread of both (median MAD) and take the smaller
    one - that choice is what makes the same code work on a tripod shot and on
    a handheld pan.
    """
    g_r, med_r, mad_r, keep_r, plate_r = _plate_stats(raw, hw, hh)
    best = (raw, g_r, med_r, mad_r, keep_r, plate_r, False)
    if warped is not None and warped.shape == raw.shape:
        g_w, med_w, mad_w, keep_w, plate_w = _plate_stats(warped, hw, hh)
        # warping must clearly pay for itself before we trust it
        if float(np.median(med_w)) < float(np.median(med_r)) * 0.9:
            best = (warped, g_w, med_w, mad_w, keep_w, plate_w, True)

    stack, g_small, med_d, mad_d, keep, plate_g, used_warp = best
    if plate_mode == "percentile":
        p35 = np.percentile(g_small, 35, axis=0).astype(np.float32)
        p65 = np.percentile(g_small, 65, axis=0).astype(np.float32)
        ref_small = cv2.cvtColor(cv2.resize(ref, (hw, hh), interpolation=cv2.INTER_AREA),
                                 cv2.COLOR_BGR2GRAY).astype(np.float32)
        farther = np.abs(ref_small - p35) > np.abs(ref_small - p65)
        plate_g = np.where(farther, p35, p65).astype(np.float32)

    H, W = ref.shape[:2]
    ref_small = cv2.cvtColor(cv2.resize(ref, (hw, hh), interpolation=cv2.INTER_AREA),
                             cv2.COLOR_BGR2GRAY).astype(np.float32)
    d_ref = cv2.blur(np.abs(ref_small - plate_g), (3, 3))
    alpha = cv2.resize(adaptive_alpha(d_ref, med_d, mad_d, k=z_thresh), (W, H),
                       interpolation=cv2.INTER_LINEAR)
    if prior is not None:
        alpha = alpha * prior
    if shape_filter:
        alpha = alpha * _shape_gate(alpha, W * H)
    if float(alpha.max()) < 0.08:
        return None, None, None, False

    ys, xs = np.nonzero(alpha > 0.02)
    if xs.size == 0:
        return None, None, None, False
    y0, y1 = max(0, int(ys.min()) - 3), min(H, int(ys.max()) + 4)
    x0, x1 = max(0, int(xs.min()) - 3), min(W, int(xs.max()) + 4)

    # ---- full-resolution clean plate --------------------------------------
    # per-pixel median over the neighbours that were NOT outliers at that pixel
    m = stack.shape[0]
    bw, bh = x1 - x0, y1 - y0
    acc = np.full((m, bh, bw, 3), np.nan, np.float32)
    for i in range(m):
        ki = cv2.resize(keep[i].astype(np.uint8), (bw, bh), interpolation=cv2.INTER_NEAREST) > 0
        crop = stack[i][y0:y1, x0:x1].astype(np.float32)
        acc[i][ki] = crop[ki]
    with np.errstate(invalid="ignore"):
        plate_full = np.nanmedian(acc, axis=0)
    fallback = np.median(stack[:, y0:y1, x0:x1].astype(np.float32), axis=0)
    plate_full = np.where(np.isfinite(plate_full), plate_full, fallback)
    return alpha, plate_full.astype(np.float32), (y0, y1, x0, x1), used_warp


def propagate_video(src: str, dst: str, w: int, h: int, fps: float,
                    mask: Optional[np.ndarray] = None, mode: str = "static",
                    window: Optional[int] = None, stride: int = 1,
                    z_thresh: Optional[float] = None, shape_filter: bool = True,
                    plate_mode: str = "",
                    temporal_smooth: float = 0.35, progress: Optional[Progress] = None,
                    total: int = 0) -> int:
    """Fill the watermark region with REAL pixels taken from other frames.

    Every neighbour frame is warped into the current frame's geometry with
    dense optical flow, so the moving background stays aligned and the fill is
    sharp instead of a smeared average. Two modes differ only in WHERE we fill:

      mode="static"  a FIXED logo. The watermark is in every frame, so time
                     cannot reveal what is behind it - we must synthesise it.
                     Fill = weighted mean of every neighbour that sees the
                     scene point (i.e. is not masked there), then diffusion
                     inpainting for the leftovers.  (poor man's ProPainter)

      mode="moving"  a FLOATING mark (Sora/Veo/Kling). Here time DOES reveal the
                     background: at any pixel the mark is present only while it
                     drifts over it. Fill = per-pixel MEDIAN of the warped
                     neighbours, applied only where the current frame is an
                     OUTLIER for its own history (robust z-score). The window
                     must be longer than the mark's dwell time on a pixel.
    """
    window = max(1, int(window or (cfg.moving_window if mode == "moving" else cfg.inpaint_window)))
    stride = max(1, int(stride or (cfg.moving_stride if mode == "moving" else 1)))
    z_thresh = float(z_thresh or cfg.clean_plate_z)
    plate_mode = plate_mode or (cfg.moving_plate if mode == "moving" else "median")
    closer = OpenCVBackend()

    soft = hole = None
    prior = None
    if mask is not None and np.asarray(mask).any():
        m_full = ensure_mask(cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST))
        if mode == "static":
            soft = feather(m_full, cfg.mask_feather_px).astype(np.float32) / 255.0
            hole = m_full > 127
            if not hole.any():
                shutil.copyfile(src, dst)
                return 0
        else:
            ink = (m_full > 127).astype(np.float32)
            prior = np.clip(cv2.blur(ink, (9, 9)) * 3.0, 0, 1).astype(np.float32)
            # the union mask only says "the mark passes SOMEWHERE here"; keep a
            # floor so a slightly-off detection slows us down instead of
            # silently disabling the removal
            prior = np.clip(0.25 + 0.75 * prior, 0, 1)

    scale = 0.5
    fw, fh = max(16, int(w * scale)), max(16, int(h * scale))

    # ---- pass 1: forward + backward flow for every consecutive pair --------
    _p(progress, 0.02, "estimating motion")
    fwd: list[np.ndarray] = []
    bwd: list[np.ndarray] = []
    prev = None
    for frame in media.decode_frames(src, max_side=cfg.max_side, fps=fps):
        pg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            ps = cv2.resize(prev, (fw, fh), interpolation=cv2.INTER_AREA)
            cs = cv2.resize(pg, (fw, fh), interpolation=cv2.INTER_AREA)
            fwd.append(compute_flow(ps, cs, scale=1.0).astype(np.float16))
            bwd.append(compute_flow(cs, ps, scale=1.0).astype(np.float16))
        prev = pg
        if total and len(fwd) >= total - 1:
            break
    npair = len(fwd)
    _p(progress, 0.15, f"motion ready ({npair} frame pairs)")

    def displacement(t: int, d: int) -> np.ndarray:
        """Displacement field mapping frame t coordinates -> frame t+d."""
        acc = np.zeros((fh, fw, 2), np.float32)
        if d > 0:
            for k in range(t, min(t + d, npair)):
                acc = compose_step(acc, fwd[k].astype(np.float32))
        else:
            for k in range(t - 1, max(t + d - 1, -1), -1):
                acc = compose_step(acc, bwd[k].astype(np.float32))
        return acc

    offsets = [d for d in range(-window, window + 1) if d != 0 and d % stride == 0]
    hw, hh = max(8, w // 2), max(8, h // 2)

    # ---- pass 2: streaming fill -------------------------------------------
    buf: deque[np.ndarray] = deque()
    base = -window                 # global index of buf[0] (leading pad frames < 0)
    n_written = 0
    next_emit = 0                  # global index of the next frame to write
    n_real = 0                     # how many real frames the decoder produced
    prev_out: Optional[np.ndarray] = None
    hist: deque[np.ndarray] = deque(maxlen=6)

    def emit(writer, ref_i: int) -> None:
        nonlocal prev_out, n_written
        ref_t = base + ref_i
        ref = buf[ref_i]

        disp_cache: dict[int, np.ndarray] = {}

        def _disp(d: int) -> np.ndarray:
            if d not in disp_cache:
                disp = displacement(ref_t, d)
                disp_full = cv2.resize(disp, (w, h), interpolation=cv2.INTER_LINEAR)
                disp_full[..., 0] *= w / float(fw)
                disp_full[..., 1] *= h / float(fh)
                disp_cache[d] = disp_full
            return disp_cache[d]

        raws: list[np.ndarray] = []
        warps: list[np.ndarray] = []
        for d in offsets:
            j, k = ref_t + d, ref_i + d
            if j < 0 or j > npair or k < 0 or k >= len(buf):
                continue
            raws.append(buf[k])
            warps.append(warp_image(buf[k], _disp(d)))

        if not raws:
            writer.write(ref)
            n_written += 1
            return

        if mode == "moving":
            alpha, plate_full, box, _uw = _moving_alpha(
                np.stack(raws).astype(np.uint8), np.stack(warps).astype(np.uint8),
                ref, hw, hh, z_thresh, prior, shape_filter, plate_mode)
            if alpha is None:
                writer.write(ref)
                n_written += 1
                return
            fill = np.zeros((h, w, 3), np.float32)
            (y0, y1, x0, x1) = box
            fill[y0:y1, x0:x1] = plate_full
            a = alpha[..., None]
        else:
            stack = np.stack(warps).astype(np.uint8)
            acc = np.zeros((h, w, 3), np.float32)
            wsum = np.zeros((h, w, 1), np.float32)
            valid_d: dict[int, np.ndarray] = {}
            for d, warped in zip(offsets, warps):
                j = ref_t + d
                if j < 0 or j > npair:
                    continue
                if d not in valid_d:
                    valid_d[d] = warp_image(255 - (hole * 255).astype(np.uint8), _disp(d),
                                            flags=cv2.INTER_NEAREST).astype(np.float32) / 255.0
                valid = valid_d[d]
                wt = float(np.exp(-abs(d) / max(1.0, window / 1.6)))
                acc += warped.astype(np.float32) * valid[..., None] * wt
                wsum += valid[..., None] * wt
            good = wsum > 1e-4
            fill = np.zeros((h, w, 3), np.float32)
            fill[good[..., 0]] = acc[good[..., 0]] / wsum[good[..., 0]]
            fill_u8 = np.clip(fill, 0, 255).astype(np.uint8)
            residual = (hole & ~good[..., 0]).astype(np.uint8) * 255
            if residual.any():
                fill_u8 = closer.close_holes(fill_u8, residual, radius=5)
            fill = fill_u8.astype(np.float32)
            a = soft[..., None]

        out = ref.astype(np.float32) * (1 - a) + fill * a
        if prev_out is not None and temporal_smooth > 0:
            out = out * (1 - temporal_smooth * a) + prev_out.astype(np.float32) * (temporal_smooth * a)
        out = np.clip(out, 0, 255).astype(np.uint8)
        writer.write(out)
        prev_out = out
        n_written += 1

    with media.VideoWriter(dst, w, h, fps, audio_src=src, crf=cfg.crf) as writer:
        for frame in media.decode_frames(src, max_side=cfg.max_side, fps=fps):
            if not buf:                       # left-pad so frame 0 gets a full window
                for _ in range(window):
                    buf.append(frame)
            buf.append(frame)
            n_real += 1
            if len(buf) < 2 * window + 1:
                continue
            emit(writer, window)
            next_emit += 1
            buf.popleft()
            base += 1
            if progress and total:
                progress(0.15 + 0.83 * min(1.0, n_written / float(total)),
                         f"{'clean plate' if mode == 'moving' else 'inpainting'} frame {n_written}/{total}")

        # tail: every real frame that has not been written yet, exactly once
        while next_emit < n_real and buf:
            while buf and base < next_emit:
                buf.popleft()
                base += 1
            if not buf:
                break
            emit(writer, 0)
            next_emit += 1
            buf.popleft()
            base += 1

    return n_written


# --------------------------------------------------------------------------- #
#  top level
# --------------------------------------------------------------------------- #
def process_video(src: str, dst: str, mask: Optional[np.ndarray] = None,
                  mode: str = "auto", backend_name: Optional[str] = None,
                  remove_text: bool = False, text_detector=None,
                  progress: Optional[Progress] = None) -> Detection:
    _p(progress, 0.01, "probing")
    info, w, h, fps = plan_video(src)
    total = int(min(info.nb_frames or (info.duration * fps), cfg.max_frames)) or 1

    if mask is not None:
        det = Detection(mask=ensure_mask(mask, (h, w)), score=1.0,
                        method="manual-brush", kind="static" if mode != "moving" else "moving")
    else:
        _p(progress, 0.05, "sampling frames")
        det = detect_video(src, target_shape=(h, w))
        if not det.found and not remove_text:
            raise LookupError("WATERMARK_NOT_FOUND")

    # ---- "koi bhi text nahi chahiye": subtitles / captions / timestamps -----
    # video me text lagbhag hamesha fixed hota hai, isliye use temporal median
    # frame par detect karna kaafi (aur 100x sasta) hai.
    if remove_text:
        from .detect_text import detect_text_regions, text_mask

        _p(progress, 0.08, "detecting text")
        frames = sample_for_detection(src)
        if frames:
            stack = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
                              for f in frames])
            med = np.median(stack, axis=0).astype(np.uint8)
            boxes = text_detector(med) if text_detector else detect_text_regions(med, aggressive=True)
            tmask = text_mask(med, boxes)
            if tmask.any():
                tmask = cv2.resize(tmask, (w, h), interpolation=cv2.INTER_NEAREST)
                base = det.mask if (det.found and det.mask is not None and det.mask.any()) \
                    else np.zeros((h, w), np.uint8)
                det.mask = ensure_mask(cv2.bitwise_or(ensure_mask(base, (h, w)), tmask))
                det.boxes = _boxes(det.mask)
                det.method = (det.method or "text") + "+text"
                det.score = max(float(det.score or 0), 0.85)
                det.notes = (det.notes or "") + f" +{len(boxes)} text region(s)"
                det.found_check = True
        if not (det.mask is not None and det.mask.any()):
            raise LookupError("WATERMARK_NOT_FOUND")

    kind = det.kind if mode == "auto" else mode
    if kind == "moving":
        _p(progress, 0.1, "moving watermark -> flow-warped temporal clean plate")
        propagate_video(src, dst, w, h, fps, mask=det.mask, mode="moving",
                        progress=progress, total=total)
    else:
        _p(progress, 0.1, "fixed watermark -> flow-guided inpainting")
        propagate_video(src, dst, w, h, fps, mask=det.mask, mode="static",
                        progress=progress, total=total)

    _p(progress, 1.0, "done")
    return det


def _p(progress: Optional[Progress], v: float, stage: str) -> None:
    if progress:
        progress(v, stage)


# --------------------------------------------------------------------------- #
#  debug helper: save the overlay the UI shows
# --------------------------------------------------------------------------- #
def save_preview(src: str, out: str, mask: np.ndarray) -> None:
    frame = media.first_frame(src, max_side=cfg.max_side)
    if frame is None:
        frame = cv2.imread(src)
    if frame is None:
        return
    m = ensure_mask(mask, frame.shape)
    cv2.imwrite(out, overlay_preview(frame, m))


def mask_png(mask: np.ndarray) -> bytes:
    return to_png_bytes(mask)


def combine_masks(auto_mask: Optional[np.ndarray], user_add: Optional[np.ndarray],
                  user_sub: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """auto ∪ brush  −  eraser"""
    parts = [m for m in (auto_mask, user_add) if m is not None and m.any()]
    if not parts:
        return None
    m = union(parts)
    if user_sub is not None and user_sub.any():
        m = cv2.bitwise_and(m, cv2.bitwise_not(ensure_mask(user_sub, m.shape)))
    return m
