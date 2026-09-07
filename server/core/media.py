"""Thin ffmpeg wrapper: probe, decode frames to numpy, encode numpy back to mp4.

The app never shells out to random binaries: it uses the ffmpeg shipped by the
`imageio-ffmpeg` wheel unless FFMPEG_BIN points somewhere else. That keeps the
project installable with plain `pip install -r requirements.txt` (no apt, no
system ffmpeg).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

_BIN_CACHE: Optional[str] = None


def ffmpeg_bin() -> str:
    global _BIN_CACHE
    if _BIN_CACHE:
        return _BIN_CACHE
    for cand in (
        os.environ.get("FFMPEG_BIN", ""),
        _try_imageio(),
        shutil.which("ffmpeg") or "",
        "ffmpeg",
    ):
        if cand and (os.path.exists(cand) or shutil.which(cand)):
            _BIN_CACHE = cand
            return cand
    raise RuntimeError("ffmpeg not found. Install with: pip install imageio-ffmpeg")


def _try_imageio() -> str:
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return ""


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float
    has_audio: bool
    vcodec: str = ""
    nb_frames: int = 0


def probe(path: str) -> VideoInfo:
    """Parse ffmpeg's stderr (no ffprobe needed)."""
    p = subprocess.run(
        [ffmpeg_bin(), "-hide_banner", "-i", path],
        capture_output=True, text=True, errors="ignore",
    )
    txt = p.stderr or ""

    dur = 0.0
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", txt)
    if m:
        dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))

    w = h = 0
    fps = 0.0
    vcodec = ""
    # ffmpeg prints e.g.:
    #   Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(...), 640x360, ...
    vm = re.search(
        r"Stream #\d+:\d+(?:\[[^\]]*\])?(?:\([^)]*\))?.*?Video:\s*(\w+).*?,\s*(\d{2,5})x(\d{2,5})",
        txt,
    )
    if vm:
        vcodec, w, h = vm.group(1), int(vm.group(2)), int(vm.group(3))
        fm = re.search(r"(\d+(?:\.\d+)?)\s*fps", txt)
        if fm:
            fps = float(fm.group(1))
    has_audio = bool(re.search(r"Stream #\d+:\d+.*?:\s*Audio:", txt))

    if not w or not h:
        raise RuntimeError(f"Could not read video stream from {path!r}")
    if not fps:
        fps = 25.0
    return VideoInfo(w, h, fps, dur, has_audio, vcodec, int(dur * fps))


def scaled_size(w: int, h: int, max_side: Optional[int]) -> tuple[int, int]:
    if not max_side or max(w, h) <= max_side:
        return w - (w % 2), h - (h % 2)
    s = max_side / float(max(w, h))
    nw, nh = int(round(w * s)), int(round(h * s))
    return max(2, nw - nw % 2), max(2, nh - nh % 2)


def decode_frames(
    path: str,
    max_side: Optional[int] = None,
    fps: Optional[float] = None,
    pix_fmt: str = "bgr24",
) -> Iterator[np.ndarray]:
    """Stream frames as uint8 BGR arrays, optionally downscaled / resampled."""
    info = probe(path)
    w, h = scaled_size(info.width, info.height, max_side)

    vf = []
    if max_side and (info.width, info.height) != (w, h):
        vf.append(f"scale={w}:{h}:flags=bicubic")
    if fps and fps < info.fps:
        vf.append(f"fps={fps:.4f}")

    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-i", path]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ["-f", "rawvideo", "-pix_fmt", pix_fmt, "-an", "-"]

    bpp = 3 if pix_fmt == "bgr24" else 1
    frame_bytes = w * h * bpp
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=frame_bytes * 4)
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)  # type: ignore[union-attr]
            if not buf or len(buf) < frame_bytes:
                break
            arr = np.frombuffer(buf, dtype=np.uint8)
            yield arr.reshape(h, w, bpp) if bpp == 3 else arr.reshape(h, w)
    finally:
        try:
            proc.stdout.close()  # type: ignore[union-attr]
        except Exception:
            pass
        proc.wait(timeout=30)


def read_frames(path: str, max_side: Optional[int] = None, fps: Optional[float] = None) -> list[np.ndarray]:
    return list(decode_frames(path, max_side=max_side, fps=fps))


class VideoWriter:
    """Pipe raw BGR frames to ffmpeg -> h264 mp4, muxing the source audio back."""

    def __init__(self, out_path: str, width: int, height: int, fps: float,
                 audio_src: Optional[str] = None, crf: int = 18, preset: str = "veryfast"):
        self.out_path = out_path
        self.w, self.h, self.fps = width, height, fps
        cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "-"]
        if audio_src:
            cmd += ["-i", audio_src]
        cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        if audio_src:
            cmd += ["-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-b:a", "160k", "-shortest"]
        else:
            cmd += ["-an"]
        cmd += [out_path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self._err: Optional[str] = None
        self._t = threading.Thread(target=self._drain, daemon=True)
        self._t.start()

    def _drain(self) -> None:
        try:
            self._err = (self.proc.stderr.read() or b"").decode("utf-8", "ignore")  # type: ignore[union-attr]
        except Exception:
            pass

    def write(self, frame: np.ndarray) -> None:
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        if frame.shape[0] != self.h or frame.shape[1] != self.w:
            import cv2

            frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        self.proc.stdin.write(frame.tobytes())  # type: ignore[union-attr]

    def close(self) -> None:
        try:
            self.proc.stdin.close()  # type: ignore[union-attr]
        except Exception:
            pass
        rc = self.proc.wait(timeout=180)
        self._t.join(timeout=5)
        if rc != 0:
            raise RuntimeError(f"ffmpeg encode failed ({rc}): {(self._err or '').strip()[:500]}")

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *a) -> None:
        self.close()


def first_frame(path: str, max_side: Optional[int] = None) -> Optional[np.ndarray]:
    for f in decode_frames(path, max_side=max_side, fps=None):
        return f.copy()
    return None


def remux_copy(src_video: str, audio_src: str, out_path: str) -> None:
    """Replace a video's audio track with the original (used after pure-video encodes)."""
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y",
           "-i", src_video, "-i", audio_src,
           "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac",
           "-b:a", "160k", "-shortest", out_path]
    subprocess.run(cmd, check=True, capture_output=True)
