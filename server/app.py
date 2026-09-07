"""Vmake-style watermark remover - HTTP API.

    POST /api/upload          multipart file -> asset_id (+ poster frame for video)
    POST /api/detect          {asset_id}     -> auto-detected mask + boxes
    POST /api/process         {asset_id, mask?, mode?} -> job_id
    GET  /api/jobs/{id}       job state / progress / result url
    GET  /api/backends        which inpainting backends are configured
    GET  /media/{...}         uploaded + produced files

Run:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import shutil
import traceback
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from server.core import config as cfgmod
from server.core import jobs, media, pipeline
from server.core.inpaint import describe as describe_backends
from server.core.maskops import from_png_bytes, overlay_preview, to_png_bytes

cfg = cfgmod.cfg
app = FastAPI(title="Watermark Remover", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".avi", ".gif"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def asset_dir(asset_id: str) -> Path:
    d = cfg.upload_dir / asset_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def b64_to_mask(data_url: str) -> np.ndarray | None:
    """`data:image/png;base64,....` -> uint8 mask (or None)."""
    import base64

    if not data_url:
        return None
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    return from_png_bytes(raw)


# --------------------------------------------------------------------------- #
#  API
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    return {"ok": True, "backends": describe_backends()}


@app.get("/api/backends")
def backends():
    return describe_backends()


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename or "upload.bin").suffix.lower()
    kind = "video" if ext in VIDEO_EXT else ("image" if ext in IMAGE_EXT else "unknown")
    if kind == "unknown":
        raise HTTPException(415, f"unsupported file type {ext}")

    asset_id = uuid.uuid4().hex[:12]
    d = asset_dir(asset_id)
    src = d / f"source{ext}"
    with src.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    info: dict = {"asset_id": asset_id, "kind": kind, "name": file.filename,
                  "source": f"/media/uploads/{asset_id}/source{ext}"}

    if kind == "video":
        vi = media.probe(str(src))
        info.update(width=vi.width, height=vi.height, fps=round(vi.fps, 3),
                    duration=round(vi.duration, 2), has_audio=vi.has_audio,
                    codec=vi.vcodec)
        poster = media.first_frame(str(src), max_side=min(1280, cfg.max_side))
        if poster is not None:
            cv2.imwrite(str(d / "poster.jpg"), poster, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            info["poster"] = f"/media/uploads/{asset_id}/poster.jpg"
    else:
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "cannot decode image")
        info.update(width=img.shape[1], height=img.shape[0])
        info["poster"] = info["source"]
    return info


@app.post("/api/detect")
def detect(asset_id: str = Form(...)):
    d = asset_dir(asset_id)
    src = next((p for p in d.iterdir() if p.name.startswith("source")), None)
    if src is None:
        raise HTTPException(404, "asset not found")
    ext = src.suffix.lower()
    try:
        if ext in VIDEO_EXT:
            vi = media.probe(str(src))
            h = vi.height if vi.width <= cfg.max_side else int(vi.height * cfg.max_side / vi.width)
            w = vi.width if vi.width <= cfg.max_side else cfg.max_side
            det = pipeline.detect_video(str(src), target_shape=(h - h % 2, w - w % 2))
            frame = media.first_frame(str(src), max_side=min(1280, cfg.max_side))
        else:
            frame = cv2.imread(str(src), cv2.IMREAD_COLOR)
            from server.core.detect_image import detect_watermark_image

            det = detect_watermark_image(frame)
    except Exception as exc:                                   # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(500, f"detection failed: {exc}") from exc

    out = det.to_json()
    if det.found and frame is not None:
        cv2.imwrite(str(d / "mask.png"), det.mask)
        cv2.imwrite(str(d / "overlay.jpg"), overlay_preview(frame, det.mask),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        out["mask_url"] = f"/media/uploads/{asset_id}/mask.png"
        out["overlay_url"] = f"/media/uploads/{asset_id}/overlay.jpg"
    return out


@app.post("/api/process")
def process(asset_id: str = Form(...), mask: str = Form(""), mode: str = Form("auto"),
            backend: str = Form("auto")):
    d = asset_dir(asset_id)
    src = next((p for p in d.iterdir() if p.name.startswith("source")), None)
    if src is None:
        raise HTTPException(404, "asset not found")
    ext = src.suffix.lower()
    user_mask = b64_to_mask(mask)

    job = jobs.create()
    out_dir = cfg.result_dir / asset_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if ext in VIDEO_EXT:
        dst = out_dir / f"clean_{job.id}.mp4"

        def work(progress):
            det = pipeline.process_video(str(src), str(dst), mask=user_mask,
                                         mode=mode, backend_name=backend,
                                         progress=progress)
            jobs.update(job.id, detection=det.to_json())

        result_url = f"/media/results/{asset_id}/clean_{job.id}.mp4"
        result_name = f"{src.stem}_clean.mp4"
    else:
        dst = out_dir / f"clean_{job.id}{'.png' if ext in ('.png', '.webp') else '.jpg'}"

        def work(progress):
            det = pipeline.process_image(str(src), str(dst), mask=user_mask,
                                         backend_name=backend, progress=progress)
            jobs.update(job.id, detection=det.to_json())

        result_url = f"/media/results/{asset_id}/{dst.name}"
        result_name = f"{src.stem}_clean{dst.suffix}"

    jobs.update(job.id, result_url=result_url, result_name=result_name)
    jobs.run(job.id, work)
    return {"job_id": job.id}


@app.get("/api/jobs/{job_id}")
def job_state(job_id: str):
    st = jobs.get(job_id)
    if st is None:
        raise HTTPException(404, "no such job")
    return st.__dict__


@app.exception_handler(Exception)
async def _unhandled(request, exc):                            # noqa: ANN001
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# --------------------------------------------------------------------------- #
#  static
# --------------------------------------------------------------------------- #
app.mount("/media/uploads", StaticFiles(directory=str(cfg.upload_dir)), name="uploads")
app.mount("/media/results", StaticFiles(directory=str(cfg.result_dir)), name="results")
app.mount("/", StaticFiles(directory=str(cfgmod.WEB_DIR), html=True), name="web")
