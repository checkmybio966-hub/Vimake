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
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from server.core import config as cfgmod
from server.core import jobs, media, pipeline
from server.core.inpaint import describe as describe_backends
from server.core import storage as storage_mod
from server.core import worker as worker_mod
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
def detect(asset_id: str = Form(...), remove_text: str = Form("1")):
    d = asset_dir(asset_id)
    src = next((p for p in d.iterdir() if p.name.startswith("source")), None)
    if src is None:
        raise HTTPException(404, "asset not found")
    ext = src.suffix.lower()
    want_text = remove_text not in ("0", "false", "no", "")
    try:
        if ext in VIDEO_EXT:
            vi = media.probe(str(src))
            h = vi.height if vi.width <= cfg.max_side else int(vi.height * cfg.max_side / vi.width)
            w = vi.width if vi.width <= cfg.max_side else cfg.max_side
            frame = media.first_frame(str(src), max_side=min(1280, cfg.max_side))
            det = pipeline.detect_video(str(src), target_shape=(h - h % 2, w - w % 2))
            if want_text and frame is not None:
                from server.core.detect_text import detect_text_regions, text_mask

                g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                boxes = detect_text_regions(g, aggressive=True)
                tm = text_mask(g, boxes)
                if tm.any():
                    tm = cv2.resize(tm, (det.mask.shape[1], det.mask.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
                    base = det.mask if det.mask is not None and det.mask.any() \
                        else np.zeros_like(tm)
                    det.mask = np.maximum(base, tm)
                    det.found = True
                    det.method = (det.method or "") + "+text"
                    det.notes = (det.notes or "") + f" +{len(boxes)} text region(s)"
        else:
            frame = cv2.imread(str(src), cv2.IMREAD_COLOR)
            from server.core.detect_image import detect_watermark_image

            det = detect_watermark_image(frame, remove_text=want_text)
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


@app.get("/api/storage/presign")
def presign(filename: str = ""):
    """Direct-to-storage upload URL (production). Local mode → {"mode":"local"}."""
    st = storage_mod.get_storage()
    key = f"u/{uuid.uuid4().hex[:12]}/{Path(filename or 'file').name}"
    url = st.presign_put(key)
    if not url:
        return {"mode": "local", "key": None, "url": None}
    return {"mode": st.mode, "key": key, "url": url, "final_url": st.url(key)}


def _save_upload(data: bytes, name: str) -> str:
    """Store an asset through the configured storage backend -> source path/url."""
    st = storage_mod.get_storage()
    key = f"u/{uuid.uuid4().hex[:12]}/{name}"
    if st.mode == "local":
        d = cfg.upload_dir / uuid.uuid4().hex[:12]
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"source{Path(name).suffix.lower() or '.bin'}"
        p.write_bytes(data)
        return str(p)
    st.put(data, key)
    return key


@app.post("/api/remote/process")
async def remote_process(request: Request):
    """WORKER role: same app, running long-lived with ffmpeg + OpenCV.

    Accepts either JSON {"source_url": ...} or multipart (file + fields).
    """
    if cfg.worker_token and request.headers.get("x-worker-token") != cfg.worker_token:
        raise HTTPException(401, "bad worker token")

    ctype = request.headers.get("content-type", "")
    params = {"kind": "image", "mode": "auto", "remove_text": "1", "backend": "auto"}
    if ctype.startswith("application/json"):
        payload = await request.json()
        params.update({k: payload.get(k, v) for k, v in params.items()})
        src_url = payload.get("source_url")
        suffix = Path(src_url or "").suffix or ".bin"
        d = cfg.upload_dir / uuid.uuid4().hex[:12]
        d.mkdir(parents=True, exist_ok=True)
        src = str(d / f"source{suffix}")
        storage_mod.download(src_url, src)
        user_mask = pipeline.b64_to_mask(payload.get("mask", "")) if hasattr(pipeline, "b64_to_mask") else None
    else:
        form = await request.form()
        params.update({k: str(form.get(k, v)) for k, v in params.items()})
        up = form.get("file")
        if up is None:
            raise HTTPException(400, "file or source_url required")
        data = await up.read() if hasattr(up, "read") else b""
        src = _save_upload(data, up.filename or "upload.bin")
        mask_field = form.get("mask")
        user_mask = None
        if mask_field:
            user_mask = b64_to_mask(str(mask_field))

    job = jobs.create()
    out_dir = cfg.result_dir / uuid.uuid4().hex[:12]
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(src).suffix.lower()
    want_text = params.get("remove_text", "1") not in ("0", "false", "no", "")

    if ext in VIDEO_EXT:
        dst = out_dir / f"clean_{job.id}.mp4"
        result_url, result_name = f"/media/results/{Path(out_dir).name}/{dst.name}", "clean.mp4"

        def work(progress):
            det = pipeline.process_video(src, str(dst), mask=user_mask,
                                         mode=params.get("mode", "auto"),
                                         remove_text=want_text,
                                         backend_name=params.get("backend", "auto"),
                                         progress=progress)
            jobs.update(job.id, detection=det.to_json())
    else:
        dst = out_dir / f"clean_{job.id}{'.png' if ext in ('.png', '.webp') else '.jpg'}"
        result_url, result_name = f"/media/results/{Path(out_dir).name}/{dst.name}", f"clean{dst.suffix}"

        def work(progress):
            det = pipeline.process_image(src, str(dst), mask=user_mask,
                                         remove_text=want_text,
                                         backend_name=params.get("backend", "auto"),
                                         progress=progress)
            jobs.update(job.id, detection=det.to_json())

    jobs.update(job.id, result_url=result_url, result_name=result_name)

    # serverless (Vercel) mode: video / heavy work goes to the long-running worker
    if worker_mod.enabled() and (ext in VIDEO_EXT or cfg.light):
        try:
            mask_path = None
            if user_mask is not None:
                mask_path = str(asset_dir(asset_id) / "user_mask.png")
                Path(mask_path).write_bytes(to_png_bytes(user_mask))
            rid = worker_mod.submit_file(str(src), "video" if ext in VIDEO_EXT else "image",
                                         mode=mode, remove_text=want_text, backend=backend,
                                         mask_path=mask_path)
            if rid:
                jobs.update(job.id, status="running", stage="queued on worker",
                            message=rid, progress=0.02)
                return {"job_id": job.id, "remote": rid}
        except Exception as exc:                                  # noqa: BLE001
            traceback.print_exc()
            jobs.update(job.id, status="error", message=f"worker submit failed: {exc}")
            return {"job_id": job.id}

    jobs.run(job.id, work)
    return {"job_id": job.id}


@app.get("/api/remote/jobs/{job_id}")
def remote_job(job_id: str, request: Request):
    if cfg.worker_token and request.headers.get("x-worker-token") != cfg.worker_token:
        raise HTTPException(401, "bad worker token")
    st = jobs.get(job_id)
    if st is None:
        raise HTTPException(404, "no such job")
    return st.__dict__


@app.post("/api/process")
def process(asset_id: str = Form(...), mask: str = Form(""), mode: str = Form("auto"),
            backend: str = Form("auto"), remove_text: str = Form("1")):
    d = asset_dir(asset_id)
    src = next((p for p in d.iterdir() if p.name.startswith("source")), None)
    if src is None:
        raise HTTPException(404, "asset not found")
    ext = src.suffix.lower()
    user_mask = b64_to_mask(mask)
    want_text = remove_text not in ("0", "false", "no", "") and user_mask is None

    job = jobs.create()
    out_dir = cfg.result_dir / asset_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if ext in VIDEO_EXT:
        dst = out_dir / f"clean_{job.id}.mp4"

        def work(progress):
            det = pipeline.process_video(str(src), str(dst), mask=user_mask,
                                         mode=mode, backend_name=backend,
                                         remove_text=want_text, progress=progress)
            jobs.update(job.id, detection=det.to_json())

        result_url = f"/media/results/{asset_id}/clean_{job.id}.mp4"
        result_name = f"{src.stem}_clean.mp4"
    else:
        dst = out_dir / f"clean_{job.id}{'.png' if ext in ('.png', '.webp') else '.jpg'}"

        def work(progress):
            det = pipeline.process_image(str(src), str(dst), mask=user_mask,
                                         backend_name=backend, remove_text=want_text,
                                         progress=progress)
            jobs.update(job.id, detection=det.to_json())

        result_url = f"/media/results/{asset_id}/{dst.name}"
        result_name = f"{src.stem}_clean{dst.suffix}"

    jobs.update(job.id, result_url=result_url, result_name=result_name)

    # serverless (Vercel) mode: video / heavy work goes to the long-running worker
    if worker_mod.enabled() and (ext in VIDEO_EXT or cfg.light):
        try:
            mask_path = None
            if user_mask is not None:
                mask_path = str(asset_dir(asset_id) / "user_mask.png")
                Path(mask_path).write_bytes(to_png_bytes(user_mask))
            rid = worker_mod.submit_file(str(src), "video" if ext in VIDEO_EXT else "image",
                                         mode=mode, remove_text=want_text, backend=backend,
                                         mask_path=mask_path)
            if rid:
                jobs.update(job.id, status="running", stage="queued on worker",
                            message=rid, progress=0.02)
                return {"job_id": job.id, "remote": rid}
        except Exception as exc:                                  # noqa: BLE001
            traceback.print_exc()
            jobs.update(job.id, status="error", message=f"worker submit failed: {exc}")
            return {"job_id": job.id}

    jobs.run(job.id, work)
    return {"job_id": job.id}


@app.get("/api/jobs/{job_id}")
def job_state(job_id: str):
    st = jobs.get(job_id)
    if st is None:
        raise HTTPException(404, "no such job")
    # job was forwarded to a worker -> mirror its state
    if st.message and st.stage == "queued on worker" or (st.message and worker_mod.enabled()
                                                         and st.status == "running"):
        rid = st.message
        try:
            remote = worker_mod.poll(rid)
        except Exception:                                         # noqa: BLE001
            remote = None
        if remote:
            st.status = remote.get("status", st.status)
            st.progress = remote.get("progress", st.progress)
            st.stage = remote.get("stage", st.stage)
            st.detection = remote.get("detection") or st.detection
            rurl = remote.get("result_url")
            if rurl and str(rurl).startswith("/"):
                rurl = cfg.worker_url.rstrip("/") + rurl     # browser fetches it directly
            st.result_url = rurl or st.result_url
            st.result_name = remote.get("result_name") or st.result_name
            if remote.get("status") == "error":
                st.message = remote.get("message", "worker failed")
            jobs.update(job_id, **st.__dict__)
    return st.__dict__


@app.exception_handler(Exception)
async def _unhandled(request, exc):                            # noqa: ANN001
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# --------------------------------------------------------------------------- #
#  static
# --------------------------------------------------------------------------- #
app.mount("/media/uploads", StaticFiles(directory=str(cfg.upload_dir)), name="uploads")
try:
    app.mount("/media/blobs", StaticFiles(directory=str(cfg.workdir / "blobs")), name="blobs")
except Exception:                                                 # noqa: BLE001
    pass
app.mount("/media/results", StaticFiles(directory=str(cfg.result_dir)), name="results")
app.mount("/", StaticFiles(directory=str(cfgmod.WEB_DIR), html=True), name="web")
