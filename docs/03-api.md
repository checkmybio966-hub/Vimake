# HTTP API contract

Saare endpoints relative hain (`fetch('/api/...')`) — isliye frontend kisi bhi
host/proxy ke peeche bina CORS ke chal jaata hai.

```
GET  /api/health          → {"ok": true, "backends": {...}}
GET  /api/backends        → kaunse inpainting backends configured hain
POST /api/upload          multipart file → asset_id + metadata (+ poster for video)
POST /api/detect          {asset_id} → mask + boxes + score + kind
POST /api/process         {asset_id, mask?, mode?, backend?} → job_id
GET  /api/jobs/{job_id}   → status / progress / result_url
GET  /media/uploads/...   → source, poster, mask.png, overlay.jpg
GET  /media/results/...   → processed files
```

## POST /api/upload

```bash
curl -F "file=@clip.mp4" http://localhost:8000/api/upload
```

```jsonc
// image
{"asset_id":"1fb2e6222b44","kind":"image","name":"photo.png",
 "source":"/media/uploads/1fb2e6222b44/source.png",
 "width":640,"height":360,"poster":"/media/uploads/1fb2e6222b44/source.png"}

// video
{"asset_id":"2033dcb1ce76","kind":"video","name":"clip.mp4","source":"...","width":640,
 "height":360,"fps":25.0,"duration":4.0,"has_audio":true,"codec":"h264",
 "poster":"/media/uploads/2033dcb1ce76/poster.jpg"}
```

## POST /api/detect

```bash
curl -F "asset_id=2033dcb1ce76" http://localhost:8000/api/detect
```

```jsonc
{"found":true,"score":1.0,"method":"temporal-outlier (clean plate)",
 "kind":"moving",                       // "static" | "moving"
 "boxes":[[220,163,78,85], ...],        // [x, y, w, h]
 "notes":"outlier z inside=10.80 vs outside=0.41",
 "coverage":0.0454,
 "mask_url":"/media/uploads/.../mask.png",
 "overlay_url":"/media/uploads/.../overlay.jpg"}
```

`kind` batata hai ki kaunsa removal algorithm chalega. `score` 0..1 — UI me
"found 90%" dikhane ke liye.

## POST /api/process

```bash
curl -F "asset_id=2033dcb1ce76" \
     -F "mode=auto" \                    # auto | static | moving
     -F "backend=auto" \                 # auto | opencv | lama | iopaint | replicate | fal | custom
     -F "mask=data:image/png;base64,iVBORw0..."   # optional: user ka brush
     http://localhost:8000/api/process
# → {"job_id":"5e7a1b6f8864"}
```

`mask` dene ka matlab: auto-detection **skip** ho jaata hai (manual mode).
Mask PNG me **safed (255) = hatao**, kaala = rakho.

## GET /api/jobs/{job_id}

```jsonc
{"id":"5e7a1b6f8864","status":"running",   // queued|running|done|error
 "progress":0.748,"stage":"clean plate frame 72/100","message":"",
 "result_url":"/media/results/2033dcb1ce76/clean_5e7a1b6f8864.mp4",
 "result_name":"source_clean.mp4",
 "detection":{...}}                        // done hone par
```

Polling interval ~700 ms rakhna theek hai (`web/app.js` me wahi hai).

## Library ke taur par (bina HTTP ke)

```python
from server.core import pipeline
from server.core.detect_image import detect_watermark_image
import cv2

# --- image ---
frame = cv2.imread("photo.jpg")
det   = detect_watermark_image(frame)          # .mask, .score, .method, .boxes
pipeline.process_image("photo.jpg", "clean.png", mask=det.mask)

# --- video ---
det = pipeline.detect_video("clip.mp4")       # .kind = "static" | "moving"
pipeline.process_video("clip.mp4", "clean.mp4", mask=det.mask, mode=det.kind)

# --- mask aapke paas hai, bas removal chahiye ---
pipeline.process_video("clip.mp4", "clean.mp4", mask=my_mask, mode="static")
```

Progress chahiye to callback do:

```python
pipeline.process_video("in.mp4", "out.mp4",
                       progress=lambda p, stage: print(f"{p:.0%} {stage}"))
```

## Errors

| HTTP | Kab | Kya karein |
|---|---|---|
| 415 | unsupported extension | sirf image/video bhejo |
| 404 | galat asset_id / job_id | dubara upload karo |
| 500 | `WATERMARK_NOT_FOUND` | UI brush mode khol de (detection ne kuch nahi paaya) |
| 500 | encode/decode error | ffmpeg log dekho (`WM_...` env se debug) |

## Config (env)

| Var | Default | Meaning |
|---|---|---|
| `WM_BACKEND` | `auto` | `opencv|lama|iopaint|replicate|fal|custom` |
| `WM_API_PROVIDER` | `replicate` | remote provider |
| `WM_API_KEY` | – | provider token (`REPLICATE_API_TOKEN` / `FAL_KEY` bhi chalega) |
| `WM_API_MODEL` | – | provider model id |
| `WM_MAX_SIDE` | 960 | processing resolution cap |
| `WM_MAX_FRAMES` | 600 | uske aage temporal subsample |
| `WM_DETECT_SAMPLES` | 48 | detection ke liye kitne frames sample |
| `WM_MASK_DILATE` / `WM_MASK_FEATHER` | 6 / 3 | mask margin / soft edge (px) |
| `WM_CP_Z` | 5.0 | moving-mark outlier threshold (badhao = conservative) |
| `WM_MOVING_WINDOW` / `WM_MOVING_STRIDE` | 24 / 2 | clean-plate temporal window |
| `WM_INPAINT_WINDOW` | 8 | fixed-watermark propagation window |
| `WM_WORKDIR` | `./runtime` | uploads/results kahan |
| `FFMPEG_BIN` | imageio-ffmpeg | apna ffmpeg path |
