# AI Watermark & Text Remover — images + video

**GitHub:** https://github.com/checkmybio966-hub/Vimake/tree/arena/01a07c00-vimake
**Deploy guide:** [`DEPLOY.md`](DEPLOY.md) (Vercel + Docker worker)

Image ya video upload karo → **AI khud watermark + text dhoondhta hai** → brush
se theek karo → remove. Vmake.ai jaisa experience, poori tarah open code.

- **Images:** watermark, tiled `@handle`, caption, timestamp, subtitle, sticker
  text — sab hataya jaata hai ("koi bhi text nahi chahiye" mode default ON).
- **Videos:** fixed logo ho ya Sora/Veo/Kling wala moving mark — dono.
- **Audio original rehta hai**, resolution preserve hoti hai.
- Backends pluggable: OpenCV (CPU) → LaMa → IOPaint → Replicate → fal → apna.

---

## 1. 30 second me chalao (local)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r server/requirements.txt

python tools/make_sample.py            # test media (watermarked + clean + GT mask)
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Browser: `http://localhost:8000` → file drop → **Remove watermark**.
Quality report: `python tests/test_pipeline.py`

Docker: `docker compose up --build` (poora stack, ffmpeg ke saath).

## 2. Vercel par kaise chadhaayein (2 min + ek worker)

```
Browser ──► Vercel  (static UI + FastAPI serverless, images yahin bante hain)
              └── videos ──► Worker (Docker/Railway/Render/Fly) ffmpeg + OpenCV
```

1. https://vercel.com/new → import `checkmybio966-hub/Vimake` (branch
   `arena/01a07c00-vimake`) → Deploy. `vercel.json` sab set karta hai.
2. Worker (video ke liye): `railway up` / Render / Fly / `docker compose up`
   — **wahi repo, bas full mode.**
3. Vercel env: `WM_LIGHT=1`, `WM_WORKER_URL=https://<worker>`, `WORKER_TOKEN=…`
   (poori table + troubleshooting [`DEPLOY.md`](DEPLOY.md) me).

Vercel par video isliye forward karte hain: serverless timeout (60 s),
4.5 MB body limit, 250 MB bundle limit. Images bilkul theek chalti hain (~1-5 s).

## 3. Vmake.ai andar kya karta hai (aur yahan kya hai)

| Layer | Vmake (closed SaaS) | Is repo me |
|---|---|---|
| Frontend | Next.js + canvas mask editor (Auto + brush/protect zone), batch | `web/` — vanilla JS, batch, auto+manual |
| Detect | server-side multi-frame analysis, moving mark ke liye tracking | `core/detect_video.py` (A + P + adaptive z), `core/detect_image.py` (repeat-shape), `core/detect_text.py` (MSER text) |
| Remove | GPU video inpainting (ProPainter family) | `core/pipeline.py::propagate_video` + LaMa/ProPainter/Replicate backend |
| Encode | ffmpeg, audio preserve | `core/media.py::VideoWriter` |
| API | unka `api.vmodel.ai` / `unwatermark.ai` (~$0.02/s) | `POST /api/...` (contract [`docs/03-api.md`](docs/03-api.md)) |

Unka source closed hai; unka **model-hosting API** (`vmodel`) aur
`unwatermark.ai` public hai — chahein to direct call kar sakte ho
([`docs/04-backends.md`](docs/04-backends.md)).

## 4. Pipeline (ek nazar me)

```
IMAGE
  ink-mask + MSER ─┬─► repeated-shape clustering (tiled @handle / grid)
                   ├─► text detector  (caption / timestamp / subtitle)   ◄── default ON
                   └─► stroke+prior scoring (ek logo)
                              │
                              ▼
                   LaMa / IOPaint / OpenCV inpaint ──► png|jpg

VIDEO
  sample 40-48 frames
    variance-attenuation A ─┐
    edge-persistence   P   ─┼─► FIXED?  ──► flow-guided propagation
    adaptive z-outlier z  ─┘   MOVING?  ──► flow/raw temporal clean plate
                              │
                              ▼
              ffmpeg: h264 + ORIGINAL audio (aac remux)
```

Do alag algorithms isliye: **fixed** watermark har frame me hota hai (time se
kuch nahi milta → doosre frames se pixels udhaar lo), **moving** mark har pixel
par sirf kuch frames me hota hai (us pixel ka temporal median = asli background).

## 5. Measurements (`python tests/test_pipeline.py`, apne samples par)

Video ka "before" ek **re-encoded unprocessed** copy hai — taaki codec ki
generational loss algorithm ke naam na likhi jaaye.

| Asset | Detection | Quality (CPU backend) |
|---|---|---|
| Tiled watermark photo | IoU 0.63, 9/9 tiles | PSNR 29.8 → **48.0 dB** |
| Text/caption photo | 10 text regions | PSNR 24.8 → **30.2 dB** (+5.4) |
| Fixed logo + timestamp (video) | kind=static ✓ | +1.9 dB |
| Moving "Sora" mark (video) | kind=moving ✓ | +2.1 dB |

Local CPU backend (OpenCV diffusion) sabse kamzor kadi hai — LaMa/ProPainter/
hosted API lagao to image me 50+ dB aur video me +5..10 dB aam hai.

## 6. Repo map

```
api/index.py              Vercel serverless entry (ASGI + path fix)
vercel.json               builds + catch-all rewrite + maxDuration
Dockerfile / docker-compose.yml / requirements-vercel.txt
DEPLOY.md                 Vercel + worker step-by-step

server/app.py             FastAPI: upload / detect / process / jobs /
                          storage-presign / worker-role endpoints
server/core/
  config.py               saare knobs (env)
  media.py                ffmpeg wrapper (probe, decode→numpy, encode+audio mux)
  detect_image.py         image watermark auto-detect (+ remove_text)
  detect_text.py          MSER/SWT text detector (+ EasyOCR/PaddleOCR slot)
  detect_video.py         A + P + adaptive z → static / moving classification
  flow.py                 optical flow + robust global motion (LK+RANSAC)
  pipeline.py             propagation engine (2 modes) + image path
  storage.py              local / S3-compatible (presigned direct upload)
  worker.py               heavy jobs ko worker par bhejna
  jobs.py                 memory / Redis (Upstash) job store
  maskops.py, types.py
  inpaint/                base, opencv, lama(ONNX + IOPaint), api(Replicate/fal/custom)
web/                      index.html, app.js, styles.css
tools/make_sample.py      synthetic watermarked + text + clean + GT media
tests/test_pipeline.py    end-to-end IoU + PSNR report
docs/                     01 vmake analysis · 02 algorithms · 03 api · 04 backends · 05 production
```

## 7. Apne app me lagana

```bash
# 1) upload
curl -F "file=@photo.jpg"  $HOST/api/upload            # → asset_id
# 2) detect (text bhi: remove_text=1 default)
curl -F "asset_id=$AID" -F "remove_text=1" $HOST/api/detect
# 3) process
curl -F "asset_id=$AID" -F "mode=auto" $HOST/api/process   # → job_id
# 4) poll
curl $HOST/api/jobs/$JOB
```

Ya library ki tarah:

```python
from server.core import pipeline
det = pipeline.process_image("in.jpg", "out.png", remove_text=True)     # image
det = pipeline.process_video("in.mp4", "out.mp4", remove_text=True)     # video
```

Poora contract: [`docs/03-api.md`](docs/03-api.md).

## 8. Imaandaar limitations

* CPU backend textured background par smudge chhod deta hai → LaMa/ProPainter
  ya hosted API use karo (`WM_BACKEND=replicate`).
* "Remove text" har detected text hata deta hai — **scene me likha hua text
  bhi** (sign board, t-shirt). Aap UI me toggle off kar sakte ho.
* Static watermark + bilkul static background (tripod) = sabse mushkil; brush
  se region bata do.
* Ye **visible** overlays hatata hai. Invisible provenance (SynthID, C2PA)
  support nahi karta — sirf apne/authorised content par use karein.
* Lambi video: `WM_MAX_FRAMES` (600) ke baad temporal subsample.

---

Shuru karo: [`docs/01-vmake-analysis.md`](docs/01-vmake-analysis.md) →
[`docs/02-algorithms.md`](docs/02-algorithms.md) → [`DEPLOY.md`](DEPLOY.md).
