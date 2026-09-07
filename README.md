# AI Watermark Remover — images + video (Vmake-style)

Ek complete, runnable implementation: **image ya video upload karo → AI khud
watermark dhoondhta hai → brush se theek karo → remove**. Vmake.ai jaisa
experience, par poori tarah open code — aap apne app me laga sakte ho.

Backend Python (FastAPI + OpenCV + ffmpeg), frontend vanilla JS (koi build step
nahi). GPU/model available ho to LaMa / IOPaint / ProPainter / Replicate
auto-plug ho jaate hain, warna CPU fallback chalta hai.

---

## 1. 30 second me chalao

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r server/requirements.txt

# test media banao (watermarked + clean + ground-truth mask)
python tools/make_sample.py

# server (UI + API)
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Browser me `http://localhost:8000` kholo → file drop karo → **Remove watermark**.

Quality check karna hai to:

```bash
python tests/test_pipeline.py
```

## 2. Vmake.ai actually kya karta hai (reverse-engineered)

`https://vmake.ai/video-watermark-remover/editor?effectModel=video_remove_watermark`
par jao to ye hota hai:

| Layer | Kya milta hai | Hamara equivalent |
|---|---|---|
| Frontend | Next.js app, canvas mask editor (Auto tab + Manual tab: brush, "protect zone"), batch upload (30 files), free 5-second preview, credits ke baad full download | `web/` — vanilla JS canvas editor, batch, auto+manual |
| Upload | file ko encrypted cloud storage me daala, CDN pe serve | `POST /api/upload` → local disk (production me S3/R2) |
| Detect | server-side: video ke frames scan karke watermark region nikalta hai | `server/core/detect_video.py`, `detect_image.py` |
| Remove | GPU par **video inpainting** (flow-guided propagation + generative fill), temporal consistency ke saath; moving watermark ke liye tracking | `server/core/pipeline.py` (local CPU) ya LaMa/ProPainter/Replicate backend |
| Encode | ffmpeg se dubara encode + original audio mux | `server/core/media.py` |
| Billing | credits / subscription, queue | `server/core/jobs.py` (in-memory; production me Celery) |

**Unka "code" public nahi hai** — wo ek hosted product hai (Vmake Labs / Vmodel
family; inke model hosting ki API `api.vmodel.ai` par milti hai, aur
`unwatermark.ai` par watermark-removal API bechi jaati hai). Lekin unka
*approach* industry-standard hai aur neeche poori tarah likha hai:
detect → track → inpaint → re-encode.

Zyada detail: [`docs/01-vmake-analysis.md`](docs/01-vmake-analysis.md)

## 3. Pipeline (ek nazar me)

```
                 ┌────────────── IMAGE ──────────────┐
                 │ ink-mask → blobs → shape-cluster  │  (repeated @handle / tiled logo)
   upload ───────┤ ya stroke+prior scoring           │──► LaMa / IOPaint / OpenCV inpaint ──► jpg/png
                 └───────────────────────────────────┘

                 ┌────────────── VIDEO ──────────────┐
                 │ sample 40-48 frames               │
                 │ variance-attenuation A            │
                 │ edge-persistence   P   ─► FIXED?  │──► flow-guided propagation (har frame
   upload ───────┤ adaptive z-outlier z   ─► MOVING? │    ke liye doosre frames se asli
                 │                                   │    pixels uthao) + diffusion fill
                 └───────────────────────────────────┘
                        MOVING ──► flow-warped temporal clean plate
                                                        │
                                                        ▼
                                        ffmpeg: h264 + original audio
```

Do cheezein poori tarah alag algorithms hain, isliye dono cases acche bante hain:

* **Fixed watermark** (corner logo, timestamp, TikTok/CapCut export mark):
  time se kuch nahi milta (mark har frame me hai) → doosre frames se pixels
  udhaar lo aur synthesize karo.
* **Moving watermark** (Sora / Veo / Kling / animated badge): har pixel par mark
  sirf kuch frames me hota hai → **us pixel ka temporal median = asli background**.
  Ye near-perfect kaam karta hai.

## 4. Measurements (apne synthetic samples par, `samples/`)

`python tests/test_pipeline.py` se reproduce hota hai. Video me "before" ek
**re-encoded unprocessed** copy hai, taaki codec ki generational loss ko
algorithm ke naam na likha jaaye.

| Asset | Detection | Quality |
|---|---|---|
| Tiled "DEMO" photo | IoU **0.64**, 9/9 tiles | PSNR 29.8 → **48.0 dB** |
| Fixed logo + timestamp (video) | kind=static ✓ IoU 0.34 | PSNR 31.4 → **33.1 dB** (+1.75) |
| Moving "Sora" mark (video) | kind=moving ✓ | PSNR 36.9 → **39.1 dB** (+2.13) |

Local CPU backend (OpenCV diffusion) sabse kamzor kadi hai. LaMa/ProPainter/API
lagao to image me 48 dB se bhi upar aur video me +5..10 dB aam baat hai —
[`docs/04-backends.md`](docs/04-backends.md).

## 5. Repo map

```
server/
  app.py                 FastAPI: upload / detect / process / jobs
  core/
    config.py            saare knobs (env se override)
    media.py             ffmpeg wrapper (probe, decode→numpy, encode+mux audio)
    types.py             Detection / JobState
    maskops.py           mask utils (dilate, feather, cleanup, overlay)
    detect_image.py      IMAGE auto-detect  (repeat-shape + stroke/prior)
    detect_video.py      VIDEO auto-detect  (A + P + adaptive z)
    flow.py              optical flow + robust global motion (LK+RANSAC)
    pipeline.py          image + video removal (propagation engine)
    jobs.py              in-memory job queue
    inpaint/
      base.py            backend interface
      opencv_backend.py  CPU diffusion (fallback)
      lama_backend.py    LaMa ONNX + IOPaint HTTP client
      api_backend.py     Replicate / fal / custom HTTP
web/                     index.html, app.js, styles.css
tools/make_sample.py     synthetic watermarked+clean media banata hai
tests/test_pipeline.py   end-to-end IoU + PSNR report
docs/                    analysis, algorithms, backends, production roadmap
```

## 6. Apne existing app me kaise lagayein

Teen tareeke (sab `server/core/inpaint/` me pehle se hain):

1. **Sirf API** — `POST /api/upload` → `/api/detect` → `/api/process` → poll
   `/api/jobs/{id}`. Poora contract [`docs/03-api.md`](docs/03-api.md) me.
2. **Library** —
   ```python
   from server.core import pipeline
   det = pipeline.detect_video("in.mp4")            # ya detect_watermark_image(frame)
   pipeline.process_video("in.mp4", "out.mp4", mask=det.mask, mode=det.kind)
   ```
3. **Sirf detection, removal aapke model se** — `detect_*` se mask lo, phir apne
   LaMa/SDXL/ProPainter se fill karo.

Production (GPU, queue, S3, auth, rate-limit) ke liye:
[`docs/05-production.md`](docs/05-production.md).

## 7. Imaandaar limitations

* Local CPU backend textured background par smudge chhod deta hai — LaMa /
  ProPainter / hosted API lagao to farq padta hai.
* Static watermark + bilkul static background (tripod, koi motion nahi) = sabse
  mushkil case: time axis se signal hi nahi milta. Brush se region bata do.
* Ye **visible** watermark hatata hai. Invisible provenance watermarks
  (SynthID, C2PA metadata) alag cheez hain aur intentionally support nahi kiye
  gaye. Sirf apne ya authorised content par use karein.
* Long videos: `WM_MAX_FRAMES` (default 600) ke baad temporal subsample hota
  hai, taaki RAM control me rahe.

---

Aage poori detail: [`docs/01-vmake-analysis.md`](docs/01-vmake-analysis.md) se shuru karein.
