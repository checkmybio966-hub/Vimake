# Vmake.ai ka watermark remover — andar kya chal raha hai

URL: `https://vmake.ai/video-watermark-remover/editor?effectModel=video_remove_watermark`

## 0. Pehle: "konsa code use hota hai" — seedhi baat

Vmake ka **frontend/backend source code public nahi hai**. Wo ek hosted SaaS hai.
Jo cheezein hum *dekh* sakte hain (public page se):

* Page ek Next.js-style app hai; static assets `static-assets.vmake.ai` (Volcano/BytePlus
  CDN, `imageView2` image transforms — ByteDance stack) se aate hain.
* User material / example assets `kapkap-common.stariidata.com` par hain — "Kapkap"
  Vmake ke content platform ka internal naam hai, `stariidata` unka data/asset
  bucket hai.
* Ek model-serving platform bhi isi family ka hai: `vmodel.ai`, jahan
  `vmodel/video-watermark-remover` model public hai aur REST API deta hai:

```bash
curl -X POST https://api.vmodel.ai/api/tasks/v1/create \
  -H "Authorization: Bearer $VModel_API_TOKEN" \
  -d '{"version":"0ad4ca94...","input":{"video":"...mp4","mask_stage":"...jpg"}}'
# → {"task_id": "...", "status":"succeeded", "output":["...output.mp4"]}
```

Yaani pattern bilkul wahi hai jo humne implement kiya hai: **task banao → poll
karo → output URL**. Isi model ki marketing `unwatermark.ai` ke naam se hoti hai
("Video Watermark Remover API", ~$0.02/second of video, max 1080p / 3 min).

**Bottom line:** unka "code" copy nahi ho sakta, par unka *algorithm* koi raaz
nahi hai — ye industry-standard pipeline hai aur is repo me likha hai.

## 1. Unka product flow (UI se jo dikhta hai)

```
1. Upload (drag / paste / click, batch me 30 tak, ya URL se import)
2. Har file ka apna card; "Auto" tab me AI khud detect karta hai
3. "Manual" tab me brush + "protect zone" (jo area AI ko nahi chhoona)
4. Preview: free me sirf pehle 5 second; full download ke liye credits
5. Server GPU par process → ffmpeg re-encode → download
```

Product-level decisions jo humne bhi copy kiye:

* **Auto + Manual dono.** 100% accurate detector koi nahi hai, isliye brush
  hamesha available hona chahiye. Hamare UI me bhi auto mask ke upar brush
  chalta hai (`web/app.js` me layers: `auto ∪ brush − eraser`).
* **Batch.** Har asset ka alag job, alag progress.
* **Free preview, paid full.** Server cost real hai (GPU second).

## 2. Unka technical pipeline (reconstructed)

```
[upload] → object storage (encrypted)
        → job queue
        → [GPU worker]
             ├─ decode frames (ffmpeg / PyAV)
             ├─ DETECT watermark region
             │     • static logo  : multi-frame analysis (low temporal variance +
             │                      persistent edges + corner / border prior)
             │     • moving mark  : per-frame residual vs temporal median
             │     • text/logo    : (optional) text detector / OCR-ish model
             ├─ TRACK (moving marks ke liye): mask ko frame-to-frame propagate
             │     (optical flow / XMem / SAM2 jaisa video object segmentation)
             ├─ INPAINT: video inpainting model — ProPainter / E2FGVI family
             │     (flow completion + spatiotemporal transformer) ya diffusion fill
             └─ ENCODE: ffmpeg, original audio + bitrate preserve
        → result CDN par → user download
```

Iske har block ka open-source equivalent is repo me hai:

| Vmake (closed) | Is repo me | File |
|---|---|---|
| frame decode | ffmpeg → numpy stream | `core/media.py` |
| static detection | variance-attenuation + edge-persistence | `core/detect_video.py` |
| moving detection | adaptive robust z-score vs temporal median | `core/detect_video.py` |
| image detection | ink-mask + repeated-shape clustering | `core/detect_image.py` |
| mask tracking | dense optical-flow propagation (`compose_step`) | `core/flow.py` |
| video inpainting | flow-guided neighbour blending + diffusion | `core/pipeline.py::propagate_video` |
| (GPU option) | LaMa / IOPaint / ProPainter / Replicate | `core/inpaint/*` |
| re-encode + audio | `VideoWriter` (libx264 + aac mux) | `core/media.py` |
| queue/jobs | in-memory → Celery/Redis | `core/jobs.py` |

## 3. Unke model ka input/output shape (public API se)

```jsonc
// POST /api/tasks/v1/create
{ "version": "0ad4ca94...",
  "input": { "video": "<url>", "mask_stage": "<mask.jpg>" } }

// GET task until status == "succeeded"
{ "task_id": "d9zz...", "status": "succeeded",
  "output": ["https://.../output.mp4"], "predict_time": 8.5 }
```

`mask_stage` ka matlab: wo bhi **mask-driven** kaam karte hain — pehle mask
banaya (detect ya user brush), phir us mask ko poori video me inpainting se
bhara. Bilkul wahi architecture jo humne likhi hai. Agar aap chahein to apne app
me seedha ye API call kar sakte ho (`WM_API_PROVIDER=custom` +
`WM_API_BASE=...`), ya Replicate ke `jd7h/xmem-propainter-inpainting` (pehle
frame ka mask lekar poori video me propagate + inpaint karta hai) use kar sakte
ho — [`docs/04-backends.md`](04-backends.md).

## 4. Accuracy ke claims kitne sacche hain

Marketing me "98.7% accuracy" likha hai — ye koi standard metric nahi hai,
benchmark disclose nahi hai. Practical reality (reviews + hamare experiments):

* **Static corner logo on normal footage** — kaafi achha, aksar bilkul saaf.
* **Semi-transparent moving AI mark** — results unpredictable; ghosting/flicker
  aam hai jab mark kisi detail (face, text) ke upar se guzarta hai.
* **Complex / textured background** — inpainting ko background guess karna
  padta hai; kabhi-kabhi galat texture ban jaata hai.

Isliye hamara design detection ko *confidence* ke saath return karta hai aur
brush ko first-class banata hai — "AI ne galat jagah chhua to user theek kar le"
hi asli product answer hai.

## 5. Legal / ethical note

Visible watermark hatana tab theek hai jab aap content ke malik ho ya
permission ho (apna exported CapCut video, apni Sora generation, client ka
footage). Doosron ke copyrighted material se watermark hatana galat hai.
**Invisible** provenance markers (SynthID, C2PA) intentionally is repo ka hissa
nahi hain.
