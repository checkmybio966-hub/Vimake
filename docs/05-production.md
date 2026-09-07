# Demo se production tak

Abhi ka code ek single box par chalta hai (in-memory jobs, local disk). Real
product ke liye ye badlaav karne honge — sab likhe hue hain, bas wire karna hai.

## 1. Queue + workers (sabse zaroori)

`core/jobs.py` abhi thread + dict hai. Isse badlo:

```python
# worker.py  (Celery + Redis)
from celery import Celery
from server.core import pipeline

app = Celery("wm", broker=os.environ["REDIS_URL"])

@app.task(bind=True)
def remove(self, src, dst, mask_key=None, mode="auto"):
    def cb(p, stage):
        self.update_state(state="PROGRESS", meta={"progress": p, "stage": stage})
    det = pipeline.process_video(src, dst, mode=mode, progress=cb)
    return det.to_json()
```

* API sirf task enqueue kare, turnt `job_id` de.
* GPU workers alag autoscaling pool me; CPU-only cheap tasks (detect) alag queue.
* Long videos chunk karo (har 30 s ka alag task), phir concat — retries aasan.

## 2. Storage

`cfg.upload_dir` / `cfg.result_dir` ki jagah S3/R2:

```python
# core/storage.py
def put(path, key): s3.upload_file(path, BUCKET, key, ExtraArgs={"ContentType": ...})
def presign(key, ttl=3600): return s3.generate_presigned_url(...)
```

Lifetime policy: uploads 24 h, results 7 din (privacy + cost dono).

## 3. Security / privacy

* **Zero-retention** promise karo (Vmake yahi marketing karta hai): processing ke
  turant baad source delete.
* Upload size/type validate (`WM_MAX_UPLOAD_MB`), magic bytes check karo —
  sirf extension mat dekho.
* ffmpeg ko sandboxed chalao (`-nostdin`, timeout, isolated container). User
  file ko kabhi bhi shell argument me mat do (path pass karo, command string
  nahi).
* Rate limit per user/IP, auth token, signed upload URLs.
* PII ho to processing region pin karo (GDPR).

## 4. Scale ke numbers

Ek 5 s / 1080p clip ke liye (approx, 1× A10G):

| Stage | Time |
|---|---|
| decode 150 frames | ~2 s |
| detect (48 sampled frames) | ~1 s |
| ProPainter inpaint | ~40–60 s |
| encode + mux | ~3 s |

Isliye product me **free preview = pehle 5 s** (Vmake ka wahi model): user ko
result dikhe, GPU cost aapke control me.

Cost control:
* detect pehle chalao; mask chhota ho to sirf zaroori region process karo.
* 1080p se upar mat jao (ProPainter memory ~ resolution²).
* Batch/off-peak queue (spot GPUs ~70% saste).

## 5. Quality badhana (priority order)

1. **LaMa/IOPaint local GPU** — image quality mein sabse bada jump (hamare
   benchmark me CPU diffusion ke muqable +10 dB tak).
2. **ProPainter / E2FGVI video inpainting** — fixed watermark ke liye flicker aur
   texture dono improve.
3. **Detector ko fine-tune karo** — YOLOv8/YOLO11 ko apne watermark dataset par
   (2–3 h GPU). Heuristics ke muqable recall 0.47 → 0.9+ jaata hai.
4. **SAM2 / XMem tracking** — moving mark ko frame-by-frame propagate karna
   flow-composition se zyada stable hai.
5. **Temporal flicker filter** — mask region par ek chhota EMA ya bilateral
   temporal filter (hamare paas EMA hai, `temporal_smooth=0.35`).

## 6. Observability

* Har job me log karo: `asset_id, kind, score, coverage, backend, duration_s`.
* Failure taxonomy: `WATERMARK_NOT_FOUND` / `ENCODE_FAILED` / `FLOW_FAILED` /
  `REMOTE_TIMEOUT` — inke hisaab se UI message aur retry strategy alag.
* Quality proxy metric: processed vs original me sirf mask region ka difference
  (bahut zyada badla to galat hua), aur frame-to-frame difference (flicker).

## 7. Legal / compliance

* Sirf owned/authorised content. ToS me likho.
* Invisible watermarks (SynthID, C2PA) hatane ka support mat do.
* Kuch regions me watermark removal regulated hai — counsel se check karwao.

## 8. Mobile / on-device (agar app native hai)

* Detection to on-device chal sakti hai (chhota YOLO, CoreML/TFLite), par
  inpainting ke liye server hi sahi hai.
* Flow: client → presigned upload → job → CDN URL → player. Exactly wahi API jo
  `docs/03-api.md` me hai.
