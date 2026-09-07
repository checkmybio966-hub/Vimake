# Deploy: Vercel (UI + API) + worker (video)

Repo: **https://github.com/checkmybio966-hub/Vimake/tree/arena/01a07c00-vimake**

```
Browser ──► Vercel  (api/index.py → FastAPI app)
              │  images: yahin process (OpenCV inpainting, 1-5 s, result data-url)
              │  videos: WM_WORKER_URL par forward  (serverless limit 60 s)
              ▼
        Worker service (Docker / Railway / Render / Fly / apna GPU box)
              ffmpeg + OpenCV + (optional) LaMa / ProPainter
```

Vercel par video isliye alag bhejte hain: serverless function ki timeout
(Hobby 60 s / Pro 300 s), 4.5 MB request-body limit aur ffmpeg ka na hona.
**Worker wahi repo hai — bas full mode me.**

---

## A. Worker (Docker) — 5 minute me

Koi bhi ek chuno:

```bash
# Railway
railway up                       # Dockerfile auto-detect
#   env: PORT=8000, WM_WORKDIR=/data, WORKER_TOKEN=<secret>

# Render  → New > Web Service > Docker > repo select
#   env: PORT=8000, WORKER_TOKEN=<secret>

# Fly.io
fly launch --dockerfile Dockerfile
fly secrets set WORKER_TOKEN=<secret>

# ya local / apne VPS par
docker compose up --build         # http://localhost:8000
```

Worker ke env vars:

| Var | Value |
|---|---|
| `PORT` | `8000` |
| `WM_WORKDIR` | `/data` (persistent volume mount karo) |
| `WORKER_TOKEN` | shared secret (Vercel par bhi yahi set karo) |
| `WM_BACKEND` | `auto` (ya `lama` / `replicate` / `fal` quality ke liye) |
| `REPLICATE_API_TOKEN` / `FAL_KEY` | agar hosted GPU use karna hai |
| `LAMA_ONNX_PATH` | agar apna LaMa ONNX model hai |
| `WM_STORAGE` | `s3` + S3_* (agar browser seedha storage par upload kare) |

Test karo: `curl https://worker.up.railway.app/api/health`

## B. Vercel — 2 minute me

1. https://vercel.com/new → **Import Git Repository** →
   `checkmybio966-hub/Vimake` (branch `arena/01a07c00-vimake`).
2. **Kuch bhi configure karne ki jarurat nahi** — Vercel `requirements.txt` me
   fastapi dekh kar FastAPI preset lagata hai aur `api/index.py` ka `app`
   entrypoint pakad leta hai. Root Directory = repo root, Build Command khali.
3. Environment variables (Settings → Environment Variables):

| Var | Value | Kab |
|---|---|---|
| `WM_LIGHT` | `1` | hamesha (video worker ko bhejne ke liye) |
| `WM_WORKER_URL` | `https://worker.up.railway.app` | video chahiye to |
| `WORKER_TOKEN` | worker wala secret | agar worker par set hai |
| `REPLICATE_API_TOKEN` / `FAL_KEY` | token | images ko GPU se theek karna hai to |
| `WM_STORAGE` | `s3` | 4.5 MB se badi files ke liye |
| `S3_BUCKET`, `S3_REGION`, `S3_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `S3_PUBLIC_BASE_URL` | R2/S3 creds | `WM_STORAGE=s3` ke saath |
| `REDIS_URL` | Upstash `redis://...` | optional (video job-state persist) |

4. **Deploy** → `https://<project>.vercel.app` khul jaayega.

> Hobby plan par function timeout 60 s hai (`vercel.json` me `maxDuration: 60`).

### Serverless me kaam kaise chalta hai (important)

Serverless function ka filesystem (`/tmp`) **har instance ka apna** hota hai,
isliye purana `upload → /media/... → process` flow wahan reliable nahi hai.
Ab:

* `GET /api/config` app ko batata hai `{"serverless": true, "worker": ...}`.
* Browser **har request me file dubhara bhejta hai** (`/api/detect`,
  `/api/process` dono `file` accept karte hain; `asset_id` bhi abhi chalega).
* Images ka result **seedha `result_data` data-url** me milta hai — polling ki
  jarurat hi nahi.
* Video worker ko jaata hai aur browser **worker se directly** status poochhta
  hai (`worker_base + /api/jobs/<remote_id>`); Redis ki jarurat nahi.
* Preview/mask bhi data-url me aate hain (`poster_data`, `mask_data`,
  `overlay_data`), to `/media/...` ke 404 ka koi chance nahi.

### Kaunsi cheez kahan chalti hai

| Asset | Vercel par | Worker par (agar set hai) |
|---|---|---|
| Image (≤ 4.5 MB) | detect + OpenCV inpaint, ~1-5 s, result data-url | GPU backend ho to behtar quality |
| Image (> 4.5 MB) | `WM_STORAGE=s3` lagao, browser seedha upload karega | – |
| Video (≤ 4.5 MB) | worker ko forward, browser worker se poll karta hai | poora pipeline + ffmpeg + audio |
| Video (> 4.5 MB) | `WM_STORAGE=s3` ke saath hi hoga | poora pipeline |

## C. Local (sabse pehle yahi test karo)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # video ke liye: -r requirements-local.txt
python tools/make_sample.py
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
python tests/test_pipeline.py
```

Serverless mode ko locally jaisa test karna ho:

```bash
VERCEL=1 WM_LIGHT=1 WM_WORKDIR=/tmp/wm-vercel \
  python -m uvicorn api.index:app --host 0.0.0.0 --port 8100
# doosra terminal (worker):
WM_WORKDIR=/tmp/wm-worker python -m uvicorn server.app:app --port 8101
# aur upar wale ko: WM_WORKER_URL=http://127.0.0.1:8101
```

## D. Vercel par storage kyun zaroori hai

Serverless request body **4.5 MB** tak hi jaata hai. Badi image/video ke liye:

1. Browser `GET /api/storage/presign?filename=clip.mp4` karta hai
   → `{"mode":"s3","key":"u/ab12/clip.mp4","url":"https://...presigned PUT","content_type":"video/mp4"}`
2. Browser seedha us URL par PUT karta hai (Cloudflare R2 / S3) — **bucket par
   CORS allow karna zaroori hai** (PUT + `Content-Type` header).
3. Browser `/api/process` ko `source_url` deta hai; Vercel function us URL ko
   worker ko de deta hai (bytes uske server se nahi guzarte).
4. Worker result banata hai aur apne `/media/results/...` par serve karta hai;
   browser wahin se download karta hai.

Local mode me `/api/storage/presign` `{"mode":"local"}` deta hai aur UI normal
multipart upload use karta hai (koi code change nahi).

## E. Common issues

| Symptom | Fix |
|---|---|
| `The functions property cannot be used in conjunction with the builds property` | purana `vercel.json` — ab repo me sirf `functions` hai (Vercel ka FastAPI preset `api/index.py` khud dhoondh leta hai) |
| Build ok par `ModuleNotFoundError: fastapi` | root me `requirements.txt` hona chahiye (yahi Vercel install karta hai) |
| `Function exceeded the time limit` | video job worker par nahi ja raha → `WM_WORKER_URL` + `WM_LIGHT=1` check karo |
| `413 / request too large` | `WM_STORAGE=s3` set karo (direct-to-blob upload) |
| Vercel build: bundle too large | `excludeFiles` (vercel.json) se `samples/ tests/ tools/ docs/` already exclude hain |
| Worker 401 | `WORKER_TOKEN` dono jagah same hona chahiye |
| Video job poll par 404 | browser worker ko directly poll karta hai — worker URL browser se reachable hona chahiye (CORS `*` hai) |
| Read-only file system error | light mode me workdir automatically `/tmp` ho jaata hai; `WM_WORKDIR` mat set karo Vercel par |
| ffmpeg missing (local) | `pip install imageio-ffmpeg` (Docker image me system ffmpeg hai) |
