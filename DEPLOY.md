# Deploy: Vercel (UI + API) + worker (video)

Repo: **https://github.com/checkmybio966-hub/Vimake/tree/arena/01a07c00-vimake**

```
Browser ──► Vercel (static UI + FastAPI serverless)
              │  images: yahin process (OpenCV inpainting, 1-5 s)
              │  videos: WM_WORKER_URL par forward  (serverless limit 60 s)
              ▼
        Worker service (Docker / Railway / Render / Fly / apna GPU box)
              ffmpeg + OpenCV + (optional) LaMa / ProPainter
```

Vercel par video isliye alag bhejte hain: serverless function ki timeout
(Hobby 60 s / Pro 300 s), 4.5 MB request-body limit aur ~250 MB bundle limit
hoti hai. **Worker wahi repo hai — bas full mode me.**

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
2. Framework preset: **Other** (ya "Python"). Build command khali chhod do —
   `vercel.json` sab handle karta hai.
3. Environment variables (Settings → Environment Variables):

| Var | Value | Kab |
|---|---|---|
| `WM_LIGHT` | `1` | hamesha (video worker ko bhejne ke liye) |
| `WM_WORKER_URL` | `https://worker.up.railway.app` | video chahiye to |
| `WORKER_TOKEN` | worker wala secret | agar worker par set hai |
| `REPLICATE_API_TOKEN` / `FAL_KEY` | token | images ko GPU se theek karna hai to |
| `WM_STORAGE` | `s3` | 4.5 MB se badi files ke liye |
| `S3_BUCKET`, `S3_REGION`, `S3_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `S3_PUBLIC_BASE_URL` | R2/S3 creds | `WM_STORAGE=s3` ke saath |
| `REDIS_URL` | Upstash `redis://...` | job state persist rakhna hai to |

4. **Deploy** → `https://<project>.vercel.app` khul jaayega.

> Hobby plan par function timeout 60 s hai (`vercel.json` me `maxDuration: 60`).
> Pro par ise 300 tak badha sakte ho, par video ke liye worker hi sahi hai.

### Kaunsi cheez kahan chalti hai

| Asset | Vercel par | Worker par (agar set hai) |
|---|---|---|
| Image (≤ 4.5 MB) | detect + OpenCV inpaint, ~1-5 s | GPU backend ho to behtar quality |
| Image (> 4.5 MB) | `WM_STORAGE=s3` lagao, browser seedha upload karega | – |
| Video | sirf queue (turant job_id) | poora pipeline + ffmpeg + audio |

## C. Local (sabse pehle yahi test karo)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r server/requirements.txt
python tools/make_sample.py
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
python tests/test_pipeline.py
```

## D. Vercel par storage kyun zaroori hai

Serverless request body **4.5 MB** tak hi jaata hai. Badi image/video ke liye:

1. Browser `GET /api/storage/presign?filename=clip.mp4` karta hai
   → `{"mode":"s3","key":"u/ab12/clip.mp4","url":"https://...presigned PUT"}`
2. Browser seedha us URL par PUT karta hai (Cloudflare R2 / S3).
3. Browser `/api/process` ko `source_url` deta hai; Vercel function us URL ko
   worker ko de deta hai (bytes uske server se nahi guzarte).
4. Worker result ko storage me daal kar URL return karta hai.

Local mode me `/api/storage/presign` `{"mode":"local"}` deta hai aur UI normal
multipart upload use karta hai (koi code change nahi).

## E. Common issues

| Symptom | Fix |
|---|---|
| `Function exceeded the time limit` | video job worker par nahi ja raha → `WM_WORKER_URL` + `WM_LIGHT=1` check karo |
| `413 / request too large` | `WM_STORAGE=s3` set karo (direct-to-blob upload) |
| Vercel build: bundle too large | `requirements-vercel.txt` hi use ho raha hai na? `server/requirements.txt` nahi |
| Worker 401 | `WORKER_TOKEN` dono jagah same hona chahiye |
| Video download blank | `result_url` worker ka relative path hota hai — API use worker URL se jod deta hai; worker reachable hai na check karo |
| ffmpeg missing (local) | `pip install imageio-ffmpeg` (Docker image me system ffmpeg hai) |
