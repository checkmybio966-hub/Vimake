# Full worker image: OpenCV + ffmpeg, videos bhi yahin process hote hain.
# Railway / Render / Fly / apne GPU box par chalao, aur Vercel se
# WM_WORKER_URL=<ye service> set kar do.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    WM_WORKDIR=/data \
    PORT=8000

# ffmpeg (system) + build tools for opencv wheel fallback
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY server ./server
COPY web ./web
COPY api ./api

EXPOSE 8000
HEALTHCHECK --interval=60s --timeout=10s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "120"]
