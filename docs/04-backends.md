# Inpainting backends — local CPU se lekar hosted GPU tak

`server/core/inpaint/` me sab pehle se hain. `WM_BACKEND=auto` best available
chun leta hai: **lama → iopaint → replicate/fal → opencv**.

```
                    ┌── quality ──┬── cost ──┬── latency (per 5s clip) ──┐
OpenCV (CPU)        │   0.35      │  free    │  ~1 s                     │
LaMa ONNX (CPU/GPU) │   0.90      │  free    │  2–20 s                   │
IOPaint (local GPU) │   0.92      │  free    │  1–5 s                    │
Replicate ProPainter│   0.88      │  ~$0.05  │  ~60 s (cold start alag)  │
fal / custom GPU    │   0.90      │  model-wise │ 10–60 s               │
```

## 1. OpenCV (default, zero-dependency)

Telea / Navier-Stokes diffusion, multi-scale (coarse→fine).
Chhote logo aur simple background ke liye theek; textured background par smudge.
`backend=opencv`.

## 2. LaMa — open model jo zyadatar "AI eraser" tools ke peeche hai

LaMa (Large Mask Inpainting with Fourier Convolutions, WACV 2022) bade masks par
bhi shape aur texture maintain karta hai.

```bash
pip install onnxruntime-gpu        # ya CPU ke liye onnxruntime
# ONNX export (big-lama) kahin bhi rakho, phir:
export LAMA_ONNX_PATH=/models/big-lama.onnx
export WM_BACKEND=lama
```

`lama_backend.LamaBackend` standard preprocessing karta hai (pad to multiple of
8, `[-1,1]`/ImageNet normalize, RGB, 4-channel input with mask).

## 3. IOPaint (lama_cleaner) — sabse tez production rasta

Ek command me LaMa / MAT / ZITS / SD / Anything + browser UI mil jaata hai:

```bash
pip install iopaint
iopaint start --model=lama --device=cuda --port=8080
export WM_IOPAINT_URL=http://127.0.0.1:8080
export WM_BACKEND=iopaint
```

Hamara `IOPaintBackend` uske `/api/v1/inpaint` ko base64 se call karta hai.

## 4. Replicate — video ke liye sabse relevant

Do models kaafi hain:

| Model | Input | Kab |
|---|---|---|
| `zylim0702/remove-object` (LaMa) | image + mask | photos |
| `jd7h/propainter` | video + mask (static ya per-frame) | fixed watermark |
| `jd7h/xmem-propainter-inpainting` | video + **pehle frame ka mask** | **best fit** — XMem mask ko poori video me propagate karta hai, phir ProPainter inpaint |

```bash
export WM_BACKEND=replicate
export WM_API_PROVIDER=replicate
export REPLICATE_API_TOKEN=r8_...
# image model override: export WM_API_MODEL=zylim0702/remove-object
```

```python
import replicate
out = replicate.run(
    "jd7h/xmem-propainter-inpainting",
    input={"video": open("clip.mp4","rb"), "mask": open("first_frame_mask.png","rb")})
```

Cost roughly **$0.05 per run** (L40S, ~53 s).

## 5. fal.ai

```bash
export WM_BACKEND=fal
export FAL_KEY=...
export WM_API_MODEL=<fal model id>      # e.g. koi eraser / inpainting model
```

`api_backend.FalBackend` queue API use karta hai
(`POST https://queue.fal.run/{model}` → poll `status_url`).

## 6. Custom / self-hosted

```bash
export WM_BACKEND=custom
export WM_API_BASE=https://gpu.mydomain.com/inpaint
export WM_API_KEY=...
```

Contract: `POST {"image": "<base64 png>", "mask": "<base64 png>"}` → raw image
bytes, ya `{"url": ...}` / `{"image_base64": ...}`. RunPod / Modal / Triton / ek
chhota FastAPI jo IOPaint wrap kare — sab isme fit ho jaate hain.

## 7. Ready-made watermark-removal APIs (agar khud model nahi chalana)

| Provider | Endpoint | Note |
|---|---|---|
| Vmodel (Vmake family) | `POST https://api.vmodel.ai/api/tasks/v1/create` | `{"version":"0ad4ca94...","input":{"video":url,"mask_stage":mask}}`, poll task, ~$0.02/second, max 1080p/3min |
| Unwatermark.ai | unwatermark.ai dashboard | vmodel model ki commercial API |
| DeWatermark | `POST https://platform.dewatermark.ai/api/object_removal/v2/erase` | `X-API-KEY`, multipart: `session_id, original_preview_image, mask_base, mask_brush, remove_text, predict_mode` |

In sabko `WM_BACKEND=custom` + thoda adapter likh kar chipka sakte ho —
`server/core/inpaint/api_backend.py` me `CustomBackend` dekho.

## Apna backend kaise jodo

```python
from server.core.inpaint.base import InpaintBackend

class MyBackend(InpaintBackend):
    name = "mine"
    quality = 0.95
    supports_video = False

    def inpaint(self, frame, mask):        # uint8 BGR, mask 255 = hole
        ...                                 # → uint8 BGR
        return filled

# server/core/inpaint/__init__.py me get_backend() me ek branch add kar do
```

Video-native backends ke liye `inpaint_video(video_path, mask_path, out_path)`
implement karo — pipeline usse prefer karega (frame-by-frame se tez hota hai).
