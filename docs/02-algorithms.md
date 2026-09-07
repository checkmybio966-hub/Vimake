# Detection + removal algorithms, poori detail me

Ye file is project ka "dimag" hai. Har formula ke saath likha hai ki wo kaam
kyun karta hai aur kab fail hota hai.

---

## Part A — Video: time axis se signal nikalna

Video me humein photo se ek cheez extra milti hai: **TIME**. Isse do bilkul
alag signal milte hain, aur dono milkar dono tarah ke watermark cover karte hain.

Notation: grayscale frames `F_t`, `T` sampled frames (default 40–48, uniformly).

### A1. Temporal median = "clean plate"

```
M(x) = median_t F_t(x)
```

Kisi pixel par agar watermark **kuch hi frames** me hai (moving mark), to us
pixel ki history me zyadaatar frames saaf hote hain → `M(x)` ≈ asli background.

### A2. FIXED watermark ke signals

Fixed mark har frame me hota hai, isliye wo apne aas-paas ke scene jaisa behave
hi nahi karta:

**Variance attenuation `A`** — ek constant layer, upar chalte hue content ke
ooper, motion ko *chapta (flatten)* kar deta hai:

```
S(x)     = std_t F_t(x)              # temporal variance
Sblur(x) = blur(S, 25x25)            # aas-paas ka variance level
A(x)     = clamp((Sblur - S) / Sblur, 0, 1)
```
Mark ke andar `S` chhota, aas-paas bada → `A ≈ 1`.

**Edge persistence `P`** — logo ki outline har frame me *exact same pixel* par
hoti hai, jabki scene ke edges aate-jaate hain:

```
P(x) = mean_t [ |Laplacian(F_t)| > 12 ]
```

**Static score:** `static = 0.55·A_norm + 0.45·P_norm`, phir top 4% pixels seed
ban kar connected components me clean hote hain (open → close → area filter).

> Hamare synthetic fixed-watermark video me: `A` inside 0.73 vs outside 0.04
> (**18×**), `P` inside 0.77 vs outside 0.13 (**5.9×**).

**Contrast test (ye sabse important hai).** Sirf "A high" kaafi nahi hai —
ek chalte hue object ka *andar* bhi flat hota hai (solid colour) aur uski
variance kam hoti hai. Isliye hum **andar vs bahar ka ratio** dekhte hain:

```
static_ok  =  P_in > 0.40  AND  P_in/(P_out+0.05) > 2.0  AND  A_in > 0.30
              AND  z_in < 2.0      # fixed mark constant hai → outlier hi nahi
```

### A3. MOVING watermark ka signal — adaptive robust z-score

Moving mark ke liye fixed threshold kaam nahi karta (busy footage me har pixel
ka residual bada hota hai). Isliye **har pixel apna khuda ka reference** banata
hai:

```
r_t(x)  = |F_t(x) - M(x)|                 # residual
med_r   = median_t r_t
mad_r   = median_t |r_t - med_r| · 1.4826
z_t(x)  = (r_t - med_r) / (mad_r + 2)
```

Mark jis pixel ke upar se guzarta hai, us pixel ke liye wo *apni hi history ka
outlier* hota hai → `z` bada. Fixed mark constant hota hai → `z ≈ 0`.
Ye ek hi statistic dono case alag karta hai:

```
moving_ok =  z_in > 3.0  AND  z_in > 2·z_out  AND  P_in > 0.25
```

`P_in > 0.25` wali shart zaroori hai: moving object ke **edge** bhi outlier hote
hain, par wo har frame alag jagah hote hain (persistence ≈ 0.06), jabki ek
drift karta hua watermark apne aap ko repeat karta hai (persistence ≈ 0.56).

### A4. Detection ka decision

```
static_ok aur (moving_ok nahi, ya s_conf >= m_conf)  →  kind = "static"
moving_ok                                            →  kind = "moving"
warna static mask hai to                             →  kind = "static" (weak)
warna                                                →  "not found" → brush
```

---

## Part B — Image: bina time ke kaise dhoondhein

Still image me koi motion nahi. Hum ** kamzor par saste cues** jodte hain,
sabse reliable se shuru karke.

### B1. Ink mask (sabse pehli cheez)

Watermark locally-bright (ya dark) layer hota hai smooth background ke upar.
Isliye har pixel ko **bade local median** se compare karo:

```
loc = medianBlur(gray, 31)
res = gray - loc
mad = 1.4826 · median(|res - median(res)|)
ink = |res| > max(5, 2.6·mad)
```

Ye glyph ko **bharta** hai (sirf outline nahi) — inpainting model ke liye yahi
chahiye.

### B2. Repeated-shape detection (sabse strong signal)

Shutterstock-jaisa grid aur repeated `@handle` ek hi glyph ko kai baar print
karta hai. Algorithm:

1. `ink` mask se connected blobs nikaalo (Canny edges se stroke-proposal +
   MSER groups).
2. Blobs ko **shape** se cluster karo:
   * bbox size ±35% ke andar
   * binary mask IoU ≥ 0.55 (donon ko common canvas par resize karke)
3. Cluster me ≥2 member ho → "ye repeated element hai".
4. Representative crop ko **binary ink mask** par `matchTemplate` se poori image
   me dhoondo (threshold 0.60, NMS) → saare instances.

> **Binary mask par match karna hi raaz hai.** Pixels par template matching is
> wajah se fail hota hai ki har tile ke neeche background alag hai
> (`0.5·text + 0.5·background` alag-alag correlation deta hai). Binary shape par
> background hai hi nahi.

Is repo ke sample par ye **9/9 tiles** pakadta hai, IoU **0.64**.

### B3. Stroke + priors (general case: ek logo, ek timestamp)

Repeated kuch na mile to har blob ko score karte hain:

| Cue | Formula / intuition | Weight |
|---|---|---|
| position | `exp(-d_border/0.10)` + corner bonus | 0.34 |
| size | 0.05%..8% of image | 0.18 |
| textness | mean stroke width = 2·area/perimeter, stroke-width consistency | 0.20 |
| translucency | low saturation + mid luminance | 0.16 |
| fill ratio | strokes block ka ~30% | 0.12 |

Sabse upar wala blob return hota hai (agar detection na chale to UI bolta hai:
"brush karo").

### B4. Production me: learned detector

`detect_image.ExternalDetector` me apna detector plug karo — YOLOv8/YOLO11
watermark boxes (best), DBNet/PaddleOCR (text overlays), ya Grounding-DINO
prompt `"watermark, logo, text overlay"` (zero-shot).

---

## Part C — Removal

### C1. Image

```
mask = dilate(auto ∪ brush − eraser) → feather
out  = backend.inpaint(frame, mask)        # LaMa / IOPaint / OpenCV
```

### C2. Video — ek hi propagation engine, do modes

Har neighbour frame ko reference frame ki geometry me laana zaroori hai, warna
background chalne par fill smear ho jaata hai. Do candidate model banate hain
aur **jo tight ho wo jeetta hai** (`_plate_stats` se residual MAD compare karke):

* **RAW** — neighbours jaise hain. Steady camera par best: koi resampling nahi,
  koi flow error nahi → residual spread chhota → mark saaf dikhta hai.
* **WARPED** — dense optical flow (DIS/Farneback) se neighbours ko align karke.
  Camera/subject really move kare to zaroori — par warping apna noise laata hai.

> Hamare sample par: RAW → **44.9 dB**, WARPED → ~38 dB. Isliye choice automatic
> hai, hard-coded nahi.

**Mode "static" (fixed logo).** Time se background nahi mil sakta (mark har
frame me hai), isliye:

```
fill(x) = Σ_d w_d · F_{t+d}(x + flow) · valid_d(x) / Σ_d w_d · valid_d(x)
valid_d = neighbour me wo pixel jo masked nahi hai
w_d     = exp(-|d| / (W/1.6))                    # nazdeeki frames ko zyada weight
```
Jo pixel kisi bhi neighbour me nahi mila → `cv2.inpaint` (Telea) se band.
Phir mask ke andar temporal smoothing (EMA) taaki flicker na ho.
Ye effectively **ProPainter ka poor-man version** hai: flow completion + spatial
diffusion, bina GPU.

**Mode "moving" (floating mark).** Yahan time deta hai:

```
plate   = robust temporal median (outlier-rejected) of the neighbour stack
z       = (|ref - plate| - med_d) / (1.4826·mad_d)
alpha   = clip((z - k)/(0.8k), 0, 1) · prior · shape_gate
out     = ref·(1-alpha) + plate·alpha
```

* `k` = `WM_CP_Z` (default 5). Badhao = zyada conservative.
* `prior` = detection wala union mask (safety: sirf wahan chhoo jahan mark
  guzarta hai) — note ki isme 0.25 ka floor hai, taaki thoda-sa ghalat detection
  removal ko disable na kar de.
* `shape_gate` = sirf compact, solid blobs (solidity ≥ 0.45, extent ≥ 0.25) —
  moving object ke edges (thin arcs/rings) isse reject ho jaate hain.

**Window size** = `WM_MOVING_WINDOW` (default 24 frames, stride 2 → ±1 sec).
Rule: window ko mark ke *dwell time* se bada hona chahiye
(`dwell ≈ mark_width / speed`), warna median contaminate ho jaata hai.

### C3. Audio + encoding

Frames ko `ffmpeg` raw pipe me bheja jaata hai:

```
-f rawvideo -pix_fmt bgr24 -s WxH -r FPS -i -
-i <original>            # audio ke liye
-c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart
-map 0:v:0 -map 1:a:0? -c:a aac -b:a 160k -shortest
```

Isliye **audio original rehta hai** (`-c:a aac` sirf remux ke liye).

---

## Part D — Kab fail hota hai (aur kya karein)

| Situation | Kya hota hai | Solution |
|---|---|---|
| Static mark + tripod (zero motion) | A bhi ~0, P bhi everywhere high | Brush se region batao; UI me "fixed position" select karo |
| Mark kisi moving subject ke upar | fill galat texture | `WM_CP_Z` badhao; LaMa/ProPainter lagao |
| Bohot fast camera pan | flow error | RAW model automatically choose ho jaata hai; warna `WM_MOVING_WINDOW` ghatao |
| Textured background | CPU smudge | LaMa / IOPaint / Replicate backend |
| Chhota/faint tiled watermark | tile detector miss | brush se ek tile paint karo, phir "Process" — mask se hi kaam chalega |
