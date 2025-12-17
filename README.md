# wildcam-motion-cnn

A research + engineering repo for **wildlife smart-camera recording on edge devices**.

**Goal:** capture the **entire animal behavior scene** (subtle motion, pauses, fast actions, interactions) while avoiding hours of empty/wind-only footage.

---

## Why this exists (problem statement)

Classic motion triggers struggle in real camera-trap conditions:

- **Night/IR + low contrast** → motion signals weaken → missed behavior (high FN)
- **Dynamic backgrounds** (grass/branches/water) → false triggers (high FP)
- **One global threshold** rarely works across day/night, animal scale, distance, and weather

So the research problem becomes:

> Build a **robust motion gate** that preserves behavior (high recall) while still saving storage/compute (high saved%) and running in real-time on constrained hardware.

---

## What’s implemented so far

### Stage 1 — ViBe baseline + evaluation
We implemented a simple **ViBe** background subtraction baseline and a reproducible evaluation script that writes per-video results to CSV:

**CSV metrics (frame-level):**
- Confusion matrix: `tp, fp, fn, tn`
- Accuracy: `precision, recall` (F1 is easy to add downstream)
- Runtime: `avg_fps, min_fps`
- Storage proxy: `saved_frames_pct` (percentage of original frames *not kept*)

> **Definition:** `saved_frames_pct = 100 - kept_frames_pct`, where `kept_frames_pct` is the fraction of frames predicted as motion.

⚠️ **Important:** this is **frame-level** gating. A real camera records **contiguous clips** with pre-roll/post-roll, so true storage savings will be lower than `saved_frames_pct`.

### Stage 1b — ViBe variants (AT / ALR / AT+ALR)
We added two scene-adaptive upgrades inspired by α‑ViBe style ideas:

- **AT (Adaptive Thresholding):** dynamic `min_matches` based on mean frame intensity (day vs night robustness)
- **ALR (Adaptive Learning Rate):** motion-aware background update rate (update slower during heavy motion to avoid “absorbing” animals)

## Data format

### `manifest.json`
A list of benchmark items:

```json
{
  "items": [
    {
      "id": "Test_Deer_Day",
      "video_path": "data/benchmark_videos/Test_Deer_Day.mp4",
      "fps": 30.0,
      "annotation_path": "data/benchmark_annotations/Test_Deer_Day.json"
    }
  ]
}
```

### Per-video annotation JSON
Event intervals in seconds:

```json
{
  "video_id": "Test_Deer_Day",
  "events": [
    { "start_sec": 10.0, "end_sec": 25.0 }
  ]
}
```

A frame is **GT-positive** if its timestamp falls inside any event interval.

---

## Install

```bash
pip install numpy opencv-python pyyaml tqdm
```

---

## Run evaluation

```bash
python scripts/eval_vibe.py 
```

### Notes on tqdm vs debug windows
- If `ui.show_debug: true`, the OpenCV window runs and **tqdm is disabled** (they fight over stdout).
- If tqdm appears “stuck”, run in a real terminal (not some IDE consoles) and ensure `show_debug: false`.

---

## Current benchmark summary (what the numbers say)

Across the same benchmark set (deer/fox, day/night), the mean results show the expected trade-offs:

| method | precision_mean | recall_mean | f1_mean | saved_frames_pct_mean | kept_frames_pct_mean |
|---|---:|---:|---:|---:|---:|
| Baseline (ViBe) | 0.9271 | 0.6582 | 0.7541 | 59.2867 | 40.7133 |
| AT | 0.8524 | 0.8182 | 0.8196 | 44.1135 | 55.8865 |
| ALR | 0.8167 | 0.7243 | 0.7498 | 49.1772 | 50.8228 |
| AT+ALR | 0.8118 | 0.8605 | 0.8198 | 37.4079 | 62.5921 |

**Interpretation (practical):**
- Baseline is the most “conservative”: best precision + best saved%, but misses behavior (lower recall).
- AT improves recall (fewer misses) but keeps more frames (saved% drops).
- ALR is more about stability and avoiding model corruption under motion; in our runs it slightly improved recall but not F1.
- AT+ALR yields the best recall/F1, but it keeps the most frames — it’s the most “don’t-miss-behavior” setting.

---

## Conclusions (so far)

1. **Fixed-parameter ViBe is not robust across day/night + dynamic backgrounds.**
2. **There’s a real research trade-off:** raising recall often reduces saved%.
3. The right operating point depends on the system objective:
   - **Behavior capture first:** prefer AT+ALR (then add clip-level logic so you don’t over-save).
   - **Storage first:** keep baseline/ALR and recover recall with temporal logic and selective confirmation.

---

## Next step: implement MHI (Motion History Image)

