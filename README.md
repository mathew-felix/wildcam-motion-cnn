# wildcam-motion-cnn — motion gate for wildlife smart cameras

**Research + engineering repo** for motion-triggered recording on constrained edge devices (Jetson-class).

**Core goal:** record the **entire animal behavior scene** (subtle motion, pauses, fast actions) while avoiding hours of empty / wind-only footage.

---

## Why this is a research problem

Outdoor camera-trap footage breaks naive motion triggers:

- **Night / IR / low contrast** → weak motion signal → many missed frames (**FN**)
- **Dynamic backgrounds** (grass, branches, water, snow) → false triggers (**FP**)
- A single global threshold rarely generalizes across illumination + weather + animal scale

So the practical research objective is:

> Build an **edge‑efficient motion gate** that preserves behavior (high recall / event capture) while still saving storage + compute (high saved%) and running real time.

---

## What’s implemented

### Stage 1 — ViBe baseline (frame-level gate)
- `src/motion/vibe.py`: ViBe background subtraction (grayscale uint8)
- `scripts/eval_vibe.py`: frame‑level evaluation against event-interval GT
- Outputs a CSV with: `tp, fp, fn, tn, precision, recall, avg_fps, min_fps, saved_frames_pct`

**Important:** frame‑level “saved%” is an optimistic proxy. Real camera behavior is **clip-based** (pre/post roll), so true saved% will be lower.

### Stage 1b — ViBe variants: AT / ALR / AT+ALR
These are scene-adaptive upgrades inspired by “make the gate robust across conditions”:

- **AT (Adaptive Thresholding):** dynamic `min_matches` based on mean frame intensity  
  *Day vs night robustness; reduces missed behavior under low-light.*
- **ALR (Adaptive Learning Rate):** motion-aware update rate  
  *Slows background updates during heavy motion to avoid absorbing moving objects.*
- **AT+ALR:** both enabled.

### Stage 2 — ViBe + MHI (clip-like gate)
- `src/motion/vibe_mhi.py`: Motion History Image (short-term motion memory) + hysteresis gate
- `scripts/eval_vibe_mhi.py`: clip-like evaluation with START/ STAY logic

Key design choice (important for outdoor footage):
- **START** uses a strict instantaneous ViBe metric (prevents “wind latch”)
- **STAY** uses lenient MHI evidence (bridges pauses / flicker → better behavior continuity)

---

## Benchmark format

### `manifest.json`
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

### Per‑video annotation JSON
```json
{
  "video_id": "Test_Deer_Day",
  "events": [
    { "start_sec": 10.0, "end_sec": 25.0 }
  ]
}
```

A frame is **GT‑positive** if its timestamp falls inside any event interval.

---

## Install

```bash
pip install numpy opencv-python pyyaml tqdm
```

---

## Run evaluations

### ViBe (frame-level)
```bash
python scripts/eval_vibe.py --config configs/eval_vibe.yaml
```

### ViBe + MHI (clip-like)
```bash
python scripts/eval_vibe_mhi.py --config configs/eval_vibe_mhi.yaml
```

### tqdm vs OpenCV debug window
- If `ui.show_debug: true`, the OpenCV window runs and **tqdm is disabled** (stdout contention).
- If tqdm looks “stuck”, run from a real terminal (PowerShell / cmd / bash), not some IDE consoles.

---

## Results summary (mean over benchmark videos)

The table below is computed from the CSVs in `results/` (mean across videos; plus runtime & saved%).

| variant           |   precision_mean |   recall_mean |   f1_mean |   saved_frames_pct_mean |   avg_fps_mean |   min_fps_mean |
|:------------------|-----------------:|--------------:|----------:|------------------------:|---------------:|---------------:|
| MHI – ALR         |           0.8059 |        0.7166 |    0.7451 |                 49.8054 |        41.0672 |         8.4864 |
| MHI – AT          |           0.8391 |        0.8065 |    0.8116 |                 44.3287 |        40.0045 |         8.07   |
| MHI – AT+ALR      |           0.8046 |        0.8551 |    0.8162 |                 37.7392 |        38.7109 |         7.4114 |
| MHI – Baseline    |           0.8931 |        0.6666 |    0.7543 |                 58.4262 |        41.5509 |         8.519  |
| MHI – HighRecall  |           0.8145 |        0.9267 |    0.8566 |                 33.6309 |        40.0948 |         8.3663 |
| No MHI – ALR      |           0.8167 |        0.7243 |    0.7498 |                 49.1772 |        39.8172 |         7.1669 |
| No MHI – AT       |           0.8524 |        0.8182 |    0.8196 |                 44.1135 |        39.6682 |         6.5132 |
| No MHI – AT+ALR   |           0.8118 |        0.8605 |    0.8198 |                 37.4079 |        42.2835 |         7.6946 |
| No MHI – Baseline |           0.9271 |        0.6582 |    0.7541 |                 59.2867 |        42.5113 |         7.9968 |

**HighRecall vs No‑MHI AT+ALR (micro):** TP +656, FP +29, FN -656.

### Per-video snapshot (F1 and Saved% for key variants)

| video_id   |   No MHI – Baseline F1 |   No MHI – Baseline Saved% |   No MHI – AT+ALR F1 |   No MHI – AT+ALR Saved% |   MHI – AT+ALR F1 |   MHI – AT+ALR Saved% |   MHI – HighRecall F1 |   MHI – HighRecall Saved% |
|:-----------|-----------------------:|---------------------------:|---------------------:|-------------------------:|------------------:|----------------------:|----------------------:|--------------------------:|
| deer_day   |                 0.6177 |                    57.4353 |               0.756  |                  28.5231 |            0.7333 |               31.9405 |                0.8274 |                   21.1978 |
| deer_night |                 0.601  |                    80.2965 |               0.7554 |                  67.6961 |            0.7648 |               65.5034 |                0.8315 |                   59.9135 |
| fox_day    |                 0.9838 |                    39.9698 |               0.9838 |                  39.9698 |            0.9832 |               40.0452 |                0.9838 |                   39.9698 |
| fox_night  |                 0.8139 |                    59.4451 |               0.7839 |                  13.4426 |            0.7837 |               13.4678 |                0.7839 |                   13.4426 |

---

## What the numbers mean (research interpretation)

### Without MHI (pure frame gate)
- **Baseline** is the most conservative: best precision + best saved%, but misses behavior (lower recall).
- **AT / AT+ALR** typically raises recall (fewer missed behavior frames) but keeps more frames (saved% drops).  
  This is a standard trade-off: **behavior capture vs storage**.

### With MHI (clip-like gate)
- MHI helps most when motion is intermittent (night/IR, pauses, flicker).  
- But MHI can **amplify noisy FG masks** (wind speckles) if blob filtering is weak → “latching” (recording too long).

### High‑Recall gate
A tuned **MHI HighRecall** setting is useful when your objective is “don’t miss behavior”:
- It increases recall and F1 by primarily converting **FN → TP**, with only a small FP cost.

---

## Recommended configs

- **Frame-level baseline/ablations:** `configs/eval_vibe.yaml` (+ AT/ALR toggles under `vibe:`)
- **Clip-like gate baseline:** `configs/eval_vibe_mhi.yaml`
- **MHI tuned presets:**
  - `configs/eval_vibe_mhi_balanced.yaml` — reduce fox_night latch + still improve deer_day starts
  - `configs/eval_vibe_mhi_high_recall.yaml` — maximize behavior capture (accept more kept frames)

---

## Next research steps (publishable direction)

1) **Switch evaluation from frame-level to event-level**
   - Event recall: % of GT intervals that are captured (≥X% coverage)
   - Mean temporal coverage per event
   - Fragmentation metrics (number of clips per event)

2) **Night-aware gating**
   - Use intensity bins to adapt: blob area thresholds, MHI decay, stay thresholds

3) **Two-stage confirmation (selective CNN)**
   - Run CNN only when ViBe+MHI suggests an event and blobs look plausible  
   - This targets FP from vegetation without YOLO-level compute.

4) **Downstream ROI-aware compression**
   - Once clip gating is robust, integrate ROI extraction + H.264 ROI (or learned codec baselines)
   - Report bitrate/quality trade-offs under “record less but keep ROI quality” constraints

---

## Repo map (suggested)

- `src/motion/vibe.py` — ViBe (+ optional AT/ALR)
- `src/motion/vibe_mhi.py` — MHI + hysteresis clip gate
- `scripts/eval_vibe.py` — frame-level evaluation → CSV
- `scripts/eval_vibe_mhi.py` — clip-like evaluation → CSV
- `configs/` — YAML configs for reproducibility
- `data/benchmarks/manifest.json` — benchmark manifest
- `results/` — CSV outputs

---

## References (high-level)
- ViBe background subtraction (classic sample-consensus BGS)
- Motion History Images (temporal motion templates)
- “Adaptive motion detection” style ideas motivating AT/ALR parameters

(Full bib will go into the thesis/paper, but the code is organized to support clean ablations.)

---

*Last updated: December 17, 2025*
