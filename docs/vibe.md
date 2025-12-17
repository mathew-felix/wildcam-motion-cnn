# ViBe motion detection baseline — evaluation notes

This note summarizes the frame-level results in `vibe_eval.csv` and translates them into **insights + a clear research direction** for your wildlife motion-gating pipeline.

## What this evaluation measures

- **Ground truth**: event intervals (`start_sec`, `end_sec`) in each video’s annotation JSON.
- **Labeling rule**: a frame is **positive** if its frame index lies inside any GT event interval; otherwise **negative**.
- **Prediction rule**: ViBe produces a foreground mask; a frame is **predicted positive** if foreground pixels exceed a threshold (`min_fg_pixels` or `min_fg_ratio`).
- **Confusion matrix**: TP/FP/FN/TN are **frame-level** (after warmup frames).
- **Saved percentage**: `saved_frames_pct = 100 - kept_frames_pct`, where `kept_frames_pct` is the percentage of frames flagged as motion (i.e., frames you’d keep if you only recorded motion-positive frames).  
  *Note:* in a real camera, you’ll usually record **contiguous clips** with pre-roll/post-roll, so real “saved%” will be lower than this number.

## Results (per video)

| video_id   |   precision |   recall |    f1 |   specificity |   saved_frames_pct |   kept_frames_pct |   avg_fps |   min_fps |   tp |   fp |   fn |   tn |   gt_pos_rate |   pred_pos_rate_scored |
|:-----------|------------:|---------:|------:|--------------:|-------------------:|------------------:|----------:|----------:|-----:|-----:|-----:|-----:|--------------:|-----------------------:|
| deer_day   |       0.74  |    0.53  | 0.618 |         0.72  |             57.435 |            42.565 |    46.012 |     9.411 | 1861 |  655 | 1649 | 1686 |         0.6   |                  0.43  |
| deer_night |       1     |    0.43  | 0.601 |         1     |             80.296 |            19.704 |    49.09  |     8.201 |  638 |    0 |  847 | 1693 |         0.467 |                  0.201 |
| fox_day    |       0.992 |    0.975 | 0.984 |         0.987 |             39.97  |            60.03  |    40.176 |     8.901 |  790 |    6 |   20 |  450 |         0.64  |                  0.629 |
| fox_night  |       0.976 |    0.698 | 0.814 |         0.977 |             59.445 |            40.555 |    44.162 |     9.399 | 1570 |   38 |  680 | 1617 |         0.576 |                  0.412 |

## Quick takeaways (numbers that matter)

- Average precision ≈ **0.927**, average recall ≈ **0.658**, average F1 ≈ **0.754**.
- Average saved frames ≈ **59.3%** (aggressive gating on average).
- Speed on this machine: average throughput ≈ **44.9 FPS**, but **min FPS drops to ~8–9** in all runs (expect worse on Jetson; these spikes matter for real-time).

### Sequence-level patterns

- **Best case (high recall): `fox_day`**
  - Precision 0.992, recall 0.975.  
  - Interpretation: strong/clear motion + stable background → ViBe works extremely well.

- **Worst case (low recall): `deer_night`**
  - Precision 1.000, recall 0.430.  
  - Interpretation: ViBe becomes **too conservative** (many missed frames), typically caused by low contrast, sensor noise/IR at night, or subtle animal motion.

### Failure modes suggested by these results

1. **Night/low-light = high FN (misses)**
   - Your “deer_night” style results show near-zero FP but poor recall → the thresholding + background model is not sensitive enough under noise/low contrast.

2. **Day vegetation/wind = higher FP**
   - “deer_day” has noticeably higher FP than the fox sequences → likely moving grass/branches or global illumination changes increasing false triggers.

3. **A single global threshold is not stable**
   - One `min_fg_pixels` setting cannot simultaneously handle: bright day + wind + dark night IR + different animal sizes/distances.

## Research problem (paper/thesis framing)

**Problem:** Build a **robust motion-gating system for wildlife camera traps** on constrained edge hardware that:
- **does not miss animal behavior** (high recall / low FN) across day/night and background motion,
- **reduces recording and compute** (high saved% / low false triggers),
- runs **real-time** on an edge device (FPS + memory + power).

The results show why this is a real research problem: ViBe can be near-perfect in some scenes, but recall can collapse in others (especially at night), and FP can rise with vegetation motion.

## Direction: what to do next (concrete + research-worthy)

### 1) Stop optimizing only frame-level metrics → evaluate **event-level** detection
Your goal is “record the animal behavior scene,” not “label every frame perfectly.”

Add an **event-level** evaluation:
- Convert predictions into events using hysteresis (e.g., start when motion persists for K frames; end after L quiet frames).
- Score with event-precision / event-recall (did we capture the event?) + temporal coverage (how much of the event was recorded?).

This will align the metric with your real objective and usually makes the system look more realistic than frame-level scoring.

### 2) Add temporal logic (cheap) to fix both FP and FN
Use three knobs that are easy, fast, and publishable as ablations:

- **Pre-roll buffer** (e.g., keep 1–2 seconds of frames before the first motion trigger).
- **Post-roll hold** (keep recording for 1–3 seconds after motion disappears).
- **Hysteresis thresholds**: one threshold to start recording, a lower one to stay recording.

This will significantly increase “behavior capture” without needing YOLO-level compute.

### 3) Make thresholds adaptive (scene-aware), not fixed
Replace a fixed `min_fg_pixels` with an adaptive threshold based on background noise:
- maintain a running statistic of FG pixel counts during “background-only” periods,
- set threshold = mean + α·std (or percentile-based threshold),
- optionally change threshold for day vs night (estimated from frame brightness).

This directly targets why `deer_night` is missing so much: the FG signal is weaker relative to noise.

### 4) Reduce flicker/vegetation motion (your biggest FP source)
Two promising paths (both fit your project):

- **Dynamic background suppression / flicker reduction** (e.g., WisenetMD-style suppression you mentioned before).
- **Foreground mask stabilization**: morphological + connected components + discard tiny blobs + region persistence.

A good paper claim here is: “reduce false triggers from natural background motion while preserving recall.”

### 5) Two-stage gate (ViBe → lightweight CNN) but used *surgically*
A CNN shouldn’t run on every frame. Instead:

- Run ViBe on all frames (cheap).
- If ViBe fires, crop candidate regions (merged blobs) and run a small CNN to confirm “animal vs not”.
- To recover false negatives, periodically sample frames (e.g., every N frames) and run CNN only if ViBe is quiet for too long but conditions suggest potential motion.

This keeps compute low while fixing ViBe’s scene-dependent brittleness.

### 6) Research deliverables you can produce from here (realistic)
- PR/saved% curves by sweeping `radius`, `min_matches`, threshold, and mask post-processing.
- Day vs night analysis + failure case gallery (best vs worst videos).
- Event-level evaluation results with pre/post-roll + hysteresis (ablation table).
- Jetson profiling (FPS + memory) for: ViBe-only vs ViBe+hysteresis vs ViBe+CNN.

## Practical recommendation based on your numbers

- **ViBe alone is not sufficient as a “don’t-miss-anything” gate**, because recall drops hard on some sequences.
- But **ViBe is a strong first-stage filter** (good precision in most cases and high saved%).
- Your research path should be: **ViBe (fast) + temporal logic (cheap) + adaptive thresholds (robust) + optional CNN confirmation (selective)**.

---

### File provenance
Generated from: `vibe_eval.csv` (4 benchmark videos) on your current configuration/run.

