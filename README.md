# wildcam-motion-cnn

## Overview
This project targets **wildlife smart-camera recording on edge devices**: capture the **entire animal behavior scene** (subtle motion, pauses, fast actions, interactions) while avoiding hours of empty or wind-only footage.

## Motivation
Classic motion detection can miss important behavior moments:
- **Subtle motion / pauses** → low foreground response → missed frames (FN)
- **Dynamic backgrounds** (grass/branches, lighting changes) → false triggers (FP)
- **Motion blur / low light (night/IR)** → weaker signals and unstable thresholds

The goal is to build a robust, low-compute pipeline that records the right moments reliably on constrained hardware.

---

## Work Done

### Stage 1 — ViBe baseline (in progress)
Implemented a **simple ViBe (background subtraction) motion baseline** and benchmarked on **4 wildlife videos** (deer/fox; day/night).

**Outputs (CSV):**
- Confusion matrix (frame-level): `tp, fp, fn, tn`
- Accuracy: `precision, recall`
- Speed: `avg_fps, min_fps`
- Storage proxy: `saved_frames_pct` (**percentage of frames not kept** compared to total frames)

> `saved_frames_pct = 100 - kept_frames_pct`, where `kept_frames_pct` is the fraction of frames flagged as motion (i.e., frames you would keep if you only saved motion-positive frames).

---

## Data Format

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

A frame is **GT-positive** if its timestamp lies within any event interval.

---

## Running the ViBe Baseline

### Install
```bash
pip install numpy opencv-python pyyaml tqdm
```

### Configure (YAML)
Example: `configs/eval_vibe.yaml`
- `vibe.*`: ViBe params (`num_samples`, `min_matches`, `radius`, `subsampling_factor`)
- `postprocess.*`: mask cleanup + motion decision threshold (`min_fg_pixels` or `min_fg_ratio`)
- `ui.use_tqdm`: progress bars
- `ui.show_debug`: OpenCV debug window (**disables tqdm**)

### Run
```bash
python scripts/eval_vibe.py --config configs/eval_vibe.yaml
```

### Output
Writes a CSV such as:
```
results/vibe_eval.csv
```

---

## Next Steps (planned)
- **Adaptive thresholds** (day vs night)
