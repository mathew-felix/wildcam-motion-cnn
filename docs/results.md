# Results & Benchmarks (paper-style)

This page is meant to read like the **Evaluation** section of a paper.  
It documents datasets, metrics, ablations, and edge-simulation settings.

---

## 1) Benchmark set

**Videos (4):**
- `deer_day` (hard: subtle motion, long events)
- `deer_night` (hard: IR noise / flicker)
- `fox_day` (easy: clear motion)
- `fox_night` (hard: false-positive pressure test)

---

## 2) Metrics (definitions)

All metrics are computed on the evaluator’s scored range.

### Frame-level classification
- **TP**: animal present & frame kept  
- **FP**: no animal & frame kept (wasted storage)  
- **FN**: animal present & frame dropped (behavior loss)  

From these:
- **Precision = TP / (TP + FP)**  
- **Recall = TP / (TP + FN)**  
- **F1 = 2PR / (P + R)**  

### Storage
- **Storage_Saved_Pct**: percent frames dropped vs “record everything”.

### Behavior preservation
- **Animal_Coverage (0–1)**: fraction of animal timeline retained (higher = better).

### Temporal side-effects
- **Mean_Trigger_Delay_Sec**: how late recording begins after GT onset.
- **Mean_Extra_Tail_Sec**: how long recording continues after GT ends (wasted storage).

### Throughput
- **Avg_FPS**: end-to-end processing throughput in the evaluation run.

---

## 3) Edge simulation settings (Docker)

We simulate Jetson Orin Nano–class constraints using Docker resource caps (approximation):
- Memory capped to **4GB**
- CPU capped (e.g., **4 cores**)
- Limited shared memory (`--shm-size=512m`)

See `README_DOCKER.md` for exact commands and caveats (this is an approximation; a real-device validation run is still recommended).

---

## 4) Leaderboard (aggregate across 4 videos)

The table below aggregates across the benchmark set.

| Variant                           |   Precision |   Recall |    F1 | Coverage   | StorageSaved   | Delay   | Tail   |   FPS |
|:----------------------------------|------------:|---------:|------:|:-----------|:---------------|:--------|:-------|------:|
| ViBe + AT + ALR + MHI + MotionNet |       0.736 |    0.965 | 0.835 | 0.957      | 26.9%          | 0.23s   | 37.62s |  36.5 |
| vibe_AT_ALR_cnn                   |       0.736 |    0.963 | 0.834 | 0.801      | 27.1%          | 2.58s   | 9.24s  |  29.8 |
| ViBe + AT + ALR + MHI             |       0.742 |    0.938 | 0.828 | 0.939      | 29.5%          | 0.23s   | 37.23s |  46.2 |
| ViBe + MotionNet (CNN latch)      |       0.864 |    0.795 | 0.828 | 0.741      | 48.7%          | 1.73s   | 4.70s  |  35.3 |
| vibe_MHI_cnn                      |       0.809 |    0.806 | 0.807 | 0.808      | 44.4%          | 0.41s   | 7.22s  |  32.3 |
| vibe_AT_ALR                       |       0.729 |    0.864 | 0.791 | 0.855      | 34.0%          | 2.32s   | 37.08s |  36.6 |
| vibe_MHI                          |       0.806 |    0.738 | 0.771 | 0.717      | 48.9%          | 0.65s   | 6.88s  |  43.1 |
| vibe_AT                           |       0.782 |    0.726 | 0.753 | 0.728      | 48.2%          | 0.66s   | 9.24s  |  35.1 |
| vibe_ALR                          |       0.742 |    0.629 | 0.681 | 0.690      | 52.7%          | 2.65s   | 9.58s  |  37   |
| ViBe (baseline)                   |       0.858 |    0.521 | 0.649 | 0.554      | 66.1%          | 2.71s   | 2.18s  |  34.6 |
| MOG2 (baseline)                   |       0.567 |    1     | 0.724 | —          | 0.0%           | —       | —      | 370.6 |
| MOG2 + MotionNet                  |       0.858 |    0.476 | 0.612 | —          | 68.6%          | —       | —      | 362.2 |

**How to read this:**
- If your goal is **behavior preservation**, prioritize **Animal_Coverage** and **Recall**.
- If your goal is **battery/storage**, prioritize **Storage_Saved_Pct** (but check coverage doesn’t collapse).
- If you care about **edge viability**, check **Avg_FPS** and also look at per-frame latency in detection logs.

---

## 5) Per-video results

| video_id   | Variant                           |   Precision |   Recall |   F1_Score |   Animal_Coverage | Storage_Saved_Pct   | Mean_Trigger_Delay_Sec   | Mean_Extra_Tail_Sec   |   Avg_FPS |
|:-----------|:----------------------------------|------------:|---------:|-----------:|------------------:|:--------------------|:-------------------------|:----------------------|----------:|
| deer_day   | vibe_AT_ALR_cnn                   |       0.722 |    0.999 |      0.838 |             0.95  | 17.8%               | 2.66s                    | 5.88s                 |      29.3 |
| deer_day   | ViBe + AT + ALR + MHI + MotionNet |       0.721 |    1     |      0.838 |             0.992 | 17.6%               | 0.01s                    | 52.94s                |      37   |
| deer_day   | ViBe + AT + ALR + MHI             |       0.725 |    0.963 |      0.827 |             0.962 | 21.2%               | 0.01s                    | 52.27s                |      46.3 |
| deer_day   | vibe_MHI_cnn                      |       0.797 |    0.813 |      0.805 |             0.828 | 39.5%               | 0.01s                    | 10.47s                |      30.6 |
| deer_day   | ViBe + MotionNet (CNN latch)      |       0.798 |    0.805 |      0.802 |             0.813 | 40.1%               | 0.72s                    | 10.30s                |      35.8 |
| deer_day   | vibe_MHI                          |       0.789 |    0.743 |      0.766 |             0.711 | 44.1%               | 0.53s                    | 10.47s                |      38.9 |
| deer_day   | vibe_AT_ALR                       |       0.697 |    0.838 |      0.761 |             0.794 | 28.6%               | 4.82s                    | 52.27s                |      37.9 |
| deer_day   | vibe_AT                           |       0.65  |    0.651 |      0.651 |             0.693 | 40.5%               | 0.46s                    | 19.08s                |      27.5 |
| deer_day   | vibe_ALR                          |       0.601 |    0.437 |      0.506 |             0.594 | 56.8%               | 5.09s                    | 15.83s                |      37.3 |
| deer_day   | ViBe (baseline)                   |       0.677 |    0.357 |      0.468 |             0.471 | 68.7%               | 5.13s                    | 4.91s                 |      32.7 |
| deer_night | vibe_AT_ALR_cnn                   |       0.855 |    0.841 |      0.848 |             0.818 | 54.9%               | 0.45s                    | 5.37s                 |      31   |
| deer_night | ViBe + AT + ALR + MHI + MotionNet |       0.854 |    0.838 |      0.846 |             0.816 | 55.0%               | 0.45s                    | 5.33s                 |      36.6 |
| deer_night | ViBe + AT + ALR + MHI             |       0.891 |    0.779 |      0.831 |             0.788 | 59.9%               | 0.45s                    | 4.82s                 |      46.8 |
| deer_night | vibe_AT_ALR                       |       0.917 |    0.671 |      0.775 |             0.726 | 66.4%               | 0.45s                    | 4.00s                 |      35.2 |
| deer_night | ViBe + MotionNet (CNN latch)      |       0.888 |    0.622 |      0.732 |             0.719 | 67.9%               | 1.21s                    | 1.03s                 |      41.2 |
| deer_night | vibe_MHI_cnn                      |       0.888 |    0.622 |      0.732 |             0.719 | 67.9%               | 1.21s                    | 1.03s                 |      32.8 |
| deer_night | vibe_MHI                          |       0.924 |    0.543 |      0.684 |             0.633 | 73.1%               | 1.25s                    | 0.55s                 |      44.9 |
| deer_night | vibe_AT                           |       0.965 |    0.525 |      0.68  |             0.619 | 75.0%               | 1.18s                    | 0.23s                 |      35.2 |
| deer_night | ViBe (baseline)                   |       0.984 |    0.454 |      0.621 |             0.534 | 78.8%               | 1.25s                    | 0.09s                 |      37   |
| deer_night | vibe_ALR                          |       0.894 |    0.466 |      0.613 |             0.561 | 76.1%               | 1.32s                    | 0.68s                 |      36.7 |
| fox_day    | ViBe (baseline)                   |       0.992 |    0.975 |      0.984 |             0.975 | 40.0%               | 0.67s                    | 0.20s                 |      37.8 |
| fox_day    | vibe_AT                           |       0.992 |    0.975 |      0.984 |             0.975 | 40.0%               | 0.67s                    | 0.20s                 |      39.4 |
| fox_day    | vibe_ALR                          |       0.992 |    0.975 |      0.984 |             0.975 | 40.0%               | 0.67s                    | 0.20s                 |      35.6 |
| fox_day    | vibe_AT_ALR                       |       0.992 |    0.975 |      0.984 |             0.975 | 40.0%               | 0.67s                    | 0.20s                 |      33.3 |
| fox_day    | vibe_MHI                          |       0.992 |    0.975 |      0.984 |             0.975 | 40.0%               | 0.67s                    | 0.20s                 |      46.2 |
| fox_day    | ViBe + AT + ALR + MHI             |       0.992 |    0.975 |      0.984 |             0.975 | 40.0%               | 0.67s                    | 0.20s                 |      43.2 |
| fox_day    | vibe_MHI_cnn                      |       0.992 |    0.975 |      0.984 |             0.975 | 40.0%               | 0.67s                    | 0.20s                 |      35.7 |
| fox_day    | ViBe + AT + ALR + MHI + MotionNet |       0.992 |    0.975 |      0.984 |             0.975 | 40.0%               | 0.67s                    | 0.20s                 |      40.6 |
| fox_day    | ViBe + MotionNet (CNN latch)      |       0.992 |    0.968 |      0.98  |             0.627 | 40.4%               | 10.07s                   | 0.20s                 |      34.3 |
| fox_day    | vibe_AT_ALR_cnn                   |       0.992 |    0.956 |      0.974 |             0.057 | 41.2%               | 0.67s                    | 0.00s                 |      31.2 |
| fox_night  | ViBe + MotionNet (CNN latch)      |       0.918 |    0.831 |      0.872 |             0.684 | 48.6%               | 0.63s                    | 0.00s                 |      30   |
| fox_night  | vibe_AT                           |       0.845 |    0.886 |      0.865 |             0.766 | 40.5%               | 0.63s                    | 3.08s                 |      44.8 |
| fox_night  | vibe_ALR                          |       0.758 |    0.912 |      0.828 |             0.82  | 31.7%               | 0.42s                    | 9.08s                 |      37.2 |
| fox_night  | vibe_MHI_cnn                      |       0.737 |    0.856 |      0.792 |             0.776 | 34.1%               | 0.42s                    | 8.78s                 |      33.5 |
| fox_night  | vibe_AT_ALR                       |       0.649 |    0.99  |      0.784 |             0.99  | 13.4%               | 0.26s                    | 48.50s                |      37.1 |
| fox_night  | ViBe + AT + ALR + MHI             |       0.649 |    0.99  |      0.784 |             0.99  | 13.4%               | 0.26s                    | 48.50s                |      46.4 |
| fox_night  | ViBe + AT + ALR + MHI + MotionNet |       0.649 |    0.99  |      0.784 |             0.99  | 13.4%               | 0.26s                    | 48.50s                |      34.5 |
| fox_night  | vibe_AT_ALR_cnn                   |       0.649 |    0.989 |      0.784 |             0.824 | 13.5%               | 4.57s                    | 20.36s                |      29.1 |
| fox_night  | ViBe (baseline)                   |       0.948 |    0.659 |      0.777 |             0.544 | 60.5%               | 0.63s                    | 0.00s                 |      34.6 |
| fox_night  | vibe_MHI                          |       0.725 |    0.774 |      0.748 |             0.689 | 39.4%               | 0.42s                    | 7.88s                 |      47.1 |

---

## 6) Ablation insights (what each component is doing)

### Baseline: ViBe only
- Strong storage savings but weakest behavior retention on subtle-motion clips.
- Best used as a **storage-first** baseline.

### Add MHI (temporal persistence)
- Increases within-event continuity by bridging short motion gaps.
- Main downside: can create **long tails** if background motion persists (wind/grass/IR speckle).

### Add MotionNet (CNN presence latch)
- Improves semantic relevance (rejects some motion-only false triggers).
- Can increase trigger delay if run infrequently → tune CNN stride and rely on motion for start.

### Combined: ViBe + AT + ALR + MHI (+ optional MotionNet)
- Best behavior coverage and F1 in this benchmark set.
- Requires explicit tail control to avoid “record most of the night” behavior.

---

## 7) Plots to include in README (recommended)

If you only include **one** plot in `README.md`, use:

1) **Storage Saved vs Animal Coverage**  
   *This is the research story in one figure: storage savings vs behavior completeness.*

Then optionally add:
2) **Weighted F1 by Variant** (quick “overall” comparison)  
3) **Tail vs Coverage** (shows why tail control matters in high-recall configs)

---

## 8) How to reproduce these exact tables

From repo root:

```bash
# run one config
python scripts/eval_vibe_mhi_cnn.py --config configs/ablations/vibe_AT_ALR_MHI.yaml

# run all ablations and produce a merged summary CSV
python scripts/run_all_ablations.py --configs_dir configs/ablations --merged_csv results/benchmark_summary_all.csv

# run MOG2 baselines (optional)
python scripts/eval_mog2_cnn.py --config configs/ablations/mog2.yaml
python scripts/eval_mog2_cnn.py --config configs/ablations/mog2_cnn.yaml
```

Outputs:
- `results/benchmark_summary_*.csv`  
- `results/detection_log__*.csv`  
- (optional) `results/plots/*.png`

---

## 9) Next improvements (the honest roadmap)

**The biggest next lever:** reduce **extra tails** without losing recall/coverage.

Practical knobs already supported by the repo:
- Lower `mhi.tau_sec` (shorter memory)
- Increase stay threshold (harder to keep recording)
- Filter tiny blobs before MHI update (reduce grass/IR speckle persistence)
- Run CNN on weak-motion frames to “shut down” false tails (adaptive policy)

---

*Last updated:* 2025-12-19
