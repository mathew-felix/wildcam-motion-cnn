# Wildlife Motion Gate (ViBe + MHI + MotionNet)
Edge‑friendly wildlife recording gate focused on **behavior coverage** (keep the animal’s full activity timeline), not just “event happened”.

This repo implements a **motion‑triggered clip gate** designed for **Jetson‑class / edge‑like constraints** (see `README_DOCKER.md`).  
**Core idea:** **START strict, STAY smart** — start clips with a strict motion signal, then keep clips alive through brief motion gaps using temporal persistence and optional semantic confirmation.

---

## Quickstart (reproduce results)
> The benchmark videos are **not included** in this repo. See **Data** below.

### Option A — Local (fast iteration)
```bash
pip install -r requirements.txt
python scripts/run_all_ablations.py
```

### Option B — Docker edge simulation (recommended for “edge-like” numbers)
Use the scripts in `docker/scripts/` for a consistent, shareable run:

```bash
# from repo root
bash docker/scripts/build.sh

# run with edge-like limits (4GB RAM + CPU cap)
bash docker/scripts/run_edge_sim.sh wildlife-motion:latest "python scripts/run_all_ablations.py"
```

> This simulates **resource limits** (CPU + memory) on your laptop, not ARM/TensorRT performance.
> See `README_DOCKER.md` for details and troubleshooting.


**Outputs (expected):**
- `results/benchmark_summary_all.csv`
- `results/plots/*.png`
- `results/detection_log__*.csv`

---

## 30‑second overview (what the system does)

```
Frames
  ├─► ViBe (BG subtraction) ─► motion mask ─► START trigger (strict)
  ├─► MHI (motion memory) ────────────────► STAY trigger (bridge gaps)
  └─► MotionNet (optional CNN latch) ─────► semantic confirm + keep-alive
              (distilled MobileNetV3-Small on motion ROIs; throttled for edge)

All signals → ClipGate state machine (pre-roll + post-roll) → keep/drop frames
```

### Why this matters
Pure motion gating can drop subtle behavior (standing, slow head movement).  
This gate targets **behavior preservation** while still reducing unnecessary recording.

## Demo
If you add a short overlay GIF (e.g., motion mask / kept frames / gate state), place it at `results/plots/demo_overlay.gif` and embed it here:

<!-- Uncomment after adding the file:
![Demo overlay](results/plots/demo_overlay.gif)
-->

---

## What’s in the repo

### Motion pipeline variants
To make experiments readable and reproducible, variants follow a consistent naming scheme:

- **ViBe**: base sample‑based background subtraction
- **AT**: adaptive thresholding
- **ALR**: adaptive learning rate
- **MHI**: motion history image (temporal persistence)
- **MotionNet**: CNN latch (binary animal vs background on motion crops)

> In the results table, some variants are shown with a friendly name and some with config-like names.
> When finalizing the repo, prefer **one naming convention** (recommended: the friendly names).

---

## Metrics (what “good” means here)

**Primary objective:** maximize **Animal_Coverage** (keep the animal’s timeline).  
**Practical constraint:** avoid runaway recording (Tail / extra recording).

- **Precision / Recall / F1:** frame‑level keep/drop classification on the scored range (higher is better)
- **Animal_Coverage (0–1):** fraction of GT animal frames kept (higher is better)
- **StorageSaved (%):** estimated storage reduction vs recording everything (higher is better)
- **Delay (s):** time to start recording after animal appears (lower is better)
- **Tail (s):** extra recording after animal leaves (lower is better)
- **FPS:** throughput of the evaluation pipeline (higher is better)

Small glossary: **TP** = animal present & frame kept; **FP** = no animal & frame kept (wasted storage); **FN** = animal present & frame dropped (behavior loss). See `docs/results.md` for the full evaluation write‑up.

> Note: Very high FPS numbers for simple baselines (e.g., MOG2) are expected because the baseline does far less work
> than pipelines that include ROI extraction + CNN inference + logging.

---

## Results (edge‑simulated constraints)

### Evaluation protocol (summary)
- **Data:** 4 benchmark wildlife clips (`deer_day`, `deer_night`, `fox_day`, `fox_night`) described in `data/benchmarks/manifest.json`.
- **Labels:** frame‑level animal presence GT for the scored range (see `docs/results.md`).
- **Scoring:** each frame is classified as **kept** or **dropped** by the gate.
- **Metrics:** TP/FP/FN computed on the scored range; Coverage = fraction of GT animal frames kept.

### Edge‑like simulation (what it simulates)
- ✅ **Simulates:** memory pressure (e.g., 4GB), CPU budget / thread caps, and GPU execution inside Docker on your laptop.
- ❌ **Does not simulate:** Jetson ARM CPU behavior, Jetson-specific CUDA/TensorRT speed, or Jetson power/thermal throttling.
- Use this for **relative comparisons** (ablations, config ranking). Do a final sanity run on real Jetson before publication-level claims.

Aggregated across 4 benchmark videos: `deer_day`, `deer_night`, `fox_day`, `fox_night`.

For dataset/metric definitions, per‑video tables, and edge‑simulation details, see **`docs/results.md`** (paper‑style evaluation).  
See `README_DOCKER.md` for how the edge simulation is configured.

### Experimental setup (for the numbers below)
- **Host (example):** Dell G15 laptop — i9‑12900H, RTX 3070 Ti, 32GB RAM
- **Docker edge-like limits:** `--cpus="4"`, `--memory="4g"`, `--memory-swap="4g"`, `--shm-size="1g"`
- **Thread caps:** `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`

> Interpret FPS as **Docker‑constrained throughput on the host**, not native Jetson performance (see `README_DOCKER.md`).

### Leaderboard (aggregate)
| Variant                           |   Precision |   Recall |    F1 | Coverage   | StorageSaved   | Delay   | Tail   |   FPS |
|:----------------------------------|------------:|---------:|------:|:-----------|:---------------|:--------|:-------|------:|
| ViBe + AT + ALR + MHI + MotionNet |       0.736 |    0.965 | 0.835 | 0.957      | 26.9%          | 0.23s   | 37.62s |  36.5 |
| ViBe + AT + ALR + MotionNet       |       0.736 |    0.963 | 0.834 | 0.801      | 27.1%          | 2.58s   | 9.24s  |  29.8 |
| ViBe + AT + ALR + MHI             |       0.742 |    0.938 | 0.828 | 0.939      | 29.5%          | 0.23s   | 37.23s |  46.2 |
| ViBe + MotionNet                  |       0.864 |    0.795 | 0.828 | 0.741      | 48.7%          | 1.73s   | 4.70s  |  35.3 |
| ViBe + MHI + MotionNet            |       0.809 |    0.806 | 0.807 | 0.808      | 44.4%          | 0.41s   | 7.22s  |  32.3 |
| ViBe + AT + ALR                   |       0.729 |    0.864 | 0.791 | 0.855      | 34.0%          | 2.32s   | 37.08s |  36.6 |
| ViBe + MHI                        |       0.806 |    0.738 | 0.771 | 0.717      | 48.9%          | 0.65s   | 6.88s  |  43.1 |
| ViBe + AT                         |       0.782 |    0.726 | 0.753 | 0.728      | 48.2%          | 0.66s   | 9.24s  |  35.1 |
| ViBe + ALR                        |       0.742 |    0.629 | 0.681 | 0.690      | 52.7%          | 2.65s   | 9.58s  |  37.0 |
| ViBe (baseline)                   |       0.858 |    0.521 | 0.649 | 0.554      | 66.1%          | 2.71s   | 2.18s  |  34.6 |
| MOG2 (baseline)                   |       0.567 |    1.000 | 0.724 | 1.000      | 0.0%           | 0.00s   | 60.75s | 370.6 |
| MOG2 + MotionNet                  |       0.858 |    0.476 | 0.612 | 0.460      | 68.6%          | 3.32s   | 1.79s  | 362.2 |

### Quick takeaways
- **Best behavior coverage:** `ViBe + AT + ALR + MHI + MotionNet` reaches **0.957 Coverage** and **0.965 Recall**, but pays a **Tail** cost (keeps recording longer).
- **Best storage saver:** `ViBe (baseline)` saves the most storage (**66.1%**) but loses behavior (**0.554 Coverage**).
- **Why MOG2 looks “fast”:** it’s a lightweight baseline (no ROI cropping/CNN/logging), so FPS is not directly comparable to the full pipeline.

### Key plots
Use Markdown image links for reliable GitHub rendering:

![Storage vs Coverage](results/plots/tradeoff_storage_vs_coverage.png)

![Tail vs Coverage](results/plots/tail_vs_coverage.png)

![Weighted F1 by Variant](results/plots/weighted_f1_by_variant.png)

---

## Recommended configs (how to pick)
- **Default behavior‑preserving (good tradeoff):** **ViBe + AT + ALR + MHI**
- **Max recall/coverage (expect tail cost):** **ViBe + AT + ALR + MHI + MotionNet**
- **Storage‑first baseline:** **ViBe (baseline)**
- **Precision‑oriented (less over‑recording, later starts):** **ViBe + MotionNet**

---

## Tuning (practical knobs)
This system optimizes behavior coverage, but the cost is often **Tail**.

- **Tail = time we keep recording after the animal is gone.**
- Tail increases because MHI / stay logic intentionally bridges motion gaps (good for subtle behavior),
  but dynamic backgrounds (wind/grass/IR flicker) can keep the gate open longer.

**To reduce Tail (save storage/battery):**
- Decrease `mhi.tau_sec`
- Increase “stay” threshold (require stronger evidence to keep recording)
- Require **K consecutive quiet frames** to close

**To increase coverage/recall (keep more behavior):**
- Increase `mhi.tau_sec`
- Lower the “stay” threshold
- Run MotionNet more frequently during weak-motion periods

---

## Limitations / failure cases
- **Dynamic backgrounds** (wind/grass, water ripples, IR flicker) can keep MHI active and increase Tail.
- **Low-motion behavior** (standing still, slow head movement) is the core target; extreme low-motion + heavy occlusion can still cause missed frames.
- **MotionNet domain shift:** night/IR clips, motion blur, rain/snow, and camera noise can reduce CNN reliability without additional fine‑tuning.
- **ROI quality depends on motion mask:** if motion segmentation misses parts of the animal, the crop can be partial and the CNN signal may be weaker.
- **This is a gate, not a detector:** it decides what to record/keep; it is not meant to output tight boxes or species labels.

## Distilled ROI classifier (Teacher–Student)
To reduce false triggers while staying lightweight, we train a compact **MobileNetV3‑Small** binary classifier that predicts **animal vs background** on motion ROIs.

- **Teacher:** `YOLO11x-cls`
- **Student:** `MobileNetV3-Small` (deployed as MotionNet)
- **Training data:** ROI crops from **Caltech Camera Traps (CCT/CCT20)** with `train/val/test` splits
- **Leakage control:** benchmark evaluation videos are held out and not used for training/validation

---

## Data
This repo does not ship benchmark videos or CCT data.

Expected paths:
- Benchmark videos: `data/benchmark_videos/*.mp4`
- Benchmark manifest: `data/benchmarks/manifest.json`

---

## Repo layout
```
configs/
  ablations/            # YAML configs for each variant
  standard/
scripts/
  eval_vibe_mhi_cnn.py  # main evaluator (logs + summary)
  eval_mog2_cnn.py      # MOG2 baseline evaluator
  run_all_ablations.py  # run + merge results
src/
  vibe.py               # ViBe + AT/ALR
  vibe_mhi.py           # MHI + clip gate
  vibe_mhi_cnn.py       # CNN latch + hybrid gate (ViBe path)
  mog2_cnn.py           # MOG2 + CNN latch (baseline)
results/
  benchmark_summary_all.csv
  detection_log__*.csv
  plots/
docs/
  results.md            # paper-style evaluation writeup
README_DOCKER.md        # edge simulation details
```

---

## Reference (AT/ALR motivation)
This project is inspired by recent adaptive motion detection work that improves robustness under dynamic backgrounds and illumination changes.
- Mpofu, J.B., Li, C., Gao, X., Su, X. Adaptive Motion Detection for Enhanced Video Surveillance. IEEE IMCEC 2024.


---

## Citation
```bibtex
@software{wildlife-motion-gate,
  title        = {Wildlife Motion Gate: ViBe + MHI + MotionNet for Edge Camera Traps},
  author       = {Felix Mathew},
  year         = {2025},
  note         = {GitHub repository},
}
```

