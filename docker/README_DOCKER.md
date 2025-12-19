# Docker (Edge-like Simulation) Guide

This `docker/` folder is designed to make your repo **reproducible** for:
- **Professors** (PhD review): "Can I run this and reproduce the table?"
- **Recruiters** (jobs): "Does it run cleanly in a container, with a one-command demo?"

## What this simulates (and what it doesn't)

✅ **Simulates**
- Memory pressure (e.g., 4GB like Jetson Orin Nano 4GB)
- CPU budget (limit CPU cores and thread counts)
- GPU execution inside Docker (your laptop GPU, not Jetson GPU)

❌ **Does not simulate**
- ARM CPU behavior (Jetson is ARM; your laptop is x86)
- Jetson-specific CUDA/TensorRT speed characteristics
- Jetson memory bandwidth / power / thermal throttling

Use this for **relative comparisons** (ablations, config ranking). Do a **final sanity run on real Jetson** before publication claims.

---

## Quick Start (recommended)

### 1) Build image (from repo root)
```bash
bash docker/scripts/build.sh
```

### 2) Run a command with edge-like limits (4GB RAM + CPU cap)
```bash
bash docker/scripts/run_edge_sim.sh wildlife-motion:latest "python -m src.eval.eval_motion --config configs/vibe_mhi.yaml"
```

> Replace the python command with your project entrypoint(s).
> If your project uses scripts like `run_all_ablations.py`, you can call that too.

---

## Using docker run directly (no scripts)

```bash
docker run --rm -it \
  --gpus all \
  --cpus="4" \
  --memory="4g" --memory-swap="4g" \
  --shm-size="1g" \
  -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 \
  -v "$PWD:/workspace" -w /workspace \
  wildlife-motion:latest bash
```

---

## Docker Compose (optional)

If you prefer `docker compose`:

```bash
docker compose -f docker/compose.yaml run --rm motion bash -lc "python -m src.eval.eval_motion --config configs/vibe_mhi.yaml"
```

---

## Troubleshooting

### GPU not visible
- Ensure NVIDIA Container Toolkit is installed.
- Test: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`

### OpenCV import error / missing libGL
We install `libgl1` and `libglib2.0-0` in the Dockerfile. Rebuild:
```bash
docker build -f docker/Dockerfile -t wildlife-motion:latest .
```

### Dataloader shared memory errors
Increase shm:
- `--shm-size=2g`

---

## Notes for repo finalization
- Keep **one canonical** Dockerfile (`docker/Dockerfile`)
- Keep **one canonical** run script (`docker/scripts/run_edge_sim.sh`)
- Keep benchmark outputs in `results/` (mounted from host)
