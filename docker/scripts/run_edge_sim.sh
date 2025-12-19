#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-wildlife-motion:latest}"
CMD="${2:-bash}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

docker run --rm -it \
  --gpus all \
  --cpus="4" \
  --memory="4g" --memory-swap="4g" \
  --shm-size="1g" \
  -e OMP_NUM_THREADS=2 \
  -e MKL_NUM_THREADS=2 \
  -e OPENBLAS_NUM_THREADS=2 \
  -v "${PROJECT_ROOT}:/workspace" \
  -w /workspace \
  "${IMAGE}" bash -lc "${CMD}"
