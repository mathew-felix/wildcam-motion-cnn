#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-wildlife-motion:latest}"
echo "[docker] Smoke test for ${IMAGE}"
docker run --rm \
  --gpus all \
  "${IMAGE}" bash -lc "python - <<'PY'
import sys
print('python', sys.version)
try:
    import torch
    print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
except Exception as e:
    print('torch import failed:', e)
try:
    import cv2
    print('opencv', cv2.__version__)
except Exception as e:
    print('opencv import failed:', e)
PY"
echo "[docker] Smoke test done."
