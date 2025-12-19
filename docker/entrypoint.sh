#!/usr/bin/env bash
set -euo pipefail

echo "==================== Container Info ===================="
echo "Date:   $(date)"
echo "User:   $(whoami)"
echo "PWD:    $(pwd)"
echo "Python: $(python -V)"
echo "--------------------------------------------------------"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU:"
  nvidia-smi || true
else
  echo "GPU: nvidia-smi not found (is --gpus all enabled?)"
fi
echo "--------------------------------------------------------"
python - <<'PY'
import os, platform
print("Platform:", platform.platform())
print("OMP_NUM_THREADS:", os.getenv("OMP_NUM_THREADS"))
print("MKL_NUM_THREADS:", os.getenv("MKL_NUM_THREADS"))
print("OPENBLAS_NUM_THREADS:", os.getenv("OPENBLAS_NUM_THREADS"))
PY
echo "========================================================"

exec "$@"
