#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-wildlife-motion:latest}"
echo "[docker] Building image: ${TAG}"
docker build -f docker/Dockerfile -t "${TAG}" .
echo "[docker] Done."
