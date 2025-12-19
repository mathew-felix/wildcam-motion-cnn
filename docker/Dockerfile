# docker/Dockerfile
# GPU-enabled runtime image for wildlife-motion-cnn style projects.
# Build from repo root:
#   docker build -f docker/Dockerfile -t wildlife-motion:latest .
#
# Notes:
# - This DOES NOT emulate Jetson ARM GPU/CPU. It approximates *resource pressure*
#   using Docker limits (memory/cpu/thread caps) + your laptop GPU.
# - For true Jetson validation, run at least a short final test on real hardware.

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps commonly needed for OpenCV + video IO
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    ffmpeg \
    libgl1 libglib2.0-0 \
    libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy repo into image
COPY . /workspace

# Install Python deps:
# 1) If repo has requirements.txt, prefer that
# 2) Else, fall back to docker/requirements.lock.txt (ships with this bundle)
RUN python -m pip install --upgrade pip && \
    if [ -f requirements.txt ]; then \
        pip install -r requirements.txt; \
    elif [ -f docker/requirements.lock.txt ]; then \
        pip install -r docker/requirements.lock.txt; \
    else \
        pip install numpy pandas pyyaml opencv-python; \
    fi

# Ensure scripts are executable
RUN chmod +x /workspace/docker/entrypoint.sh /workspace/docker/scripts/*.sh || true

# Default entrypoint prints environment + then runs CMD
ENTRYPOINT ["/workspace/docker/entrypoint.sh"]
CMD ["bash"]
