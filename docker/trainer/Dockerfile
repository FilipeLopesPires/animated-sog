# Flip-book training image: the Open4DGS TRiGS trainer and timeline exporter (external, AGPL-3.0, cloned from
# their public repositories at fixed commits) next to the `asog` package, for GPU machines such as RunPod pods.
# `asog flipbook ...` runs the trainer and exporter as subprocesses (ASOG_TRIGS_PYTHON, ASOG_OPEN4DGS_CORE).
#
# Build context: the repository root.   docker build -f docker/trainer/Dockerfile -t asog-trainer .
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ninja-build curl ca-certificates libgl1 libglib2.0-0 ffmpeg \
        openssh-server rsync tmux \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# Data-centre and workstation GPUs RunPod offers: A100 (8.0), A40/A6000/3090 (8.6), L40S/4090 (8.9),
# H100 (9.0), RTX 50-series (12.0)
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0" \
    MAX_JOBS=4 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv

ARG TRIGS_REF=ed7a84aff1aeccd18dfb48d2806e521255ccf9ef
ARG CORE_REF=3cac7ba62d930e3ce56cb532bb60870e4ebf6db5
RUN git clone https://github.com/ASU-NEM-Labs/open4dgs-trigs.git /opt/trigs && git -C /opt/trigs checkout "$TRIGS_REF" \
 && git clone https://github.com/ASU-NEM-Labs/open4dgs-core.git /opt/core && git -C /opt/core checkout "$CORE_REF"

# The trainer's recorded recipe: torch 2.10 + cu128 first, then its lockfile with gsplat and pytorch3d built
# against that torch (no build isolation)
WORKDIR /opt/trigs
RUN uv venv --python 3.11 /opt/venv \
 && uv pip install --python /opt/venv/bin/python --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.10.0+cu128 torchvision==0.25.0+cu128 \
 && uv pip install --python /opt/venv/bin/python setuptools wheel ninja numpy \
 && uv sync --locked --inexact --preview-features extra-build-dependencies \
        --no-build-isolation-package pytorch3d --no-build-isolation-package gsplat \
 && rm -rf /root/.cache

COPY python /opt/asog
RUN uv pip install --python /opt/venv/bin/python "/opt/asog[flipbook]" && rm -rf /root/.cache

ENV PATH=/opt/venv/bin:$PATH \
    ASOG_TRIGS_PYTHON=/opt/venv/bin/python \
    ASOG_OPEN4DGS_CORE=/opt/core
COPY docker/trainer/start.sh /start.sh
RUN chmod +x /start.sh
WORKDIR /workspace
CMD ["/start.sh"]
