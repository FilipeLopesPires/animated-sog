# Flip-book training image: the Open4DGS TRiGS trainer and timeline exporter (external, AGPL-3.0, cloned from
# their public repositories at fixed commits) next to the `asog` package, for GPU machines such as RunPod pods.
# `asog flipbook ...` runs the trainer and exporter as subprocesses (ASOG_TRIGS_PYTHON, ASOG_OPEN4DGS_CORE).
# The install itself is setup.sh, which also sets up a stock CUDA machine without this image.
#
# Build context: the repository root.   docker build -f docker/trainer/Dockerfile -t asog-trainer .
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ninja-build curl ca-certificates libgl1 libglib2.0-0 ffmpeg \
        openssh-server rsync tmux \
    && rm -rf /var/lib/apt/lists/*

# Data-centre and workstation GPUs RunPod offers: A100 (8.0), A40/A6000/3090 (8.6), L40S/4090 (8.9),
# H100 (9.0), RTX 50-series (12.0)
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0" MAX_JOBS=4
COPY python /opt/asog-src
COPY docker/trainer/setup.sh /opt/setup.sh
RUN ASOG_SOURCE="/opt/asog-src[flipbook]" bash /opt/setup.sh

ENV PATH=/opt/venv/bin:$PATH \
    ASOG_TRIGS_PYTHON=/opt/venv/bin/python \
    ASOG_OPEN4DGS_CORE=/opt/core
COPY docker/trainer/start.sh /start.sh
RUN chmod +x /start.sh
WORKDIR /workspace
CMD ["/start.sh"]
