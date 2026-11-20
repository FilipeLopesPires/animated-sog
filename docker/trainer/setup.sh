#!/bin/bash
# Installs the flip-book training environment into /opt: the Open4DGS TRiGS trainer and timeline exporter
# (external, AGPL-3.0, cloned at fixed commits) and the `asog` package, in one venv at /opt/venv.
# Used by the Dockerfile, and runs as is on a stock CUDA 12.8 machine (e.g. a RunPod PyTorch pod):
#   TORCH_CUDA_ARCH_LIST=9.0 bash setup.sh        (compute capability of the GPU(s) it will run on)
# ASOG_SOURCE: path or pip URL of the asog package (default: this repository on GitHub).
set -euo pipefail
TRIGS_REF=ed7a84aff1aeccd18dfb48d2806e521255ccf9ef
CORE_REF=3cac7ba62d930e3ce56cb532bb60870e4ebf6db5
ASOG_SOURCE=${ASOG_SOURCE:-"animated-sog[flipbook] @ git+https://github.com/arrival-space/animated-sog#subdirectory=python"}
export UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/opt/python UV_PROJECT_ENVIRONMENT=/opt/venv
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:?set it to the GPU compute capability, e.g. 9.0}
export MAX_JOBS=${MAX_JOBS:-$(nproc)}

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
[ -d /opt/trigs ] || git clone https://github.com/ASU-NEM-Labs/open4dgs-trigs.git /opt/trigs
[ -d /opt/core ] || git clone https://github.com/ASU-NEM-Labs/open4dgs-core.git /opt/core
git -C /opt/trigs checkout -q "$TRIGS_REF"
git -C /opt/core checkout -q "$CORE_REF"

# the trainer's recorded recipe: torch 2.10 + cu128 first, then its lockfile with gsplat and pytorch3d built
# against that torch (no build isolation)
cd /opt/trigs
[ -x /opt/venv/bin/python ] || uv venv --python 3.11 /opt/venv
uv pip install --python /opt/venv/bin/python --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.10.0+cu128 torchvision==0.25.0+cu128
uv pip install --python /opt/venv/bin/python setuptools wheel ninja numpy
uv sync --locked --inexact --preview-features extra-build-dependencies \
    --no-build-isolation-package pytorch3d --no-build-isolation-package gsplat
uv pip install --python /opt/venv/bin/python "$ASOG_SOURCE"
rm -rf /root/.cache

cat > /etc/profile.d/asog.sh <<'EOF'
export PATH=/opt/venv/bin:$PATH ASOG_TRIGS_PYTHON=/opt/venv/bin/python ASOG_OPEN4DGS_CORE=/opt/core
EOF
/opt/venv/bin/python -c "import torch, gsplat; from pytorch3d.ops import knn_points; import asog; print('ok', torch.__version__, torch.version.cuda)"
