#!/bin/bash
# Container start: SSH for the key in $PUBLIC_KEY (RunPod passes the account's key this way), then idle.
# Work runs over SSH: copy the PLY frames in, run `asog flipbook run ...`, copy the .asog out.
set -e
if [ -n "$PUBLIC_KEY" ]; then
    mkdir -p /root/.ssh && chmod 700 /root/.ssh
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
    ssh-keygen -A >/dev/null
    mkdir -p /run/sshd && /usr/sbin/sshd
fi
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv || true
python -c "import torch, gsplat; from pytorch3d.ops import knn_points; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
exec sleep infinity
