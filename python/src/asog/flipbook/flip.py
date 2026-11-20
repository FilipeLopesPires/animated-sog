"""Rotate a TRiGS checkpoint 180 degrees about z: from the trainer's y up to the y-down TRiGS / COLMAP
orientation that Open4DGS clips have and that players expect.

Positions, anchors and every se(3) twist vector (`zeta_*`) rotate as vectors; quaternions are
pre-multiplied by the z rotation; a real SH coefficient of order m picks up the factor (-1)^|m|.

The quaternions are written **xyzw**. The trainer stores wxyz, but Open4DGS's `make-trigs-timeline.py`
(version 2026-09-16.runtime-bin-v2) copies the checkpoint's quaternions verbatim and labels them
`quaternionOrder: "xyzw"`; exported unflipped, every splat renders mis-rotated ("furry"). The flipped
checkpoint is therefore meant for the exporter only, not for further training.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

SH_ABS_ORDER = [1, 0, 1, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3]   # |m| of the degree 1..3 coefficients


def flip_checkpoint(src: str | Path, dst: str | Path) -> None:
    """Write `src` rotated 180 degrees about z (quaternions xyzw) to `dst`."""
    import torch

    ck = torch.load(src, map_location="cpu", weights_only=True)
    p = ck["params"]
    flip = torch.tensor([-1.0, -1.0, 1.0])
    for k in list(p):
        if k in ("means", "anchors") or k.startswith("zeta_"):
            p[k] = p[k] * flip
    w, x, y, z = p["quats"].unbind(1)
    p["quats"] = torch.stack([-y, x, w, -z], 1)   # q_z180 * q = (-z, -y, x, w) in wxyz, written xyzw
    if "shN" in p:
        m = torch.tensor(SH_ABS_ORDER[: p["shN"].shape[1]])
        p["shN"] = p["shN"] * torch.where(m % 2 == 1, -1.0, 1.0)[None, :, None]
    # the checkpoint also keeps copies of the motion / temporal tensors: keep them equal to params
    for group in ("temporal_params", "motion_params"):
        for k, v in ck.get(group, {}).items():
            ck[group][k] = p[k].reshape(v.shape)
    torch.save(ck, dst)
    log.info(f"flipped {src} -> {dst} ({len(p['means'])} splats)")
