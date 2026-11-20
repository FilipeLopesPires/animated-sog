"""A small synthetic TRiGS runtime file (`trigs.runtime` v1) with known motion, for the tests.

Two clips of one second each, back to back on a two-second timeline; every clip has a static group and a
dynamic group. The dynamic splats of clip 0 have a constant twist (all Bezier control points zero, only
`zeta_base_*`) with the translation along the rotation axis, so their motion has a closed form; clip 1's
splats have curved twists. Quaternions are stored xyzw (and labelled so), the half-precision tensors are
float16 as in Open4DGS exports.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

FPS = 30
N_STATIC = 400
N_DYNAMIC = 3000
ZETA = ["zeta_base_r", "zeta_base_t", "zeta_c0_r", "zeta_c0_t", "zeta_c1_r", "zeta_c1_t", "zeta_c2_r", "zeta_c2_t"]


def _group(rng: np.random.Generator, n: int, dynamic: bool, constant_twist: bool) -> dict[str, np.ndarray]:
    means = rng.uniform(-0.6, 0.6, (n, 3)) + [0, -0.9, 0]
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    g = {
        "means": means.astype(np.float32),
        "scales": rng.uniform(-6, -3, (n, 3)).astype(np.float16),
        "quats": q.astype(np.float32),                       # xyzw
        "opacities": rng.normal(1, 2, n).astype(np.float16),  # logit
        "sh0": rng.normal(0, 0.8, (n, 1, 3)).astype(np.float16),
        "shN": (rng.normal(0, 0.1, (n, 15, 3)) * rng.uniform(0, 1, (n, 1, 1))).astype(np.float16),
    }
    if dynamic:
        axis = rng.normal(size=(n, 3))
        axis /= np.linalg.norm(axis, axis=1, keepdims=True)
        speed = rng.uniform(0.2, 2.0, (n, 1))   # rad/s
        g["times"] = rng.uniform(0, 1, n).astype(np.float32)
        g["log_durations"] = np.log(rng.uniform(0.03, 0.6, n)).astype(np.float32)
        g["anchors"] = (means + rng.normal(0, 0.1, (n, 3))).astype(np.float32)
        g["zeta_base_r"] = (axis * speed).astype(np.float32)
        g["zeta_base_t"] = (axis * rng.uniform(-0.3, 0.3, (n, 1)) + rng.normal(0, 0.05, (n, 3))).astype(np.float32)
        for k in range(3):
            for kind in "rt":
                z = np.zeros((n, 3)) if constant_twist else rng.normal(0, 0.3 if kind == "r" else 0.1, (n, 3))
                g[f"zeta_c{k}_{kind}"] = z.astype(np.float32)
    return g


def make_trigsbin(path: Path, seed: int = 0) -> dict:
    """Write the synthetic file; returns its manifest."""
    rng = np.random.default_rng(seed)
    payload = bytearray()

    def tensors(group: dict[str, np.ndarray]) -> dict:
        out = {}
        for name, a in group.items():
            payload.extend(bytes(-len(payload) % 256))
            out[name] = dict(byteOffset=len(payload), byteLength=a.nbytes, dtype=str(a.dtype), shape=list(a.shape))
            payload.extend(a.tobytes())
        return out

    clips = []
    for c in range(2):
        f0 = c * FPS
        static = _group(rng, N_STATIC // 2, False, False)
        dynamic = _group(rng, N_DYNAMIC // 2, True, constant_twist=c == 0)
        clips.append({
            "id": f"clip_{c:04d}", "index": c,
            "global": {"frameStart": f0, "frameEndExclusive": f0 + FPS, "activeTimeStart": c * 1.0,
                       "activeTimeEnd": c + 1.0, "coreTimeStart": c * 1.0, "coreTimeEnd": c + 1.0},
            "local": {"frameStart": 0, "frameEndExclusive": FPS, "timeStart": 0.0, "timeEnd": 1.0},
            "render": {"fadeInFrames": 0, "fadeOutFrames": 0, "fadeCurve": "smoothstep"},
            "static": {"count": len(static["means"]), "tensors": tensors(static)},
            "dynamic": {"count": len(dynamic["means"]), "tensors": tensors(dynamic)},
        })
    manifest = {
        "format": "trigs.runtime", "version": 1,
        "semantics": {"quaternionOrder": "xyzw", "timeUnit": "seconds"},
        "timeline": {"frameRate": float(FPS), "totalFrames": 2 * FPS, "durationSeconds": 2.0},
        "clips": clips,
        "static": {"count": 0, "tensors": {}},
    }
    mj = json.dumps(manifest).encode()
    payload_offset = 64 + len(mj) + (-(64 + len(mj)) % 256)
    head = b"TRIGSBIN" + bytes(8) + struct.pack("<QQQQ", 64, len(mj), payload_offset, len(payload))
    with open(path, "wb") as f:
        f.write(head + bytes(64 - len(head)) + mj + bytes(payload_offset - 64 - len(mj)) + payload)
    return manifest
