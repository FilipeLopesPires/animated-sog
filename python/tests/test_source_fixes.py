"""Encoder fixes for Open4DGS exports of trainer checkpoints: quaternion order override, y-up -> y-down
rotation (z180) and the temporal threshold carried from the source."""

from __future__ import annotations

import json
import struct

import numpy as np

from asog import check, decode, encode, sh, trigs   # encode: the function
from asog.motion import temporal_weight


def _rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q.T
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)


def test_rotate_z180_is_a_rigid_rotation_of_the_whole_clip(synthetic_source):
    src = trigs.read(synthetic_source)
    rot = trigs.rotate_z180(src)
    R = np.diag([-1.0, -1.0, 1.0])
    dyn = ~src.static
    for k in np.linspace(-2, 2, 5):
        t = src.times + k * np.exp(src.log_durations)
        a = check.source_centres(src, t)[dyn]
        b = check.source_centres(rot, t)[dyn]
        assert np.abs(b - a @ R.T).max() < 1e-9
    assert np.abs(_rot(rot.quats) - R @ _rot(src.quats)).max() < 1e-9
    # SH: the rotated colour function seen from a rotated direction is the original one
    d = np.random.default_rng(0).normal(size=(16, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    c = src.shN.shape[1]
    f0 = np.einsum("vj,njc->nvc", sh.basis(d, c), src.shN)
    f1 = np.einsum("vj,njc->nvc", sh.basis(d @ R.T, c), rot.shN)
    assert np.abs(f1 - f0).max() < 1e-5


def test_quat_order_override(synthetic_source):
    a = trigs.read(synthetic_source)                       # labelled xyzw
    b = trigs.read(synthetic_source, quat_order="wxyz")
    raw = a.quats[:, [1, 2, 3, 0]]                         # what is stored, as read verbatim
    assert np.allclose(b.quats, raw / np.linalg.norm(raw, axis=1, keepdims=True))


def _with_semantics(src_path, dst_path, extra: dict) -> None:
    data = bytearray(src_path.read_bytes())
    mo, ml, po, pl = struct.unpack("<QQQQ", data[16:48])
    man = json.loads(data[mo:mo + ml])
    man["semantics"].update(extra)
    body = json.dumps(man).encode()
    assert len(body) + mo <= po, "manifest grew past the payload offset"
    data[mo:po] = body + b" " * (po - mo - len(body))
    data[16:48] = struct.pack("<QQQQ", mo, len(body) + (po - mo - len(body)), po, pl)
    dst_path.write_bytes(bytes(data))


def test_threshold_and_transform_round_trip(tmp_path, synthetic_source):
    src = tmp_path / "th.trigsbin"
    _with_semantics(synthetic_source, src, {"temporalCullThreshold": 0.01,
                                            "temporalOpacityMode": "smooth_cull_renormalized"})
    out = tmp_path / "th.asog"
    encode(src, out, transform="z180", quat_order="wxyz")
    a = decode.read(out)
    assert a.manifest["motion"]["threshold"] == 0.01
    assert abs(a.manifest["motion"]["sm"] - np.sqrt(-2 * np.log(0.01))) < 1e-3
    assert a.manifest["source"]["transform"] == "z180" and a.manifest["source"]["quaternionOrder"] == "wxyz"
    # a splat 2.8 sigma from its centre is visible at 0.01 and was culled at 0.05
    assert temporal_weight(np.array(2.8), np.array(1.0), 0.01) > 0 == temporal_weight(np.array(2.8), np.array(1.0))
    for i in np.nonzero(~a.static)[0]:   # a splat whose 2.8-sigma point lies inside its clip's active range
        t = float(a.mu[i] + 2.8 * a.sigma[i])
        fade, local = a.clip_weights(t)
        if fade[i] > 0 and abs(local[i] - t) < 1e-12:
            assert a.evaluate(t).weights[i] > 0
            break
    else:
        raise AssertionError("no splat to test the threshold on")
    # check re-reads the source the same way: errors stay at the quantisation level
    r = check.report(src, out)
    assert r["motion_p99"] < 50, r
