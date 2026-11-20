"""Encode the synthetic TRiGS file, decode it, and hold every attribute to the spec's dequantisation."""

import numpy as np
import pytest
import synthetic

import asog
from asog import check, container, trigs
from asog.encode import ORDER_CHUNK_SECONDS, importance, splat_order
from asog.motion import quat_mul, temporal_weight


@pytest.fixture(scope="module")
def pair(synthetic_source, synthetic_asog):
    a = asog.read(synthetic_asog)
    src = trigs.read(synthetic_source)
    return src.take(splat_order(src, a.count)), a


def test_prune_keeps_statics_and_most_important(synthetic_source, pair):
    src_all = trigs.read(synthetic_source)
    s, a = pair
    assert a.count == synthetic.N_STATIC + synthetic.N_DYNAMIC - round(0.3 * synthetic.N_DYNAMIC)
    assert s.static.sum() == synthetic.N_STATIC
    np.testing.assert_array_equal(a.static, s.static)
    np.testing.assert_array_equal(a.clip, s.clip)
    imp = importance(src_all)
    dropped = np.setdiff1d(np.arange(src_all.count), splat_order(src_all, a.count))
    assert imp[dropped].max() <= imp[splat_order(src_all, a.count)].min()


def test_order_statics_then_birth_chunk(pair):
    s, a = pair
    chunk = np.where(a.static, -1, np.floor(s.times / ORDER_CHUNK_SECONDS))
    assert np.all(np.diff(chunk) >= 0)


def test_static_attributes_within_quantisation(pair):
    s, a = pair
    M = a.manifest
    # position: half a 14-bit step of the log mapping, through the derivative of the inverse
    lo, hi = np.array(M["means"]["mins"]), np.array(M["means"]["maxs"])
    v = np.sign(s.means) * np.log1p(np.abs(s.means))
    bound = 0.5 * 4 / 65535 * (hi - lo) * np.exp(np.abs(v)) * 1.001 + 1e-9
    assert np.all(np.abs(a.means - s.means) <= bound)
    # rotation: each stored component within half an 8-bit step of sqrt(2)
    ang = 2 * np.degrees(np.arccos(np.clip(np.abs((a.quats * s.quats).sum(1)), 0, 1)))
    assert ang.max() < 1.5
    np.testing.assert_allclose(np.linalg.norm(a.quats, axis=1), 1, atol=1e-12)
    assert np.abs(a.opacity - s.opacity).max() <= 0.5 / 255 + 1e-9
    # codebooks: every value maps to its nearest entry
    for decoded, source, key in ((a.log_scales, s.log_scales, "scales"), (a.sh0, s.sh0, "sh0")):
        cb = np.array(M[key]["codebook"])
        assert np.all(np.diff(cb) >= 0) and len(cb) == 256
        nearest = np.abs(source.reshape(-1, 1) - cb[None]).min(1)
        np.testing.assert_allclose(np.abs(decoded - source).ravel(), nearest, atol=1e-12)
    # SH palette: better than dropping the SH
    assert np.abs(a.shN - s.shN).mean() < 0.9 * np.abs(s.shN).mean()
    assert M["shN"]["count"] == 128 and M["shN"]["coeffs"] == 15


def test_timing_and_anchor_within_quantisation(pair):
    s, a = pair
    M = a.manifest
    d = ~a.static
    assert np.abs(a.mu[d] - s.times[d]).max() <= 0.5 / M["mu"]["scale"] + 1e-7
    ls = M["logsigma"]
    half = 0.5 * (ls["max"] - ls["min"]) / (2 ** ls["bits"] - 1)
    assert np.abs(np.log(a.sigma[d]) - s.log_durations[d]).max() <= half + 1e-6
    assert np.abs(a.anchor[d] - s.anchors[d]).max() <= 0.5 * M["motion"]["qa"] / 1000 + 1e-7
    assert not a.anchor[~d].any() and not a.omega[:, ~d].any() and not a.g[:, ~d].any()


def test_motion_error_scales_with_qp(tmp_path, synthetic_source, synthetic_asog):
    coarse = check.report(synthetic_source, synthetic_asog)
    fine_path = tmp_path / "fine.asog"
    asog.encode(synthetic_source, fine_path, qp=0.25, qa=0.5, prune=0)
    fine = check.report(synthetic_source, fine_path)
    # measured: p99 is ~5x qp (anchor, sigma and time steps add to the twist step) at both settings
    assert coarse["motion_p99"] < 8 * 2.0
    assert fine["motion_p99"] < 8 * 0.25
    assert fine["motion_p50"] < coarse["motion_p50"] / 4


def _axis_angle_quat(axis_angle: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(axis_angle, axis=1, keepdims=True)
    return np.concatenate([np.cos(th / 2), axis_angle / th * np.sin(th / 2)], 1)


def test_evaluate_matches_closed_form_motion(tmp_path, synthetic_source):
    """Clip 0's splats spin at a constant rate about a fixed axis through the anchor and slide along it."""
    path = tmp_path / "fine.asog"
    asog.encode(synthetic_source, path, qp=0.25, qa=0.5, prune=0, tbits=14, lsbits=12)
    a = asog.read(path)
    src = trigs.read(synthetic_source)
    s = src.take(splat_order(src, a.count))
    t = 0.5
    frame = a.evaluate(t)
    m = ~s.static & (s.clip == 0) & (frame.weights > 0)
    assert m.sum() > 500
    om, nu, dt = s.omega[0, m], s.nu[0, m], t - s.times[m]
    axis = om / np.linalg.norm(om, axis=1, keepdims=True)
    w = om * dt[:, None]
    th = np.linalg.norm(w, axis=1, keepdims=True)
    k = w / th
    r = s.means[m] - s.anchors[m]   # rotating about the anchor: any point on the axis line gives the same motion
    rotated = r * np.cos(th) + np.cross(k, r) * np.sin(th) + k * (k * r).sum(1, keepdims=True) * (1 - np.cos(th))
    truth = rotated + s.anchors[m] + (nu * axis).sum(1, keepdims=True) * axis * dt[:, None]
    err = np.linalg.norm(frame.positions[m] - truth, axis=1)
    assert np.percentile(err, 99) < 2e-3 and err.max() < 5e-3
    q_truth = quat_mul(_axis_angle_quat(w), s.quats[m])
    ang = 2 * np.degrees(np.arccos(np.clip(np.abs((frame.rotations[m] * q_truth).sum(1)), 0, 1)))
    assert np.percentile(ang, 99) < 1.5
    np.testing.assert_allclose(frame.weights[m], temporal_weight(dt, np.exp(s.log_durations[m])), atol=0.02)


def test_evaluate_clip_activity(pair):
    _, a = pair
    for t, live in ((0.5, 0), (1.5, 1)):
        f = a.evaluate(t)
        assert not f.weights[a.clip != live].any()
        assert np.all(f.weights[(a.clip == live) & a.static] == 1)
        still = a.static | (f.weights == 0)
        np.testing.assert_array_equal(f.positions[still], a.means[still])
    # clip 1 runs on local time t - 1 s: its splat with the earliest centre time is at its peak there
    f = a.evaluate(1.0 + a.mu[~a.static & (a.clip == 1)].min())
    i = np.nonzero(~a.static & (a.clip == 1))[0][np.argmin(a.mu[~a.static & (a.clip == 1)])]
    assert f.weights[i] == pytest.approx(1.0)


def test_manifest(synthetic_source, synthetic_asog):
    man, _, audio = container.read(synthetic_asog)
    assert audio is None and "audio" not in man
    assert [s["name"] for s in man["streams"]] == ["means", "quats", "quatmode", "scales", "sh0", "opacity", "shlabels",
                                                    "shpalette", "mu", "logsigma", "anchor", "omega", "g", "flags"]
    assert man["source"] == {"format": "trigs.runtime", "version": 1}
    assert man["timeline"]["durationSeconds"] == 2.0
    assert all("static" not in c and "dynamic" not in c for c in man["clips"])
    assert man["motion"] == dict(qp=2.0, qa=8.0, sm=2.45, midO=3.0, midG=2.0, mref_samples=9, r_min=0.02)


def test_audio_is_embedded(tmp_path, synthetic_source):
    sound = tmp_path / "loop.mp3"
    sound.write_bytes(b"ID3" + bytes(range(256)) * 10)
    path = tmp_path / "with-audio.asog"
    asog.encode(synthetic_source, path, audio=sound)
    a = asog.read(path)
    assert a.audio == sound.read_bytes()
    assert a.manifest["audio"]["mime"] == "audio/mpeg"


def test_rejects_global_static_group(tmp_path, synthetic_source):
    import json
    import struct
    raw = bytearray(synthetic_source.read_bytes())
    mo, ml, po = struct.unpack("<QQQ", raw[16:40])
    man = json.loads(raw[mo:mo + ml])
    man["static"] = {"count": 5, "tensors": {}}
    mj = json.dumps(man).encode()
    assert 64 + len(mj) <= po
    raw[24:32] = struct.pack("<Q", len(mj))
    raw[64:64 + len(mj)] = mj
    path = tmp_path / "global-static.trigsbin"
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="global static group"):
        trigs.read(path)
