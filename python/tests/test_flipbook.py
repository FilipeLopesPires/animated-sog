"""Flip-book steps without the trainer. The checkpoint flip needs torch; the end-to-end run needs CUDA,
gsplat, the Open4DGS install and the Rubloff 1-second run (trained once, reused here)."""

import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from asog import sh
from asog.flipbook import stage
from asog.flipbook.cameras import euler_matrix, homogeneous, orbit
from asog.flipbook.ply import read_ply, rotate_ply

RUN = Path("C:/Dev/arrival.space/tmp/rubloff/runs/s5_1s_a")
FRAMES = Path("C:/Dev/arrival.space/tmp/rubloff/seg5_1s")
TRIGS_PYTHON = Path("C:/Dev/arrival.space/tmp/open4dgs/trigs/.venv/Scripts/python.exe")
OPEN4DGS_CORE = Path("C:/Dev/arrival.space/tmp/open4dgs/core")


def test_stage(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(10):
        (src / f"Frame{i:04d}.ply").write_bytes(bytes([i]))
    (src / "notes.txt").write_text("not a frame")
    staged = stage(src, tmp_path / "out", start=1, count=4, stride=2)
    assert [p.name for p in staged] == ["Frame0001.ply", "Frame0003.ply", "Frame0005.ply", "Frame0007.ply"]
    assert staged[2].read_bytes() == bytes([5])
    with pytest.raises(FileExistsError):
        stage(src, tmp_path / "out")
    with pytest.raises(ValueError, match="only 3 frames"):
        stage(src, tmp_path / "out2", start=4, count=4, stride=2)


def _write_ply(path: Path, n: int, rng, **columns) -> dict:
    """A random 3DGS PLY with normals and degree-3 SH; `columns` replace properties."""
    cols = {"x": rng.normal(size=n), "y": rng.normal(size=n), "z": rng.normal(size=n),
            "nx": np.zeros(n), "ny": np.zeros(n), "nz": np.zeros(n)}
    cols.update({f"f_dc_{i}": rng.normal(size=n) for i in range(3)})
    rest = rng.normal(size=(n, 3, 15))   # channel-major
    cols.update({f"f_rest_{i}": rest.reshape(n, 45)[:, i] for i in range(45)})
    cols["opacity"] = rng.normal(size=n)
    cols.update({f"scale_{i}": rng.normal(size=n) for i in range(3)})
    cols.update({f"rot_{i}": rng.normal(size=n) for i in range(4)})
    cols.update(columns)
    head = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n
    head += "".join(f"property float {k}\n" for k in cols) + "end_header\n"
    with open(path, "wb") as f:
        f.write(head.encode())
        f.write(np.stack(list(cols.values()), 1).astype("<f4").tobytes())
    return dict(cols, rest=rest)


def test_read_ply(tmp_path):
    cols = _write_ply(tmp_path / "f.ply", 50, np.random.default_rng(0))
    p = read_ply(tmp_path / "f.ply")
    np.testing.assert_allclose(p.means[:, 1], cols["y"], rtol=1e-6)
    np.testing.assert_allclose(p.quats[:, 0], cols["rot_0"], rtol=1e-6)
    np.testing.assert_allclose(p.opacity, 1 / (1 + np.exp(-cols["opacity"].astype(np.float32))), rtol=1e-6)
    assert p.sh.shape == (50, 16, 3)
    np.testing.assert_allclose(p.sh[:, 1:, 2], cols["rest"][:, 2, :], rtol=1e-6)   # blue, coefficients 1..15


ROTATIONS = ["-90,0,0", "30,-50,70", "0,0,180"]


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_sh_rotation_is_exact(rotation):
    R = euler_matrix(rotation)
    rng = np.random.default_rng(1)
    c = rng.normal(size=(15, 4))
    d = rng.normal(size=(200, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    M = sh.rotation(R)
    # the rotated colour seen from d is the original colour seen from R^T d, every band
    np.testing.assert_allclose(sh.basis(d) @ (M @ c), sh.basis(d @ R) @ c, atol=1e-12)
    np.testing.assert_allclose(sh.rotation(R.T) @ M, np.eye(15), atol=1e-12)
    np.testing.assert_allclose(sh.rotation(R, 8), M[:8, :8])


def test_rotate_ply_then_inverse_is_identity(tmp_path):
    _write_ply(tmp_path / "f.ply", 200, np.random.default_rng(0))
    R = euler_matrix("30,-50,70")
    rotate_ply(tmp_path / "f.ply", tmp_path / "r.ply", R)
    rotate_ply(tmp_path / "r.ply", tmp_path / "back.ply", R.T)
    a, r, b = (read_ply(tmp_path / f"{k}.ply") for k in ("f", "r", "back"))
    np.testing.assert_allclose(r.means, a.means @ R.T, atol=1e-5)
    assert np.abs(r.sh[:, 1:] - a.sh[:, 1:]).max() > 0.1   # the SH did change
    for k in ("means", "log_scales", "quats", "opacity", "sh"):
        np.testing.assert_allclose(getattr(b, k), getattr(a, k), atol=2e-5, err_msg=k)
    assert (tmp_path / "f.ply").read_bytes().split(b"end_header")[0] == (tmp_path / "back.ply").read_bytes().split(b"end_header")[0]


def test_stage_rotates(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _write_ply(src / "a.ply", 20, np.random.default_rng(0))
    stage(src, tmp_path / "out", rotation="-90,0,0")
    rotate_ply(src / "a.ply", tmp_path / "ref.ply", euler_matrix("-90,0,0"))
    assert (tmp_path / "out" / "a.ply").read_bytes() == (tmp_path / "ref.ply").read_bytes()


@pytest.mark.gpu
@pytest.mark.parametrize("rotation", ROTATIONS)
def test_rotated_frame_renders_identically(tmp_path, rotation):
    """A rotated frame seen from correspondingly rotated cameras is the original frame, with all SH bands
    (the trainer's own rotation fails this at 44 dB: its SH degrees 2 and 3 are wrong)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("gsplat")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from asog.flipbook.evaluate import _render

    rng = np.random.default_rng(2)
    n = 4000
    xyz = rng.normal(0, 0.3, (n, 3)) + [0, 0.9, 0]
    _write_ply(tmp_path / "f.ply", n, rng, x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], opacity=np.full(n, 0.4),
               **{f"scale_{i}": rng.normal(-3.5, 0.3, n) for i in range(3)})
    R = euler_matrix(rotation)
    rotate_ply(tmp_path / "f.ply", tmp_path / "r.ply", R)
    f, r = read_ply(tmp_path / "f.ply"), read_ply(tmp_path / "r.ply")
    views, K = orbit(f.means, f.log_scales, longitudes=(15, 375, 4), latitudes=(-33.3, 33.3, 2), size=256)
    a = _render(f.means, f.quats, f.log_scales, f.opacity, f.sh, views, K, 256)
    b = _render(r.means, r.quats, r.log_scales, r.opacity, r.sh, views @ homogeneous(R.T), K, 256)
    assert a.mean() > 0.05
    assert 10 * np.log10(1 / ((a - b) ** 2).mean()) > 100


def test_cameras():
    np.testing.assert_allclose(euler_matrix("-90,0,0") @ [0, 0, 1], [0, 1, 0], atol=1e-12)   # z up -> y up
    np.testing.assert_allclose(euler_matrix("0,0,180") @ [1, 2, 3], [-1, -2, 3], atol=1e-12)
    rng = np.random.default_rng(0)
    means = rng.uniform([-0.3, 0, -0.2], [0.3, 1.8, 0.2], (1000, 3))
    views, K = orbit(means, np.full((1000, 3), -5.0), longitudes=(0, 360, 12), latitudes=(-60, 60, 6))
    assert views.shape == (72, 4, 4)
    middle = [0, np.ptp(means[:, 1]) / 2, 0]   # the orbit assumes the subject stands on y = 0
    centre = views[:, :3, :3] @ middle + views[:, :3, 3]
    np.testing.assert_allclose(centre[:, :2], 0, atol=1e-9)   # the subject's middle is on every optical axis
    assert np.allclose(centre[:, 2], centre[0, 2]) and centre[0, 2] > 0
    positions = -np.einsum("vji,vj->vi", views[:, :3, :3], views[:, :3, 3])
    lon = np.degrees(np.arctan2(positions[:12, 0], positions[:12, 2])) % 360
    np.testing.assert_allclose(lon, np.arange(0, 360, 30), atol=1e-9)   # full circle: no repeated end
    assert K[0, 0] == pytest.approx(256 / math.tan(math.radians(30)))


def _quat_matrix(q):
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def test_flip_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from asog.flipbook import flip_checkpoint

    rng = np.random.default_rng(0)
    n = 64
    t = lambda *shape: torch.tensor(rng.normal(size=shape), dtype=torch.float32)
    params = {"means": t(n, 3), "anchors": t(n, 3), "quats": t(n, 4), "shN": t(n, 15, 3), "sh0": t(n, 3),
              "zeta_base_r": t(n, 3), "zeta_c1_t": t(n, 3), "times": t(n, 1)}
    ck = {"params": params, "motion_params": {"anchors": params["anchors"].clone()}, "fps": 30.0, "n_frames": 3}
    torch.save(ck, tmp_path / "a.pt")
    flip_checkpoint(tmp_path / "a.pt", tmp_path / "b.pt")
    out = torch.load(tmp_path / "b.pt", weights_only=True)
    p, q = {k: v.numpy().astype(np.float64) for k, v in out["params"].items()}, {k: v.numpy().astype(np.float64) for k, v in params.items()}
    Rz = np.diag([-1.0, -1.0, 1.0])
    for k in ("means", "anchors", "zeta_base_r", "zeta_c1_t"):
        np.testing.assert_allclose(p[k], q[k] @ Rz, atol=1e-6)
    np.testing.assert_array_equal(p["sh0"], q["sh0"])
    np.testing.assert_array_equal(p["times"], q["times"])
    np.testing.assert_array_equal(out["motion_params"]["anchors"].numpy(), out["params"]["anchors"].numpy())
    # quaternions come out xyzw and rotate like Rz(180) after the original rotation
    unit = q["quats"] / np.linalg.norm(q["quats"], axis=1, keepdims=True)
    flipped = p["quats"][:, [3, 0, 1, 2]] / np.linalg.norm(p["quats"], axis=1, keepdims=True)
    np.testing.assert_allclose(_quat_matrix(flipped), Rz @ _quat_matrix(unit), atol=1e-6)
    # the view-dependent colour seen from d after the flip is the colour seen from Rz^T d before it
    d = rng.normal(size=(100, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    np.testing.assert_allclose(np.einsum("vk,nkc->nvc", sh.basis(d, 15), p["shN"]),
                               np.einsum("vk,nkc->nvc", sh.basis(d @ Rz, 15), q["shN"]), atol=1e-5)


@pytest.mark.data
@pytest.mark.gpu
@pytest.mark.skipif(not (RUN.exists() and FRAMES.exists() and TRIGS_PYTHON.exists() and OPEN4DGS_CORE.exists()),
                    reason="Rubloff 1-second run or Open4DGS install not present")
def test_run_without_training(tmp_path):
    """`run` on the trained 1-second Rubloff run: reuses its checkpoint, so stage, flip, export (Open4DGS
    subprocess), encode and evaluate run for real."""
    pytest.importorskip("gsplat")
    from asog.flipbook import run

    (tmp_path / "train").mkdir()
    shutil.copy2(RUN / "trigs_ckpt.pt", tmp_path / "train" / "trigs_ckpt.pt")
    result = run(FRAMES, tmp_path, count=30, rotation="-90,0,0", trigs_python=TRIGS_PYTHON, open4dgs_core=OPEN4DGS_CORE)
    payload = lambda p: p.read_bytes()[int.from_bytes(p.read_bytes()[32:40], "little"):]
    assert payload(tmp_path / "clip.trigsbin") == payload(RUN / "rubloff_1s.trigsbin")
    assert len(list((tmp_path / "frames").glob("*.ply"))) == 30
    # measured 39.6 dB against the true flip book (see docs/pipeline.md on the trainer's SH rotation)
    assert result.psnr_mean > 38
    assert result.jitter_student < result.jitter_teacher
    assert (tmp_path / "compare.png").stat().st_size > 0
