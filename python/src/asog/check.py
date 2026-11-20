"""Error report of an `.asog` file against the `.trigsbin` it was encoded from.

The encoder's splat selection and order are a function of the source and the kept count, so the report
re-derives them (`encode.splat_order`) and compares splat by splat: the centre error over each dynamic
splat's visible life, and the errors of position, rotation, scale, colour, opacity and view-dependent SH.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import decode, sh, trigs
from .encode import splat_order
from .motion import SH_C0, bezier2, bezier4, life_tau, screw

LIFE_SAMPLES = 15
MOTION_SUBSAMPLE = 7   # every 7th dynamic splat for the motion error
SH_SUBSAMPLE = 9


def source_centres(src: trigs.TrigsRuntime, t_local: np.ndarray) -> np.ndarray:
    """Centres of source splats at clip-local times `t_local` with TRiGS's own twist (nu along omega)."""
    sigma = np.exp(src.log_durations)
    dt = t_local - src.times
    tau = life_tau(dt, sigma)
    om, nu = bezier2(tau, src.omega), bezier2(tau, src.nu)
    oo = (om * om).sum(1, keepdims=True)
    ok = oo >= 1e-8
    nu = np.where(ok, (nu * om).sum(1, keepdims=True) / np.where(ok, oo, 1) * om, nu)
    return screw(src.means, src.anchors, om, nu, dt)[0]


def decoded_centres(a: decode.Asog, idx: np.ndarray, t_local: np.ndarray) -> np.ndarray:
    """Centres of decoded splats `idx` at clip-local times `t_local` (no fades or culling)."""
    dt = t_local - a.mu[idx]
    tau = life_tau(dt, a.sigma[idx])
    om = bezier2(tau, a.omega[:, idx])
    nu = bezier4(tau, a.g[:, idx]).reshape(-1, 1) / np.maximum((om * om).sum(1, keepdims=True), 1e-24) * om
    return screw(a.means[idx], a.anchor[idx], om, nu, dt)[0]


def motion_error(src: trigs.TrigsRuntime, a: decode.Asog, idx: np.ndarray) -> np.ndarray:
    """Largest centre error over the visible life (+-sm sigma of the source) of splats `idx` (both in file order)."""
    s = src.take(idx)
    sigma = np.exp(s.log_durations)
    err = np.zeros(len(idx))
    for k in np.linspace(-a.manifest["motion"]["sm"], a.manifest["motion"]["sm"], LIFE_SAMPLES):
        t = s.times + k * sigma
        err = np.maximum(err, np.linalg.norm(decoded_centres(a, idx, t) - source_centres(s, t), axis=1))
    return err


def report(source: str | Path, asog: str | Path) -> dict[str, float]:
    """Errors of `asog` against `source`; position and motion errors in file units x 1e-3 (mm for metre files)."""
    a = decode.read(asog)
    how = a.manifest.get("source", {})   # the encoder's quaternion order and orientation fix of the source
    src = trigs.read(source, how.get("quaternionOrder"))
    if how.get("transform") == "z180":
        src = trigs.rotate_z180(src)
    src = src.take(splat_order(src, a.count))
    pc = lambda x, p: float(np.percentile(x, p))
    r: dict[str, float] = {
        "splats": a.count,
        "payload_bytes_per_splat": sum(s["byteLength"] for s in a.manifest["streams"]) / a.count,
    }
    dyn = np.nonzero(~a.static)[0][::MOTION_SUBSAMPLE]
    if len(dyn):
        e = motion_error(src, a, dyn) * 1000
        r.update(motion_p50=pc(e, 50), motion_p90=pc(e, 90), motion_p99=pc(e, 99), motion_p999=pc(e, 99.9))
    pos = np.linalg.norm(a.means - src.means, axis=1) * 1000
    r.update(position_p99=pc(pos, 99), position_max=float(pos.max()))
    ang = 2 * np.degrees(np.arccos(np.clip(np.abs((a.quats * src.quats).sum(1)), 0, 1)))
    r.update(rotation_deg_p50=pc(ang, 50), rotation_deg_p99=pc(ang, 99))
    ls = np.abs(a.log_scales - src.log_scales)
    r.update(log_scale_mean=float(ls.mean()), log_scale_p99=pc(ls, 99))
    dc = np.abs(a.sh0 - src.sh0) * SH_C0 * 255
    r.update(sh0_colour_mean=float(dc.mean()), sh0_colour_p99=pc(dc, 99))
    op = np.abs(a.opacity - src.opacity)
    r.update(opacity_mean=float(op.mean()), opacity_max=float(op.max()))
    if a.shN is not None:
        dirs = np.random.default_rng(0).normal(size=(64, 3))
        basis = sh.basis(dirs / np.linalg.norm(dirs, axis=1, keepdims=True), a.shN.shape[1])
        sub = np.arange(0, a.count, SH_SUBSAMPLE)
        r["sh_view_colour_mean"] = float(np.abs(np.einsum("vk,nkc->nvc", basis, a.shN[sub] - src.shN[sub])).mean() * 255)
        r["sh_view_colour_dropped"] = float(np.abs(np.einsum("vk,nkc->nvc", basis, src.shN[sub])).mean() * 255)
    return r


def format_report(r: dict[str, float]) -> str:
    lines = [f"splats {r['splats']}, payload {r['payload_bytes_per_splat']:.2f} B/splat"]
    if "motion_p50" in r:
        lines.append(f"centre error over life (file units x 1e-3): p50 {r['motion_p50']:.3f}  p90 {r['motion_p90']:.3f}  "
                     f"p99 {r['motion_p99']:.3f}  p99.9 {r['motion_p999']:.3f}")
    lines += [
        f"position error (file units x 1e-3): p99 {r['position_p99']:.4f}  max {r['position_max']:.4f}",
        f"rotation error (deg): p50 {r['rotation_deg_p50']:.3f}  p99 {r['rotation_deg_p99']:.3f}",
        f"log scale error: mean {r['log_scale_mean']:.4f}  p99 {r['log_scale_p99']:.4f}",
        f"sh0 colour error (/255): mean {r['sh0_colour_mean']:.2f}  p99 {r['sh0_colour_p99']:.2f}",
        f"opacity error: mean {r['opacity_mean']:.4f}  max {r['opacity_max']:.4f}",
    ]
    if "sh_view_colour_mean" in r:
        lines.append(f"SH view colour error (/255): mean {r['sh_view_colour_mean']:.2f} "
                     f"(dropping SH: {r['sh_view_colour_dropped']:.2f})")
    return "\n".join(lines)

