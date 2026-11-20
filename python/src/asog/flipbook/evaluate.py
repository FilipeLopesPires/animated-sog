"""Held-out evaluation of a distilled clip against its flip book.

The teacher (the per-frame PLYs) and the student (the `.asog`, as a player shows it) are rendered with
gsplat from cameras between the training orbit's views: 4 longitudes offset by half the training step,
2 rows at +-33.3 degrees. Rather than rotating splats (which would mean rotating SH), each side keeps its
own frame and the cameras are moved: teacher frame --rotation--> trainer's y up --180 deg about z--> y down.
Staged frames are already y up (rotation 0,0,0); the original flip book can be given with its rotation.

Metrics, per pixel in 0..1:

- PSNR of student vs teacher, per frame and view;
- flicker: mean |frame(t+1) - frame(t)|, over the full frame and on the subject (pixels covered in either
  image), for both sequences: per-frame PLYs flicker by themselves, a good distillation does not add any;
- jitter: |x(t) - (x(t-1) + x(t+1)) / 2| on the subject, near zero for smooth motion and large for
  splats that pop in and out;
- primitive lifetimes: visible life of the dynamic splats in frames (2 x 2.45 sigma x fps).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import decode
from .cameras import euler_matrix, homogeneous, orbit
from .ply import read_ply

log = logging.getLogger(__name__)

HELD_OUT_LONGITUDES = (15.0, 375.0, 4)
HELD_OUT_LATITUDES = (-33.3, 33.3, 2)
SUBJECT_THRESHOLD = 0.02   # a pixel with any channel above this is on the subject
Y_DOWN = euler_matrix("0,0,180")


@dataclass
class Evaluation:
    psnr_mean: float
    psnr_min_frame: float
    psnr_min_view: float
    flicker_teacher: float
    flicker_student: float
    flicker_subject_teacher: float
    flicker_subject_student: float
    jitter_teacher: float
    jitter_student: float
    life_frames: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        lf = self.life_frames
        return "\n".join([
            f"PSNR student vs teacher: mean {self.psnr_mean:.2f} dB, min frame {self.psnr_min_frame:.2f}, "
            f"min view {self.psnr_min_view:.2f}",
            f"flicker full frame: teacher {self.flicker_teacher:.5f}  student {self.flicker_student:.5f}  "
            f"ratio {self.flicker_student / self.flicker_teacher:.2f}",
            f"flicker on subject: teacher {self.flicker_subject_teacher:.4f}  student {self.flicker_subject_student:.4f}  "
            f"ratio {self.flicker_subject_student / self.flicker_subject_teacher:.2f}",
            f"jitter on subject: teacher {self.jitter_teacher:.4f}  student {self.jitter_student:.4f}  "
            f"ratio {self.jitter_student / self.jitter_teacher:.2f}",
            f"dynamic splat life in frames: median {lf['median']:.1f}, p10 {lf['p10']:.1f}, p90 {lf['p90']:.1f}; "
            f"< 3 frames {lf['under3_percent']:.1f}%",
        ])


def _render(means, quats, log_scales, opacity, sh, viewmats, K, size) -> np.ndarray:
    """gsplat render of the splats from every view: `[v, size, size, 3]` in 0..1."""
    import torch
    from gsplat import rasterization

    t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device="cuda")
    degree = int(round(np.sqrt(sh.shape[1]))) - 1
    img, _, _ = rasterization(t(means), t(quats), t(np.exp(log_scales)), t(opacity), t(sh),
                              t(viewmats), t(K)[None].expand(len(viewmats), 3, 3).contiguous(), size, size,
                              sh_degree=degree, packed=False)
    return img.clamp(0, 1).cpu().numpy()


def evaluate(teacher_frames: str | Path, clip: str | Path, *, rotation: str = "0,0,0", size: int = 512,
             compare_png: str | Path | None = None) -> Evaluation:
    """Compare the `.asog` `clip` with the PLY frames in `teacher_frames` (name order, one per clip frame;
    `rotation` takes them to the trainer's y-up frame: 0,0,0 for staged frames). Optionally writes a teacher | student strip of frame 0 and the
    middle frame, first two views, upright, to `compare_png`."""
    paths = sorted(Path(teacher_frames).glob("*.ply"))
    a = decode.read(clip)
    fps = a.manifest["timeline"]["frameRate"]
    if len(paths) != a.manifest["timeline"]["totalFrames"]:
        raise ValueError(f"{len(paths)} teacher frames for a {a.manifest['timeline']['totalFrames']}-frame clip")
    R = euler_matrix(rotation)
    first = read_ply(paths[0])
    views, K = orbit(first.means @ R.T, first.log_scales, longitudes=HELD_OUT_LONGITUDES, latitudes=HELD_OUT_LATITUDES,
                     size=size)
    views_teacher = views @ homogeneous(R)       # y-up world = R @ teacher world
    views_student = views @ homogeneous(Y_DOWN)  # y-up world = Rz(180) @ y-down world
    sh_student = a.sh0[:, None] if a.shN is None else np.concatenate([a.sh0[:, None], a.shN], 1)

    n = len(paths)
    sq_err = np.zeros((n, len(views)))
    acc = dict.fromkeys(["fl_t", "fl_s", "fl_n", "fls_t", "fls_s", "fls_n", "j_t", "j_s", "j_n"], 0.0)
    prev: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []   # (teacher, student, mask) of the last 2 frames
    strip = []
    for f, path in enumerate(paths):
        p = read_ply(path)
        teacher = _render(p.means, p.quats, p.log_scales, p.opacity, p.sh, views_teacher, K, size)
        fr = a.evaluate(f / fps)
        m = fr.weights > 0
        student = _render(fr.positions[m], fr.rotations[m], a.log_scales[m], (a.opacity * fr.weights)[m],
                          sh_student[m], views_student, K, size)
        mask = (teacher.max(-1) > SUBJECT_THRESHOLD) | (student.max(-1) > SUBJECT_THRESHOLD)
        sq_err[f] = ((teacher - student) ** 2).mean(axis=(1, 2, 3))
        if prev:
            t1, s1, m1 = prev[-1]
            acc["fl_t"] += np.abs(teacher - t1).sum()
            acc["fl_s"] += np.abs(student - s1).sum()
            acc["fl_n"] += teacher.size
            m2 = mask | m1
            acc["fls_t"] += (np.abs(teacher - t1).max(-1) * m2).sum()
            acc["fls_s"] += (np.abs(student - s1).max(-1) * m2).sum()
            acc["fls_n"] += m2.sum()
        if len(prev) == 2:
            (t0, s0, m0), (t1, s1, m1) = prev
            m3 = m0 | m1 | mask
            acc["j_t"] += (np.abs(t1 - 0.5 * (t0 + teacher)).max(-1) * m3).sum()
            acc["j_s"] += (np.abs(s1 - 0.5 * (s0 + student)).max(-1) * m3).sum()
            acc["j_n"] += m3.sum()
        prev = (prev + [(teacher, student, mask)])[-2:]
        if f in (0, n // 2):
            strip.append(np.concatenate([np.concatenate([teacher[v], student[v]], 1)[::-1] for v in range(2)], 1))
        log.info(f"  frame {f + 1}/{n}: {10 * np.log10(1 / max(sq_err[f].mean(), 1e-10)):.2f} dB")

    psnr = 10 * np.log10(1 / np.maximum(sq_err, 1e-10))
    life = 2 * a.manifest["motion"]["sm"] * a.sigma[~a.static] * fps
    if compare_png is not None:
        import imageio.v3 as iio
        iio.imwrite(compare_png, (np.concatenate(strip, 0) * 255).round().astype(np.uint8))
    return Evaluation(
        psnr_mean=float(psnr.mean()), psnr_min_frame=float(psnr.mean(1).min()), psnr_min_view=float(psnr.mean(0).min()),
        flicker_teacher=acc["fl_t"] / acc["fl_n"], flicker_student=acc["fl_s"] / acc["fl_n"],
        flicker_subject_teacher=acc["fls_t"] / acc["fls_n"], flicker_subject_student=acc["fls_s"] / acc["fls_n"],
        jitter_teacher=acc["j_t"] / acc["j_n"], jitter_student=acc["j_s"] / acc["j_n"],
        life_frames=dict(median=float(np.median(life)), p10=float(np.percentile(life, 10)),
                         p90=float(np.percentile(life, 90)), under3_percent=float((life < 3).mean() * 100)),
    )
