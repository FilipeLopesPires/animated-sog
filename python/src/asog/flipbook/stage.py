"""Stage one segment of a flip book: the frames the trainer sees, in a directory of their own, turned into
the trainer's frame (y up, the subject standing on y = 0).

The trainer reads every `*.ply` of its input directory in name order, one frame each at its `fps`. A
segment must not contain a lighting or scene cut: a TRiGS splat has one fixed colour over its life.

The rotation happens here and not in the trainer: the trainer's own `--transform.rotation` rotates SH
degrees 2 and 3 wrongly, so its ground truth would have the wrong view-dependent colour.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from .cameras import euler_matrix
from .ply import in_trainer_layout, read_vertices, rotate_ply, write_trainer_layout

log = logging.getLogger(__name__)


def stage(source: str | Path, out_dir: str | Path, *, start: int = 0, count: int | None = None,
          stride: int = 1, rotation: str | None = None) -> list[Path]:
    """Put frames `start, start + stride, ...` of the name-sorted PLYs in `source` into `out_dir`; `count`
    frames, or all to the end. Use `stride` to drop to the training frame rate (e.g. 2 for a 60 fps
    capture trained at 30 fps).

    `rotation` ("rx,ry,rz" degrees, extrinsic XYZ; "-90,0,0" for z-up sources) rotates every frame,
    including its SH. Frames are written in the original 3DGS layout the trainer reads (normals, then SH,
    opacity, scales, rotation); without a rotation a frame already in it is hard-linked (copied across
    drives). Returns the paths."""
    frames = sorted(Path(source).glob("*.ply"))
    picked = frames[start::stride]
    if count is not None:
        if len(picked) < count:
            raise ValueError(f"{source}: only {len(picked)} frames from {start} with stride {stride}, asked for {count}")
        picked = picked[:count]
    if not picked:
        raise ValueError(f"{source}: no PLY frames from {start}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("*.ply")):
        raise FileExistsError(f"{out} already holds PLY files; stage into an empty directory")
    R = None if rotation is None else euler_matrix(rotation)
    staged = []
    for i, p in enumerate(picked):
        dst = out / p.name
        if R is not None:
            rotate_ply(p, dst, R)
            if i % 30 == 0:
                log.info(f"  rotated {i + 1}/{len(picked)}")
        elif in_trainer_layout(p):
            try:
                os.link(p, dst)
            except OSError:
                shutil.copy2(p, dst)
        else:
            write_trainer_layout(dst, read_vertices(p)[1])
        staged.append(dst)
    log.info(f"staged {len(staged)} frames {picked[0].name} .. {picked[-1].name} into {out}"
             + (f", rotated {rotation}" if rotation else ""))
    return staged
