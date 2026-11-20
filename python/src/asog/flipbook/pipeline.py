"""`run`: the whole flip-book pipeline for one segment, in one work directory.

    <work>/frames/          staged teacher PLYs, rotated to y up
    <work>/train/           trainer output (trigs_ckpt.pt, tb/)
    <work>/ckpt_ydown.pt    flipped checkpoint
    <work>/clip.trigsbin    Open4DGS export
    <work>/clip.asog        the result
    <work>/compare.png      teacher | student strip

Training is the slow step (22 minutes for 180 frames on an RTX 5080), so a work directory that already
has staged frames and `train/trigs_ckpt.pt` reuses them (delete them to stage or train again); the other
steps always run.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..encode import encode
from .evaluate import Evaluation, evaluate
from .flip import flip_checkpoint
from .open4dgs import export, train
from .stage import stage

log = logging.getLogger(__name__)


def run(source: str | Path, work: str | Path, *, start: int = 0, count: int | None = None, stride: int = 1,
        rotation: str | None = None, fps: float = 30, keyframe_stride: int = 9, target_count: int = 400_000,
        steps: int = 60_000, latitude_steps: int = 6, image_size: int = 512, trigs_python: str | Path | None = None,
        open4dgs_core: str | Path | None = None, audio: str | Path | None = None, **encode_options) -> Evaluation:
    """Stage, train, flip, export, encode and evaluate one segment of the flip book in `source`.

    `rotation` ("rx,ry,rz" degrees, "-90,0,0" for z-up sources) is applied when staging, SH included; the
    trainer gets y-up frames and no rotation of its own. `encode_options` go to `asog.encode` (qp, qa,
    prune, ...). Returns the held-out evaluation against the staged frames (the flip book, rotated).
    """
    work = Path(work)
    frames = work / "frames"
    if not any(frames.glob("*.ply")):
        stage(source, frames, start=start, count=count, stride=stride, rotation=rotation)
    ckpt = work / "train" / "trigs_ckpt.pt"
    if ckpt.exists():
        log.info(f"reusing {ckpt}")
    else:
        train(frames, ckpt.parent, fps=fps, keyframe_stride=keyframe_stride,
              target_count=target_count, steps=steps, latitude_steps=latitude_steps, image_size=image_size, trigs_python=trigs_python)
    flip_checkpoint(ckpt, work / "ckpt_ydown.pt")
    export([work / "ckpt_ydown.pt"], work / "clip.trigsbin", trigs_python=trigs_python, open4dgs_core=open4dgs_core)
    stats = encode(work / "clip.trigsbin", work / "clip.asog", audio=audio, **encode_options)
    log.info(f"{work / 'clip.asog'}: {stats.bytes / 1e6:.1f} MB, {stats.count} splats")
    result = evaluate(frames, work / "clip.asog", compare_png=work / "compare.png")
    log.info(str(result))
    return result
