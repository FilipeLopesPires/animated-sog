"""Flip book -> `.asog`: convert a volumetric video stored as one independently trained 3DGS PLY per frame.

A flip book has no splat identity across frames, so there is no motion to extract; it is distilled into a
TRiGS model by the Open4DGS TRiGS trainer, which renders its own training images from the PLYs. The
trainer and the timeline exporter are external tools (Open4DGS TRiGS is AGPL-3.0): this package only runs
them as subprocesses and never imports their modules. Steps, one lighting / scene segment at a time:

1. `stage`    pick the segment's frames, rotated to y up (SH included)
2. `train`    TRiGS trainer (subprocess)                      -> trigs_ckpt.pt (y up)
3. `flip`     rotate the checkpoint 180 deg about z to y down -> ckpt_ydown.pt
4. `export`   Open4DGS make-trigs-timeline.py (subprocess)    -> .trigsbin
5. `encode`   `asog.encode`                                    -> .asog
6. `evaluate` render teacher PLYs and the .asog on held-out cameras (gsplat): PSNR, flicker, jitter

`run` chains them for one segment. See docs/pipeline.md.
"""

from .evaluate import evaluate
from .flip import flip_checkpoint
from .open4dgs import export, train
from .pipeline import run
from .stage import stage

__all__ = ["evaluate", "export", "flip_checkpoint", "run", "stage", "train"]
