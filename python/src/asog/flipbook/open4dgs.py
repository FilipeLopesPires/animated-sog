"""The external Open4DGS tools, run as subprocesses: the TRiGS PLY-sequence trainer and the timeline
exporter `make-trigs-timeline.py`.

Open4DGS TRiGS (https://github.com/ASU-NEM-Labs/open4dgs-trigs) is AGPL-3.0 and open4dgs-core
(https://github.com/ASU-NEM-Labs/open4dgs-core) holds the exporter; neither is a dependency of this
package and none of their modules are imported. Point at them with:

- `ASOG_TRIGS_PYTHON` (or `trigs_python=`): the Python interpreter of the TRiGS install, or a wrapper
  script that sets up its environment and passes its arguments on to it (on Windows a `.bat` that calls
  vcvars64 and then the venv's python.exe); it also runs the exporter, which needs numpy and torch.
- `ASOG_OPEN4DGS_CORE` (or `open4dgs_core=`): an open4dgs-core checkout.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

ENV_PYTHON = "ASOG_TRIGS_PYTHON"
ENV_CORE = "ASOG_OPEN4DGS_CORE"
TIMELINE_SCRIPT = Path("deliver") / "make-trigs-timeline.py"


def interpreter(value: str | Path | None = None) -> str:
    """The TRiGS interpreter (or wrapper): `value`, else `$ASOG_TRIGS_PYTHON`."""
    value = value or os.environ.get(ENV_PYTHON)
    if not value:
        raise RuntimeError(f"no TRiGS interpreter: pass --trigs-python or set {ENV_PYTHON}")
    return str(value)


def timeline_script(core: str | Path | None = None) -> Path:
    """open4dgs-core's make-trigs-timeline.py: under `core`, else under `$ASOG_OPEN4DGS_CORE`."""
    core = core or os.environ.get(ENV_CORE)
    if not core:
        raise RuntimeError(f"no open4dgs-core checkout: pass --open4dgs-core or set {ENV_CORE}")
    script = Path(core) / TIMELINE_SCRIPT
    if not script.is_file():
        raise FileNotFoundError(f"{script} not found (is {core} an open4dgs-core checkout?)")
    return script


def _run(cmd: list[str]) -> None:
    log.info("$ " + subprocess.list2cmdline(cmd))
    subprocess.run(cmd, check=True)


# The dense recipe for flip books (each frame trained on its own, e.g. per-frame 3DGS exports): every frame seeds the
# canonical set, thinned at random to the budget, splats live as short as the motion needs, no relocation (it cloned
# faint splats at full opacity) and no opacity regulariser (it killed splats). Needs the trainer options of the
# arrival-space fork of open4dgs-trigs (docker/trainer/setup.sh). Measured in docs/pipeline.md ("Dense recipe").
DENSE_ARGS = [
    "--config.target-random", "--config.target-random-weight", "opacity", "--config.lambda-reg", "0",
    "--config.lr-durations", "5e-3", "--config.sigma-t-max", "1.0", "--config.opacity-logit-upper-clamp-until-step", "-1",
    "--config.no-relocate", "--config.lr-means", "1.6e-5", "--config.knn-recompute-every", "5000",
    "--camera.latitude-min", "-20", "--camera.fov", "43.3", "--camera.margin", "0.825",
]
RECIPES = ("classic", "dense")


def train(frames: str | Path, run_dir: str | Path, *, fps: float = 30,
          keyframe_stride: int | None = None, target_count: int = 400_000, steps: int = 60_000, latitude_steps: int = 6,
          image_size: int = 512, sigma_t: float | None = None, recipe: str = "classic", head_boost: float = 1.0,
          trainer_args: list[str] | tuple[str, ...] = (), trigs_python: str | Path | None = None) -> Path:
    """Train TRiGS on the PLY frames in `frames`; returns `<run_dir>/trigs_ckpt.pt` (y up).

    The trainer renders its training images from the PLYs on an orbit (12 longitudes x `latitude_steps`
    rows, `image_size` px square) around a y-up scene with its lowest point at y = 0, so the frames must already be there
    (`stage --rotation`); the trainer's own `--transform.rotation` is left at its identity default because
    it rotates SH degrees 2 and 3 wrongly. Its ground-truth cache lives in RAM
    (about frames x views x image_size^2 x 4 bytes). `keyframe_stride` sets how many frames seed the 4D init (default
    9; 1 with the dense recipe). With the classic recipe at 3 and 400k splats the init's farthest-point sampling took
    over 14 minutes and filled the GPU, at 9 it starts in seconds. `sigma_t` is every splat's initial lifetime in seconds
    (the trainer's default: one keyframe gap, `keyframe_stride / fps`; 0.125 s with the dense recipe); a splat is drawn
    within about 4 sigma_t of its keyframe. A long one keeps nearly every splat active in every frame: a 15 fps clip at
    stride 30 (2 s) ran out of GPU memory around step 2,500 at 1024 px (80 GB H100) and swapped at step 2,100 on a 16 GB
    card, where 0.6 s held 5 GB.

    `recipe="dense"` adds DENSE_ARGS and frames the orbit cropped to a standing person (portrait, `image_size` high,
    0.6875 of it wide: 1536 -> 1056). `head_boost` > 1 keeps that many times the share of splats for the head
    (3 for face close-ups). `trainer_args` are appended as they are. Debug renders and per-frame student PLYs are
    switched off.
    """
    if recipe not in RECIPES:
        raise ValueError(f"recipe {recipe!r}: expected one of {RECIPES}")
    dense = recipe == "dense"
    stride = keyframe_stride or (1 if dense else 9)
    width = 16 * round(image_size * 0.6875 / 16) if dense else image_size
    if dense and sigma_t is None:
        sigma_t = 0.125
    run_dir = Path(run_dir)
    _run([interpreter(trigs_python), "-m", "trigs.cli.train_from_plys_upstream",
          "--input-dir", str(frames), "--output-dir", str(run_dir),
          "--camera.latitude-steps", str(latitude_steps),
          "--camera.width", str(width), "--camera.height", str(image_size),
          "--config.fps", str(fps), "--config.keyframe-stride", str(stride),
          "--config.target-count", str(target_count), "--config.n-steps", str(steps),
          *(["--config.init-sigma-t", str(sigma_t)] if sigma_t else []),
          *(DENSE_ARGS if dense else []),
          *(["--config.target-random-head-boost", str(head_boost)] if head_boost != 1 else []),
          *trainer_args,
          "--no-render-debug", "--no-save-plys"])
    return run_dir / "trigs_ckpt.pt"


def export(checkpoints: list[str | Path], output: str | Path, *, trigs_python: str | Path | None = None,
           open4dgs_core: str | Path | None = None) -> Path:
    """Export y-down checkpoints (from `flip_checkpoint`) to one `.trigsbin`, one clip per checkpoint,
    back to back in the given order with hard cuts. Frame counts and rate come from the checkpoints."""
    import torch

    meta = []
    for path in checkpoints:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        meta.append((float(ck["fps"]), int(ck["n_frames"])))
    rates = {fps for fps, _ in meta}
    if len(rates) != 1:
        raise ValueError(f"checkpoints trained at different frame rates: {sorted(rates)}")
    cmd = [interpreter(trigs_python), str(timeline_script(open4dgs_core)), "from-checkpoints",
           "--frame-rate", f"{rates.pop():g}", "--output", str(output), "--overwrite"]
    start = 0
    for path, (_, frames) in zip(checkpoints, meta, strict=True):
        end = start + frames
        cmd += ["--checkpoint", str(path), "--range", f"{start}:{end}:{start}:{end}"]
        start = end
    _run(cmd)
    return Path(output)
