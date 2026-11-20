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


def train(frames: str | Path, run_dir: str | Path, *, fps: float = 30,
          keyframe_stride: int = 9, target_count: int = 400_000, steps: int = 60_000, latitude_steps: int = 6,
          trigs_python: str | Path | None = None) -> Path:
    """Train TRiGS on the PLY frames in `frames`; returns `<run_dir>/trigs_ckpt.pt` (y up).

    The trainer renders its training images from the PLYs on an orbit (12 longitudes x `latitude_steps`
    rows, 512 px) around a y-up scene with its lowest point at y = 0, so the frames must already be there
    (`stage --rotation`); the trainer's own `--transform.rotation` is left at its identity default because
    it rotates SH degrees 2 and 3 wrongly. Its ground-truth cache lives in RAM
    (about frames x views x 512^2 x 4 bytes). `keyframe_stride` sets how many frames seed the 4D init; at 3
    with 400k splats the init's farthest-point sampling took over 14 minutes and filled the GPU, at 9 it
    starts in seconds. Debug renders and per-frame student PLYs are switched off.
    """
    run_dir = Path(run_dir)
    _run([interpreter(trigs_python), "-m", "trigs.cli.train_from_plys_upstream",
          "--input-dir", str(frames), "--output-dir", str(run_dir),
          "--camera.latitude-steps", str(latitude_steps),
          "--config.fps", str(fps), "--config.keyframe-stride", str(keyframe_stride),
          "--config.target-count", str(target_count), "--config.n-steps", str(steps),
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
