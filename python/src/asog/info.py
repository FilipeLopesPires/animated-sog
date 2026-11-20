"""Human-readable summary of an `.asog` file: manifest, clips, and the size of every stream."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import container


def summary(path: str | Path) -> str:
    """Manifest summary and per-stream sizes; reads only the manifest and the `flags` stream."""
    size = Path(path).stat().st_size
    with open(path, "rb") as f:
        man, header = container.read_manifest(f)
        fl = next(s for s in man["streams"] if s["name"] == "flags")
        f.seek(header.payload_offset + fl["byteOffset"])
        flags = container.decode_stream(fl, f.read(fl["byteLength"]))[:, 0]
    n = man["count"]
    tl = man["timeline"]
    n_static = int((flags & 1).sum())
    lines = [
        f"{Path(path).name}: {man['format']} v{man['version']}, {size / 1e6:.2f} MB, {size / n:.2f} B/splat",
        f"splats {n} ({n_static} static, {n - n_static} dynamic)",
        f"timeline {tl['durationSeconds']:g} s at {tl['frameRate']:g} fps ({tl.get('totalFrames', '?')} frames), "
        f"{len(man['clips'])} clip(s)",
    ]
    clip_counts = np.bincount(flags >> 1, minlength=len(man["clips"]))
    for c, clip in enumerate(man["clips"]):
        g = clip["global"]
        lines.append(f"  clip {c} {clip.get('id', '')}: {clip_counts[c]} splats, active {g['activeTimeStart']:g}-"
                     f"{g['activeTimeEnd']:g} s, core {g['coreTimeStart']:g}-{g['coreTimeEnd']:g} s")
    if "shN" in man:
        lines.append(f"SH: {man['shN']['coeffs']} coefficients, palette of {man['shN']['count']}")
    m = man["motion"]
    lines.append(f"motion: qp {m['qp']:g} mm, qa {m['qa']:g} mm; mu 1/{man['mu']['scale']} s; "
                 f"log sigma {man['logsigma']['bits']} bits over [{man['logsigma']['min']:.3f}, {man['logsigma']['max']:.3f}]")
    if "audio" in man:
        lines.append(f"audio: {man['audio']['mime']}, {man['audio']['byteLength'] / 1e3:.1f} kB")
    lines.append(f"header + manifest {header.payload_offset} B, payload {header.payload_length} B")
    lines.append(f"  {'stream':10s} {'count':>8s} {'ch':>3s} {'planes':>6s} {'delta':>5s} {'zigzag':>6s} "
                 f"{'bytes':>10s} {'B/splat':>8s} {'share':>6s}")
    for s in man["streams"]:
        lines.append(f"  {s['name']:10s} {s['count']:8d} {s['channels']:3d} {s['planes']:6d} {int(s['delta']):5d} "
                     f"{int(s['zigzag']):6d} {s['byteLength']:10d} {s['byteLength'] / n:8.3f} "
                     f"{100 * s['byteLength'] / size:5.1f}%")
    return "\n".join(lines)
