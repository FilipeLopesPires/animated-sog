"""3D Gaussian splatting PLY files (binary little endian): reading, rotating, and writing the original INRIA
3DGS layout, the only one the Open4DGS trainer reads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import sh
from ..motion import quat_mul

PLY_TYPES = {"char": "i1", "uchar": "u1", "short": "<i2", "ushort": "<u2", "int": "<i4", "uint": "<u4",
             "float": "<f4", "double": "<f8", "int8": "i1", "uint8": "u1", "int16": "<i2", "uint16": "<u2",
             "int32": "<i4", "uint32": "<u4", "float32": "<f4", "float64": "<f8"}


@dataclass
class GaussianPly:
    means: np.ndarray        # [n, 3]
    log_scales: np.ndarray   # [n, 3]
    quats: np.ndarray        # [n, 4] wxyz, as stored (not normalised)
    opacity: np.ndarray      # [n] after the sigmoid
    sh: np.ndarray           # [n, 1 + coeffs, 3] SH coefficients, degree 0 first


def read_vertices(path: str | Path) -> tuple[bytes, np.ndarray]:
    """The header (verbatim) and the vertex table (a structured array) of a PLY holding only vertices."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path}: not a PLY file")
        props: list[tuple[str, str]] = []
        count, elements, fmt = 0, [], None
        while True:
            line = f.readline().decode("ascii").strip()
            if line == "end_header":
                break
            parts = line.split()
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                elements.append(parts[1])
                if parts[1] == "vertex":
                    count = int(parts[2])
            elif parts[0] == "property" and elements[-1] == "vertex":
                if parts[1] == "list":
                    raise ValueError(f"{path}: list properties in the vertex element are not supported")
                props.append((parts[2], PLY_TYPES[parts[1]]))
        if fmt != "binary_little_endian":
            raise ValueError(f"{path}: {fmt} PLY, only binary_little_endian is supported")
        if elements != ["vertex"]:
            raise ValueError(f"{path}: elements {elements}, expected only vertex")
        header = f.tell()
        v = np.fromfile(f, dtype=np.dtype(props), count=count)
    with open(path, "rb") as f:
        return f.read(header), v


def read_ply(path: str | Path) -> GaussianPly:
    """Read a 3DGS PLY (`x y z`, `f_dc_*`, `f_rest_*`, `opacity`, `scale_*`, `rot_*`; other properties
    such as normals are skipped)."""
    _, v = read_vertices(path)
    col = lambda *keys: np.stack([v[k].astype(np.float64) for k in keys], 1)
    rest = _rest_names(v)
    sh_ = col("f_dc_0", "f_dc_1", "f_dc_2")[:, None, :]
    if rest:
        # stored channel-major: all red coefficients, then green, then blue
        sh_ = np.concatenate([sh_, col(*rest).reshape(len(v), 3, len(rest) // 3).transpose(0, 2, 1)], 1)
    return GaussianPly(
        means=col("x", "y", "z"),
        log_scales=col("scale_0", "scale_1", "scale_2"),
        quats=col("rot_0", "rot_1", "rot_2", "rot_3"),
        opacity=1 / (1 + np.exp(-v["opacity"].astype(np.float64))),
        sh=sh_,
    )


def _rest_names(v: np.ndarray) -> list[str]:
    return sorted((p for p in v.dtype.names if p.startswith("f_rest_")), key=lambda p: int(p[7:]))


def trainer_layout(v: np.ndarray) -> list[str]:
    """The original INRIA 3DGS property order for the vertex table `v`: positions, normals, SH, opacity,
    scales, rotation (the Open4DGS trainer rejects any other order, and a file without normals)."""
    return ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", *_rest_names(v),
            "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


def in_trainer_layout(path: str | Path) -> bool:
    _, v = read_vertices(path)
    return list(v.dtype.names) == trainer_layout(v) and all(v.dtype[n] == np.dtype("<f4") for n in v.dtype.names)


def write_trainer_layout(dst: str | Path, v: np.ndarray) -> None:
    """Write the vertex table `v` in the trainer's layout, all float32; normals are zero when `v` has none."""
    names = trainer_layout(v)
    out = np.zeros(len(v), dtype=[(n, "<f4") for n in names])
    for n in names:
        if n in v.dtype.names:
            out[n] = v[n]
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\n"
    header += "".join(f"property float {n}\n" for n in names) + "end_header\n"
    with open(dst, "wb") as f:
        f.write(header.encode("ascii"))
        out.tofile(f)


def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """Unit wxyz quaternion of a rotation matrix."""
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def rotate_ply(src: str | Path, dst: str | Path, R: np.ndarray) -> None:
    """Write the frame `src` rotated by `R` (about the origin) to `dst`, in the trainer's layout: positions
    and normals rotate, quaternions are pre-multiplied, SH degrees 1..3 go through their rotation matrices."""
    _, v = read_vertices(src)
    v = v.copy()

    def set_cols(keys: tuple[str, ...], values: np.ndarray) -> None:
        for i, k in enumerate(keys):
            v[k] = values[:, i]

    get = lambda keys: np.stack([v[k].astype(np.float64) for k in keys], 1)
    for keys in (("x", "y", "z"), ("nx", "ny", "nz")):
        if keys[0] in v.dtype.names:
            set_cols(keys, get(keys) @ R.T)
    rot = ("rot_0", "rot_1", "rot_2", "rot_3")
    set_cols(rot, quat_mul(np.broadcast_to(quat_from_matrix(R), (len(v), 4)), get(rot)))
    rest = _rest_names(v)
    if rest:
        k = len(rest) // 3
        c = get(rest).reshape(len(v), 3, k)                       # channel-major
        set_cols(tuple(rest), np.einsum("jk,nck->ncj", sh.rotation(R, k), c).reshape(len(v), 3 * k))
    write_trainer_layout(dst, v)
