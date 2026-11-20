"""Orbit cameras around a y-up subject standing on y = 0, the rig the TRiGS PLY trainer renders its
training images from (and the held-out views the evaluation uses), as world-to-camera matrices in the
OpenCV convention gsplat takes (x right, y down, z forward)."""

from __future__ import annotations

import math

import numpy as np


def euler_matrix(rotation: str) -> np.ndarray:
    """Rotation matrix of "rx,ry,rz" degrees, extrinsic XYZ (x first, then y, then z about the world axes):
    the convention of the trainer's `--transform.rotation`."""
    rx, ry, rz = (math.radians(float(a)) for a in rotation.split(","))
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def homogeneous(R: np.ndarray) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R
    return M


def orbit(means: np.ndarray, log_scales: np.ndarray, *, longitudes: tuple[float, float, int],
          latitudes: tuple[float, float, int], fov: float = 60.0, size: int = 512, margin: float = 1.2
          ) -> tuple[np.ndarray, np.ndarray]:
    """Cameras on a sphere around the subject's middle `(0, height / 2, 0)`, at the distance that fits the
    subject's bounding sphere (splat extents included) into `fov` with `margin`.

    `longitudes` / `latitudes` are (first, last, steps) in degrees; longitude 0 looks from +z, 90 from +x;
    a full 360 degree longitude range leaves out its repeated end. Returns viewmats `[v, 4, 4]` and the
    shared intrinsics `[3, 3]` for a `size` x `size` image.
    """
    height = means[:, 1].max() - means[:, 1].min()
    centre = np.array([0.0, height / 2, 0.0])
    bound = (np.linalg.norm(means - centre, axis=1) + np.exp(log_scales).max(1)).max()
    radius = margin * bound / math.tan(math.radians(fov) / 2)
    lon0, lon1, nlon = longitudes
    lat0, lat1, nlat = latitudes
    full = abs(lon1 - lon0 - 360) < 1e-6
    views = []
    for i in range(nlat):
        lat = math.radians(lat0 + (i / (nlat - 1) if nlat > 1 else 0.5) * (lat1 - lat0))
        for j in range(nlon):
            s = j / nlon if full else (j / (nlon - 1) if nlon > 1 else 0.5)
            lon = math.radians(lon0 + s * (lon1 - lon0))
            pos = centre + radius * np.array([math.cos(lat) * math.sin(lon), math.sin(lat), math.cos(lat) * math.cos(lon)])
            forward = (centre - pos) / np.linalg.norm(centre - pos)
            up = np.array([0.0, 0.0, 1.0]) if abs(forward[1]) > 0.99 else np.array([0.0, 1.0, 0.0])
            right = np.cross(up, forward)
            right /= np.linalg.norm(right)
            R = np.stack([right, np.cross(forward, right), forward])
            V = homogeneous(R)
            V[:3, 3] = -R @ pos
            views.append(V)
    f = 0.5 * size / math.tan(math.radians(fov) / 2)
    return np.stack(views), np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1]])
