"""Real spherical harmonics of degrees 1..3 in the 3DGS / gsplat convention (coefficient order and signs
of `f_rest_*`), and the matrices that rotate SH coefficients."""

from __future__ import annotations

import numpy as np

BANDS = ((0, 3), (3, 8), (8, 15))   # coefficient ranges of degrees 1, 2, 3


def basis(dirs: np.ndarray, coeffs: int = 15) -> np.ndarray:
    """SH basis of degrees 1..3 at unit directions `[v, 3]`: `[v, coeffs]`."""
    x, y, z = dirs.T
    xx, yy, zz = x * x, y * y, z * z
    return np.stack([
        -0.4886025119029199 * y, 0.4886025119029199 * z, -0.4886025119029199 * x,
        1.0925484305920792 * x * y, -1.0925484305920792 * y * z, 0.31539156525252005 * (2 * zz - xx - yy),
        -1.0925484305920792 * x * z, 0.5462742152960396 * (xx - yy),
        -0.5900435899266435 * y * (3 * xx - yy), 2.890611442640554 * x * y * z,
        -0.4570457994644658 * y * (4 * zz - xx - yy), 0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy),
        -0.4570457994644658 * x * (4 * zz - xx - yy), 1.445305721320277 * z * (xx - yy),
        -0.5900435899266435 * x * (xx - 3 * yy)], 1)[:, :coeffs]


def rotation(R: np.ndarray, coeffs: int = 15) -> np.ndarray:
    """Block-diagonal `[coeffs, coeffs]` matrix M (one Wigner-D block per degree) such that coefficients
    `M @ c` describe the colour function of `c` rotated by `R`: f'(d) = f(Rᵀ d).

    Each degree's functions span a space closed under rotation, so the block is the exact solution of
    `basis(d) M_l = basis(Rᵀ d)` restricted to that degree, solved by least squares over 64 directions.
    """
    d = np.random.default_rng(0).normal(size=(64, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    Y, Yr = basis(d), basis(d @ R)   # row d @ R is Rᵀ d
    M = np.zeros((15, 15))
    for a, b in BANDS:
        M[a:b, a:b] = np.linalg.lstsq(Y[:, a:b], Yr[:, a:b], rcond=None)[0]
    return M[:coeffs, :coeffs]
