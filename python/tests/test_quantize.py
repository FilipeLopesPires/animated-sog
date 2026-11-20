import sys

import numpy as np
import pytest

from asog import quantize


def test_codebook1d_maps_to_nearest_entry():
    x = np.random.default_rng(0).normal(size=(5000, 3))
    cb, idx = quantize.codebook1d(x)
    assert len(cb) == 256 and np.all(np.diff(cb) >= 0)
    np.testing.assert_array_equal(idx, np.abs(x.reshape(-1, 1) - cb[None]).argmin(1))


def test_palette_size():
    assert quantize.palette_size(1000, 65536) == 64
    assert quantize.palette_size(105000, 65536) == 4096
    assert quantize.palette_size(10_000_000, 65536) == 65536


@pytest.fixture
def no_cuda(monkeypatch):
    """Make the palette believe there is no CUDA PyTorch."""
    monkeypatch.setitem(sys.modules, "torch", None)


def test_palette_numpy_fallback(no_cuda):
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(16, 45))
    x = (centres[rng.integers(0, 16, 2000)] + rng.normal(0, 0.01, (2000, 45))).astype(np.float32)
    c, labels = quantize.sh_palette(x, 64)
    assert c.shape == (64, 45) and labels.shape == (2000,)
    err = np.abs(c[labels] - x).mean()
    assert err < 0.02
    np.testing.assert_array_equal(labels, ((x[:, None] - c[None].astype(np.float32)) ** 2).sum(-1).argmin(1))


def test_palette_without_gpu_refuses_large_inputs(no_cuda, monkeypatch):
    monkeypatch.setattr(quantize, "NUMPY_PALETTE_LIMIT", 1000)
    with pytest.raises(RuntimeError, match="needs PyTorch with CUDA"):
        quantize.sh_palette(np.zeros((100, 45), np.float32), 64)
