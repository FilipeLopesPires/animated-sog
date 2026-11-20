"""The real 1-second Rubloff clip: the package must reproduce what the arrival.space player already reads.

test-data/rubloff-1s.asog was made by the original encoder script with default options. The source
.trigsbin lives outside the repo (tests skip without it). The SH palette's k-means runs on the GPU and is
not bit-reproducible, so the SH streams are compared by decoded value, the rest exactly.
"""

from pathlib import Path

import numpy as np
import pytest
from conftest import TEST_DATA

import asog
from asog import container
from asog.cli import main

REFERENCE = TEST_DATA / "rubloff-1s.asog"
SOURCE = Path("C:/Dev/arrival.space/tmp/rubloff/runs/s5_1s_a/rubloff_1s.trigsbin")
LAYOUT = ("name", "count", "channels", "planes", "delta", "zigzag")
SH_STREAMS = ("shlabels", "shpalette")


def test_info_on_reference(capsys):
    main(["info", str(REFERENCE)])
    out = capsys.readouterr().out
    assert "splats 105000 (0 static, 105000 dynamic)" in out
    assert "timeline 1 s at 30 fps (30 frames), 1 clip(s)" in out
    assert "SH: 15 coefficients, palette of 4096" in out
    for name in ("means", "quats", "omega", "g", "flags"):
        assert f"\n  {name} " in out


@pytest.mark.data
@pytest.mark.skipif(not SOURCE.exists(), reason="Rubloff source .trigsbin not present")
def test_encode_reproduces_reference(tmp_path):
    out = tmp_path / "rubloff-1s.asog"
    stats = asog.encode(SOURCE, out)
    m_new, s_new, _ = container.read(out)
    m_ref, s_ref, _ = container.read(REFERENCE)
    assert [{k: s[k] for k in LAYOUT} for s in m_new["streams"]] == [{k: s[k] for k in LAYOUT} for s in m_ref["streams"]]
    for name in s_ref:
        if name not in SH_STREAMS:
            np.testing.assert_array_equal(s_new[name], s_ref[name], err_msg=name)
    for key in ("count", "timeline", "source", "means", "scales", "sh0", "mu", "logsigma", "motion"):
        assert m_new[key] == m_ref[key], key
    for c_new, c_ref in zip(m_new["clips"], m_ref["clips"], strict=True):
        assert {k: v for k, v in c_ref.items() if k != "flags"} == c_new   # the source's flag tensors are no longer copied
    a_new, a_ref = asog.read(out), asog.read(REFERENCE)
    assert np.abs(a_new.shN - a_ref.shN).mean() < 0.1 * np.abs(a_ref.shN).mean()
    assert abs(stats.bytes - REFERENCE.stat().st_size) < 0.01 * REFERENCE.stat().st_size
    f_new, f_ref = a_new.evaluate(0.5), a_ref.evaluate(0.5)
    np.testing.assert_allclose(f_new.positions, f_ref.positions, atol=1e-12)
    np.testing.assert_allclose(f_new.weights, f_ref.weights, atol=1e-12)
