from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import synthetic

import asog

REPO = Path(__file__).resolve().parents[2]
TEST_DATA = REPO / "test-data"


@pytest.fixture(scope="session")
def synthetic_source(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("synthetic") / "clip.trigsbin"
    synthetic.make_trigsbin(path)
    return path


@pytest.fixture(scope="session")
def synthetic_asog(synthetic_source: Path) -> Path:
    path = synthetic_source.with_suffix(".asog")
    asog.encode(synthetic_source, path)
    return path
