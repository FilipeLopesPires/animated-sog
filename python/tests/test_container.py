import struct

import numpy as np
import pytest

from asog import container


@pytest.mark.parametrize("delta", [None, False, True])
@pytest.mark.parametrize("values", [
    np.arange(1000) % 7,                                                       # small, unsigned
    np.random.default_rng(1).integers(-300, 300, (777, 3)),                    # signed, 2 planes
    np.cumsum(np.random.default_rng(2).integers(0, 50, (500, 5)), axis=0),    # smooth: delta wins
    np.random.default_rng(3).integers(-(2 ** 40), 2 ** 40, (64, 45)),          # 6 planes, many channels
    np.zeros((10, 1), np.int64),
])
def test_stream_round_trip(values, delta):
    s = container.encode_stream("x", values, delta=delta)
    out = container.decode_stream(s.desc, s.data)
    assert out.shape == (len(values), 1 if values.ndim == 1 else values.shape[1])
    np.testing.assert_array_equal(out, values.reshape(len(values), -1))
    if delta is not None:
        assert s.desc["delta"] == delta


def test_stream_layout_is_plane_then_channel_then_splat():
    values = np.array([[0x0102, 0x0304], [0x0506, 0x0708]])
    s = container.encode_stream("x", values, delta=False)
    assert s.desc == dict(name="x", count=2, channels=2, planes=2, delta=False, zigzag=False)
    import zlib
    assert zlib.decompress(s.data) == bytes([0x02, 0x06, 0x04, 0x08, 0x01, 0x05, 0x03, 0x07])


def test_zigzag_only_for_negative_values():
    assert not container.encode_stream("x", np.arange(10), delta=False).desc["zigzag"]
    s = container.encode_stream("x", np.array([0, -1, 1, -2]), delta=False)
    assert s.desc["zigzag"] and s.desc["planes"] == 1


def test_file_round_trip(tmp_path):
    a = np.random.default_rng(0).integers(-1000, 1000, (300, 3))
    b = np.arange(300)
    streams = [container.encode_stream("a", a), container.encode_stream("b", b)]
    path = tmp_path / "x.asog"
    size = container.write(path, {"format": "asog", "version": 1, "count": 300}, streams, (b"ID3sound", "audio/mpeg"))
    raw = path.read_bytes()
    assert len(raw) == size
    assert raw[:4] == b"ASOG" and raw[4:16] == bytes(12) and raw[48:64] == bytes(16)
    mo, ml, po, pl = struct.unpack("<QQQQ", raw[16:48])
    assert mo == 64 and po % 64 == 0 and po >= mo + ml and po + pl == size
    manifest, decoded, audio = container.read(path)
    np.testing.assert_array_equal(decoded["a"], a)
    np.testing.assert_array_equal(decoded["b"][:, 0], b)
    assert audio == b"ID3sound" and manifest["audio"]["mime"] == "audio/mpeg"
    offsets = [s["byteOffset"] for s in manifest["streams"]] + [manifest["audio"]["byteOffset"]]
    assert offsets == sorted(offsets) and all(o % 8 == 0 for o in offsets)


def test_rejects_other_files(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"TRIGSBIN" + bytes(56))
    with pytest.raises(ValueError, match="magic"):
        container.read(path)


def test_rejects_short_stream():
    s = container.encode_stream("x", np.arange(10))
    with pytest.raises(ValueError, match="expected"):
        container.decode_stream(dict(s.desc, count=11), s.data)
