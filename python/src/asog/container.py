"""The `.asog` container (format.md sections 1 and 3.1).

A file is a 64-byte header, a UTF-8 JSON manifest and a payload. The payload holds the streams, each an
`count x channels` integer array stored as optional delta + zigzag coding, split into byte planes and
compressed with zlib, then the optional soundtrack. TRiGS runtime files (`.trigsbin`) use the same header
layout with the magic `TRIGSBIN`, so `read_manifest` reads both.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

MAGIC = b"ASOG"
HEADER_SIZE = 64
PAYLOAD_ALIGN = 64   # payloadOffset is a multiple of this
STREAM_ALIGN = 8     # the reference encoder starts every stream and the soundtrack on 8 bytes


@dataclass(frozen=True)
class Header:
    manifest_offset: int
    manifest_length: int
    payload_offset: int
    payload_length: int


@dataclass
class Stream:
    """One encoded stream: its manifest entry (without byteOffset / byteLength) and its zlib bytes."""

    desc: dict
    data: bytes


def read_manifest(f: BinaryIO, magic: bytes = MAGIC) -> tuple[dict, Header]:
    """Read the header and the JSON manifest of an open file whose first bytes are `magic`."""
    f.seek(0)
    head = f.read(HEADER_SIZE)
    if len(head) < HEADER_SIZE or head[: len(magic)] != magic:
        raise ValueError(f"magic {head[:8]!r}, expected {magic!r}")
    header = Header(*struct.unpack("<QQQQ", head[16:48]))
    f.seek(header.manifest_offset)
    return json.loads(f.read(header.manifest_length)), header


def encode_stream(name: str, values: np.ndarray, *, delta: bool | None = None, level: int = 9) -> Stream:
    """Code an integer array `[count]` or `[count, channels]` as a stream.

    `delta=None` tries both with and without delta coding along the splat order and keeps the smaller.
    Zigzag coding is used exactly when the (delta-coded) values have a negative entry; the number of byte
    planes is the fewest that hold the largest coded value.
    """
    a = np.asarray(values).reshape(len(values), -1).astype(np.int64)
    best: Stream | None = None
    for use_delta in (False, True) if delta is None else (delta,):
        x = a.copy()
        if use_delta:
            x[1:] -= a[:-1]
        signed = bool(x.min() < 0)
        z = ((x << 1) ^ (x >> 63)).astype(np.uint64) if signed else x.astype(np.uint64)
        planes = max(1, (int(z.max()).bit_length() + 7) // 8)
        # plane-major, then channel-major, then splat: raw[(p * channels + c) * count + i]
        raw = b"".join(((z >> np.uint64(8 * p)) & np.uint64(255)).astype(np.uint8).T.tobytes() for p in range(planes))
        data = zlib.compress(raw, level)
        if best is None or len(data) < len(best.data):
            desc = dict(name=name, count=a.shape[0], channels=a.shape[1], planes=planes, delta=use_delta, zigzag=signed)
            best = Stream(desc, data)
    assert best is not None
    return best


def decode_stream(desc: dict, data: bytes) -> np.ndarray:
    """Decode one stream's zlib bytes to an int64 array `[count, channels]`."""
    count, channels, planes = desc["count"], desc["channels"], desc["planes"]
    raw = zlib.decompress(data)
    if len(raw) != count * channels * planes:
        raise ValueError(f"stream {desc['name']}: inflated {len(raw)} bytes, expected {count * channels * planes}")
    bytes_ = np.frombuffer(raw, np.uint8).reshape(planes, channels, count).astype(np.uint64)
    z = np.zeros((channels, count), np.uint64)
    for p in range(planes):
        z |= bytes_[p] << np.uint64(8 * p)
    x = z.astype(np.int64)
    if desc["zigzag"]:
        x = (x >> 1) ^ -(x & 1)
    if desc["delta"]:
        x = np.cumsum(x, axis=1)
    return x.T


def write(path: str | Path, manifest: dict, streams: list[Stream], audio: tuple[bytes, str] | None = None) -> int:
    """Write an `.asog` file: `manifest` gets its `streams` (and `audio`) entries filled in.

    `audio` is the sound file's bytes and its MIME type. Returns the file size in bytes.
    """
    payload = bytearray()
    entries = []
    for s in streams:
        payload += bytes(-len(payload) % STREAM_ALIGN)
        entries.append(dict(s.desc, byteOffset=len(payload), byteLength=len(s.data)))
        payload += s.data
    manifest = dict(manifest)
    if audio is not None:
        sound, mime = audio
        payload += bytes(-len(payload) % STREAM_ALIGN)
        manifest["audio"] = dict(mime=mime, byteOffset=len(payload), byteLength=len(sound))
        payload += sound
    manifest["streams"] = entries
    mj = json.dumps(manifest, separators=(",", ":")).encode()
    payload_offset = HEADER_SIZE + len(mj)
    payload_offset += -payload_offset % PAYLOAD_ALIGN
    head = MAGIC + bytes(12) + struct.pack("<QQQQ", HEADER_SIZE, len(mj), payload_offset, len(payload))
    head += bytes(HEADER_SIZE - len(head))
    with open(path, "wb") as f:
        f.write(head)
        f.write(mj)
        f.write(bytes(payload_offset - HEADER_SIZE - len(mj)))
        f.write(payload)
    return payload_offset + len(payload)


def read(path: str | Path) -> tuple[dict, dict[str, np.ndarray], bytes | None]:
    """Read an `.asog` file: manifest, every stream decoded (name -> `[count, channels]` int64), soundtrack bytes."""
    with open(path, "rb") as f:
        manifest, header = read_manifest(f)
        if manifest.get("format") != "asog" or manifest.get("version") != 1:
            raise ValueError(f"{path}: manifest format {manifest.get('format')!r} v{manifest.get('version')}, expected asog v1")
        streams = {}
        for desc in manifest["streams"]:
            f.seek(header.payload_offset + desc["byteOffset"])
            streams[desc["name"]] = decode_stream(desc, f.read(desc["byteLength"]))
        audio = None
        if "audio" in manifest:
            f.seek(header.payload_offset + manifest["audio"]["byteOffset"])
            audio = f.read(manifest["audio"]["byteLength"])
    return manifest, streams, audio
