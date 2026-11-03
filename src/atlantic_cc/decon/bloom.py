# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bloom filters in dupekit's format: a pure-Python reader and a loader for the native one.

File layout (``dupekit.Bloom.save_bytes``, an rbloom derivative): 8-byte little-endian ``k``, then the
bit array; bit ``i`` is byte ``i >> 3``, mask ``1 << (i & 7)``. Indexes come from a 128-bit LCG seeded
with the item: ``state = state * MUL + 1 (mod 2**128)``, ``index = (state >> 32) as u64 % bits``,
repeated ``k`` times. Items are the uint64 feature hashes of :func:`atlantic_cc.decon.matching._bloom_hash`
(always pass them as non-negative ints: dupekit sign-extends negative values, so an int64 view of a
hash probes different bits).
"""

import hashlib
import math
import mmap
import os
import tempfile
from collections.abc import Iterator

import dupekit_native
import fsspec

LCG_MULTIPLIER = 47026247687942121848144207491837418733
_MASK128 = (1 << 128) - 1
_MASK64 = (1 << 64) - 1
_HEADER_BYTES = 8
_COPY_CHUNK_BYTES = 64 * 1024 * 1024


def bloom_parameters(expected_items: int, false_positive_rate: float) -> tuple[int, int]:
    """``(k, size_in_bits)`` of ``dupekit.Bloom(expected_items, false_positive_rate)``."""
    size_in_bits = -expected_items * math.log(false_positive_rate) / (math.log(2) ** 2)
    k = int((size_in_bits / expected_items) * math.log(2))
    bits = int(size_in_bits)
    whole_bytes = bits // 8 + (1 if bits % 8 else 0)
    return k, whole_bytes * 8


def bloom_indexes(item: int, k: int, size_in_bits: int) -> Iterator[int]:
    """Bit positions probed for *item* (a uint64 hash)."""
    state = item & _MASK128
    for _ in range(k):
        state = (state * LCG_MULTIPLIER + 1) & _MASK128
        yield ((state >> 32) & _MASK64) % size_in_bits


class PyBloom:
    """Read-only bloom filter over dupekit's serialized bytes (no native dependency).

    About 10 microseconds per lookup; use it for verification and small inputs, and
    :func:`load_native_bloom` for whole crawls.
    """

    def __init__(self, data: bytes | memoryview | mmap.mmap) -> None:
        view = memoryview(data)
        if len(view) < _HEADER_BYTES:
            raise ValueError("invalid bloom filter data")
        self.k = int.from_bytes(view[:_HEADER_BYTES], "little")
        self._bits = view[_HEADER_BYTES:]
        self.size_in_bits = len(self._bits) * 8

    @classmethod
    def open(cls, local_path: str) -> "PyBloom":
        """Memory-map a local bloom file."""
        with open(local_path, "rb") as fh:
            return cls(mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ))

    def __contains__(self, item: int) -> bool:
        bits = self._bits
        return all(bits[i >> 3] & (1 << (i & 7)) for i in bloom_indexes(item, self.k, self.size_in_bits))


def local_copy(path: str, cache_dir: str | None = None) -> str:
    """Copy *path* (any fsspec URL) to a node-local file once and return the local path.

    Local paths are returned unchanged. The cache key is the sha256 of the URL, so a URL must name
    immutable contents (as-run behavior of ``decon_v4._load_bloom``).
    """
    if "://" not in path:
        return path
    cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "atlantic-cc-decon-bloom")
    local_path = os.path.join(cache_dir, hashlib.sha256(path.encode()).hexdigest())
    if os.path.exists(local_path):
        return local_path
    os.makedirs(cache_dir, exist_ok=True)
    partial_path = f"{local_path}.{os.getpid()}.partial"
    with fsspec.open(path, "rb") as src, open(partial_path, "wb") as dst:
        while chunk := src.read(_COPY_CHUNK_BYTES):
            dst.write(chunk)
    os.replace(partial_path, local_path)
    return local_path


def load_native_bloom(path: str, cache_dir: str | None = None) -> dupekit_native.Bloom:
    """Load a bloom file (local or any fsspec URL) into ``dupekit_native.Bloom``."""
    return dupekit_native.Bloom.load(local_copy(path, cache_dir))
