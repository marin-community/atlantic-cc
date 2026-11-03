# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small fsspec helpers for the release tools: local paths, ``gs://``, ``s3://`` and ``https://``.

Paths keep the form they were given in: a glob under ``gs://bucket/x`` returns ``gs://`` paths and a
glob under a local directory returns absolute local paths. Roots are normalized once, at the command
line, by :func:`normalize_root`.
"""

import json
import os
from typing import Any

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq


def has_protocol(path: str) -> bool:
    return "://" in path


def normalize_root(path: str) -> str:
    """Absolute local path or URL, without a trailing slash."""
    path = path.rstrip("/")
    return path if has_protocol(path) else os.path.abspath(path)


def _unstrip(fs: fsspec.AbstractFileSystem, path: str, keep_protocol: bool) -> str:
    return fs.unstrip_protocol(path) if keep_protocol else path


def glob(pattern: str) -> list[str]:
    fs, stripped = fsspec.core.url_to_fs(pattern)
    return [_unstrip(fs, p, has_protocol(pattern)) for p in fs.glob(stripped)]


def glob_sizes(pattern: str) -> list[tuple[str, int]]:
    """``(path, size)`` for every match, from one listing (no per-file requests)."""
    fs, stripped = fsspec.core.url_to_fs(pattern)
    return [
        (_unstrip(fs, p, has_protocol(pattern)), int(info["size"]))
        for p, info in fs.glob(stripped, detail=True).items()
    ]


def exists(path: str) -> bool:
    fs, stripped = fsspec.core.url_to_fs(path)
    return fs.exists(stripped)


def open_file(path: str, mode: str, **kwargs: Any):
    """An fsspec ``OpenFile``; local writes create missing parent directories."""
    if "w" in mode and not has_protocol(path):
        kwargs["auto_mkdir"] = True
    return fsspec.open(path, mode, **kwargs)


def read_bytes(path: str) -> bytes:
    with open_file(path, "rb") as f:
        return f.read()


def write_bytes(path: str, data: bytes) -> None:
    with open_file(path, "wb") as f:
        f.write(data)


def read_json(path: str) -> Any:
    with open_file(path, "r") as f:
        return json.load(f)


def read_parquet(path: str, **kwargs: Any) -> pa.Table:
    with open_file(path, "rb") as f:
        return pq.read_table(f, **kwargs)


def parquet_rows(path: str) -> int:
    with open_file(path, "rb") as f:
        return pq.ParquetFile(f).metadata.num_rows
