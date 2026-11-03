# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The object-store operations every multi-step stage is built on.

Coordination between workers reduces to four guarantees a :class:`Store` gives:

* ``create`` succeeds for exactly one writer, and nobody ever reads a partially written object;
* ``replace`` succeeds only against the generation the caller last saw, so a stale claim has exactly
  one taker and a worker that lost its claim learns so on its next heartbeat;
* a ranged read can be pinned to a generation, so a reader never mixes bytes of two writes;
* a listing either returns every object under a prefix or raises. A failed listing must never read
  as "no objects".

:class:`~atlantic_cc.io.local.LocalStore` gives all four on a POSIX filesystem and
:class:`~atlantic_cc.io.gcs.GcsStore` on Google Cloud Storage. :class:`~atlantic_cc.io.fsspec_store.FsspecStore`
covers any other fsspec URL (``s3://``, ``memory://``, ...) without the conditional-write guarantees,
which single-host runs do not need. :func:`atlantic_cc.io.resolve.store_for` picks the store for a path.
"""

import dataclasses
import json
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol


@dataclasses.dataclass(frozen=True)
class ObjectInfo:
    """Metadata for one object, as a listing or stat reports it."""

    path: str
    generation: int
    updated: float
    size: int


class PreconditionFailed(Exception):
    """A generation precondition did not hold: another writer got there first."""


class Store(Protocol):
    """The storage operations the pipelines need. Paths are full URLs (``gs://b/k``) or local paths."""

    def read(self, path: str, start: int | None = None, end: int | None = None, generation: int | None = None) -> bytes:
        """Bytes ``[start, end)`` of ``path``; with ``generation``, fail unless the object is that write."""
        ...

    def write(self, path: str, data: bytes | Iterable[bytes]) -> int:
        """Overwrite ``path`` unconditionally and atomically; return the new generation."""
        ...

    def create(self, path: str, data: bytes) -> int:
        """Create ``path`` only if absent; raise :class:`PreconditionFailed` if it exists."""
        ...

    def replace(self, path: str, data: bytes, generation: int) -> int:
        """Overwrite ``path`` only if its generation is ``generation``; else :class:`PreconditionFailed`."""
        ...

    def stat(self, path: str) -> ObjectInfo | None:
        """Metadata for ``path``, or ``None`` if it does not exist."""
        ...

    def list(self, prefix: str) -> list[ObjectInfo]:
        """Every object whose path starts with ``prefix``, sorted by path. Raises on any listing error."""
        ...

    def delete(self, path: str) -> None:
        """Delete ``path`` if present."""
        ...

    def now(self) -> float:
        """Current time on the clock that ``ObjectInfo.updated`` is measured against."""
        ...


def read_json(store: Store, path: str) -> dict:
    return json.loads(store.read(path))


def dumps(payload: dict) -> bytes:
    """Canonical JSON bytes: sorted keys, as every marker and result file is written."""
    return json.dumps(payload, sort_keys=True).encode()


def delete_prefix(store: Store, prefix: str, threads: int = 32) -> int:
    """Delete every object under ``prefix``; returns how many were deleted."""
    objects = store.list(prefix)
    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(lambda info: store.delete(info.path), objects))
    return len(objects)
