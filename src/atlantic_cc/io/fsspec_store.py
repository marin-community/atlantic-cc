# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

""":class:`~atlantic_cc.io.store.Store` on any fsspec filesystem (``s3://``, ``az://``, ``memory://``, ...)."""

import datetime
import time
from collections.abc import Iterable

import fsspec

from atlantic_cc.io.store import ObjectInfo, PreconditionFailed

# fsspec exposes no portable object version, so every object of this store is generation 0.
GENERATION = 0
# Fields different fsspec backends use for an object's modification time.
_TIME_FIELDS = ("mtime", "LastModified", "updated", "last_modified", "created")


class FsspecStore:
    """Objects behind one fsspec protocol.

    fsspec has no portable conditional writes or object versions, so this store gives weaker guarantees than
    :class:`~atlantic_cc.io.local.LocalStore` and :class:`~atlantic_cc.io.gcs.GcsStore`: ``create`` checks for the
    object before writing (atomic only within one process), ``replace`` is unsupported, and reads cannot be pinned
    to a generation. That is enough for inputs, outputs and the single-host local runner, whose writes are
    deterministic; the multi-host claims runner needs a store with real preconditions.
    """

    def __init__(self, protocol: str) -> None:
        self.protocol = protocol
        self._fs = fsspec.filesystem(protocol)

    def _key(self, path: str) -> str:
        """``path`` without its protocol, as fsspec reports names; a trailing slash is kept."""
        key = self._fs._strip_protocol(path)
        return key + "/" if path.endswith("/") and not key.endswith("/") else key

    def read(self, path: str, start: int | None = None, end: int | None = None, generation: int | None = None) -> bytes:
        if generation not in (None, GENERATION):
            raise PreconditionFailed(path)
        if start is not None and end is not None and end <= start:
            return b""
        return self._fs.cat_file(path, start=start, end=end)

    def write(self, path: str, data: bytes | Iterable[bytes]) -> int:
        with self._fs.open(path, "wb") as handle:
            for piece in [data] if isinstance(data, bytes | bytearray | memoryview) else data:
                handle.write(piece)
        return GENERATION

    def create(self, path: str, data: bytes) -> int:
        if self._fs.exists(path):
            raise PreconditionFailed(path)
        return self.write(path, data)

    def replace(self, path: str, data: bytes, generation: int) -> int:
        raise NotImplementedError(f"{self.protocol}:// has no conditional writes; use a local or gs:// run root")

    def stat(self, path: str) -> ObjectInfo | None:
        try:
            info = self._fs.info(path)
        except FileNotFoundError:
            return None
        if info["type"] != "file":
            return None
        return ObjectInfo(path, GENERATION, _updated(info), int(info["size"]))

    def list(self, prefix: str) -> list[ObjectInfo]:
        key_prefix = self._key(prefix)
        directory = key_prefix if key_prefix.endswith("/") else key_prefix.rsplit("/", 1)[0]
        try:
            found = self._fs.find(directory, detail=True)
        except FileNotFoundError:
            return []
        out = [
            ObjectInfo(self._fs.unstrip_protocol(name), GENERATION, _updated(info), int(info["size"]))
            for name, info in found.items()
            if info["type"] == "file" and name.startswith(key_prefix)
        ]
        return sorted(out, key=lambda info: info.path)

    def delete(self, path: str) -> None:
        try:
            self._fs.rm_file(path)
        except FileNotFoundError:
            pass

    def now(self) -> float:
        return time.time()


def _updated(info: dict) -> float:
    """Modification time from whichever field the backend reports, or 0 when it reports none."""
    for key in _TIME_FIELDS:
        value = info.get(key)
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, datetime.datetime):
            return value.timestamp()
        if isinstance(value, str):
            return datetime.datetime.fromisoformat(value).timestamp()
    return 0.0
