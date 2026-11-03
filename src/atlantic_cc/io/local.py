# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

""":class:`~atlantic_cc.io.store.Store` on a POSIX filesystem: a local disk, or a shared one for several hosts."""

import contextlib
import fcntl
import os
import time
import uuid
from collections.abc import Callable, Iterable, Iterator

from atlantic_cc.io.store import ObjectInfo, PreconditionFailed

# Temporary files live beside their target (same filesystem, so the final rename is atomic); listings skip them.
_TMP_MARK = ".__tmp__."
# A rewrite moves the generation (the mtime) forward by at least this much, so it changes even on filesystems
# that keep only microseconds.
_MIN_GENERATION_STEP_NS = 1000
# os.pread may return fewer bytes than asked for very large requests; read in slices of at most this size.
_PREAD_SLICE = 1 << 30


class LocalStore:
    """Objects as files under a POSIX filesystem.

    * ``write`` puts the payload in a hidden temporary file beside the target, fsyncs it and renames it over the
      target: a reader sees the old object or the new one, never a mix.
    * ``create`` hard-links the finished temporary file (created with ``O_EXCL``) to the target name. ``link`` fails
      if the name exists, so exactly one creator wins and no reader ever sees a partial object.
    * ``replace`` is a compare-and-swap under an ``flock`` on the parent directory.
    * The generation of an object is its mtime in nanoseconds. Every write stamps a time strictly after the previous
      generation of the path, so a rewrite always changes it. Ranged reads use ``pread`` on a descriptor whose
      ``fstat`` matched the requested generation, so a pinned read never mixes two writes.

    Args:
        clock: Seconds since the epoch, for ``now`` and the stamped mtimes. Tests pass a fake clock to age claims.
    """

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock

    def _clock_ns(self) -> int:
        return time.time_ns() if self._clock is None else int(self._clock() * 1e9)

    def _stamp(self, path: str, previous: int) -> int:
        generation = max(self._clock_ns(), previous + _MIN_GENERATION_STEP_NS)
        os.utime(path, ns=(generation, generation))
        return generation

    def _write_temporary(self, path: str, data: bytes | Iterable[bytes]) -> str:
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        tmp = os.path.join(directory, f".{os.path.basename(path)}{_TMP_MARK}{os.getpid()}.{uuid.uuid4().hex[:12]}")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            pieces = [data] if isinstance(data, bytes | bytearray | memoryview) else data
            for piece in pieces:
                view = memoryview(piece)
                while view:
                    view = view[os.write(fd, view) :]
            os.fsync(fd)
        except BaseException:
            os.close(fd)
            os.unlink(tmp)
            raise
        os.close(fd)
        return tmp

    @staticmethod
    def _generation(path: str) -> int | None:
        try:
            return os.stat(path).st_mtime_ns
        except FileNotFoundError:
            return None

    def read(self, path: str, start: int | None = None, end: int | None = None, generation: int | None = None) -> bytes:
        if start is not None and end is not None and end <= start:
            return b""
        fd = os.open(path, os.O_RDONLY)
        try:
            info = os.fstat(fd)
            if generation is not None and info.st_mtime_ns != generation:
                raise PreconditionFailed(path)
            lo = 0 if start is None else start
            hi = info.st_size if end is None else min(end, info.st_size)
            buffer = bytearray(max(0, hi - lo))
            view = memoryview(buffer)
            position = 0
            while position < len(buffer):
                got = os.preadv(fd, [view[position : position + _PREAD_SLICE]], lo + position)
                if got == 0:
                    raise OSError(f"{path}: unexpected end of file at byte {lo + position}")
                position += got
            return bytes(buffer)
        finally:
            os.close(fd)

    def write(self, path: str, data: bytes | Iterable[bytes]) -> int:
        tmp = self._write_temporary(path, data)
        generation = self._stamp(tmp, self._generation(path) or 0)
        os.replace(tmp, path)
        return generation

    def create(self, path: str, data: bytes) -> int:
        tmp = self._write_temporary(path, data)
        try:
            generation = self._stamp(tmp, 0)
            try:
                os.link(tmp, path)
            except FileExistsError as e:
                raise PreconditionFailed(path) from e
        finally:
            os.unlink(tmp)
        return generation

    @contextlib.contextmanager
    def _directory_locked(self, path: str) -> Iterator[None]:
        fd = os.open(os.path.dirname(path) or ".", os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def replace(self, path: str, data: bytes, generation: int) -> int:
        tmp = self._write_temporary(path, data)
        try:
            with self._directory_locked(path):
                if self._generation(path) != generation:
                    raise PreconditionFailed(path)
                new_generation = self._stamp(tmp, generation)
                os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)
            raise
        return new_generation

    def stat(self, path: str) -> ObjectInfo | None:
        try:
            info = os.stat(path)
        except FileNotFoundError:
            return None
        return ObjectInfo(path, info.st_mtime_ns, info.st_mtime_ns / 1e9, info.st_size)

    def list(self, prefix: str) -> list[ObjectInfo]:
        directory = prefix if prefix.endswith(os.sep) else os.path.dirname(prefix)
        walk_root = directory or os.curdir
        if not os.path.isdir(walk_root):
            return []
        out = []

        def fail(error: OSError) -> None:
            raise error

        for root, _dirs, files in os.walk(walk_root, onerror=fail):
            for name in files:
                full = os.path.join(root, name) if directory else os.path.relpath(os.path.join(root, name))
                if _TMP_MARK in name or not full.startswith(prefix):
                    continue
                info = self.stat(full)
                if info is not None:
                    out.append(info)
        return sorted(out, key=lambda info: info.path)

    def delete(self, path: str) -> None:
        with contextlib.suppress(FileNotFoundError):
            os.remove(path)

    def now(self) -> float:
        return time.time() if self._clock is None else self._clock()
