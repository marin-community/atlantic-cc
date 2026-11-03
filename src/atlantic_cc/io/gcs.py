# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

""":class:`~atlantic_cc.io.store.Store` on Google Cloud Storage, with generation preconditions.

Needs the ``gcs`` extra (``google-cloud-storage``).
"""

import os
import threading
import time
from collections.abc import Iterable

import requests
from google.api_core import exceptions as gexc
from google.cloud import storage
from google.cloud.storage.retry import DEFAULT_RETRY, DEFAULT_RETRY_IF_GENERATION_SPECIFIED

from atlantic_cc.io.store import ObjectInfo, PreconditionFailed

# Uploads of iterables stream through a resumable session in chunks of this many bytes, so a multi-GB shuffle
# object never needs a second in-memory copy.
_UPLOAD_CHUNK_BYTES = 64 * 1024 * 1024
_TIMEOUT = (30, 600)
# Connections per client: enough for the 48 ranged-read threads of a shuffle reader plus uploads.
_POOL_SIZE = 128
_LIST_PAGE_SIZE = 5000


def _split_gs(path: str) -> tuple[str, str]:
    if not path.startswith("gs://"):
        raise ValueError(f"not a gs:// path: {path!r}")
    bucket, _, key = path.removeprefix("gs://").partition("/")
    return bucket, key


class GcsStore:
    """Objects in GCS. ``create`` is ``ifGenerationMatch=0``, ``replace`` is ``ifGenerationMatch=<generation>``,
    and a ranged read with a generation is pinned to that object version.

    The ``storage.Client`` is not fork-safe, so each process builds its own on first use.

    Args:
        project: Billing / quota project for the client; ``None`` takes the default from the environment.
    """

    def __init__(self, project: str | None = None) -> None:
        self._project = project
        self._client: storage.Client | None = None
        self._pid = -1
        self._lock = threading.Lock()

    def _get_client(self) -> storage.Client:
        with self._lock:
            if self._client is None or self._pid != os.getpid():
                client = storage.Client(project=self._project)
                adapter = requests.adapters.HTTPAdapter(pool_connections=_POOL_SIZE, pool_maxsize=_POOL_SIZE)
                client._http.mount("https://", adapter)
                self._client = client
                self._pid = os.getpid()
            return self._client

    def _blob(self, path: str) -> storage.Blob:
        bucket, key = _split_gs(path)
        return self._get_client().bucket(bucket).blob(key)

    def read(self, path: str, start: int | None = None, end: int | None = None, generation: int | None = None) -> bytes:
        if start is not None and end is not None and end <= start:
            return b""
        return self._blob(path).download_as_bytes(
            start=start,
            # google-cloud-storage takes an INCLUSIVE end offset.
            end=None if end is None else end - 1,
            if_generation_match=generation,
            timeout=_TIMEOUT,
            retry=DEFAULT_RETRY,
            checksum=None if start is not None or end is not None else "md5",
        )

    def write(self, path: str, data: bytes | Iterable[bytes]) -> int:
        blob = self._blob(path)
        if isinstance(data, bytes | bytearray | memoryview):
            # Payloads are deterministic, so retrying an unconditional upload is safe.
            blob.upload_from_string(bytes(data), timeout=_TIMEOUT, retry=DEFAULT_RETRY)
            return int(blob.generation)
        blob.chunk_size = _UPLOAD_CHUNK_BYTES
        with blob.open("wb", retry=DEFAULT_RETRY, ignore_flush=True) as handle:
            for piece in data:
                handle.write(piece)
        blob.reload()
        return int(blob.generation)

    def create(self, path: str, data: bytes) -> int:
        blob = self._blob(path)
        try:
            blob.upload_from_string(
                data, if_generation_match=0, timeout=_TIMEOUT, retry=DEFAULT_RETRY_IF_GENERATION_SPECIFIED
            )
        except gexc.PreconditionFailed as e:
            raise PreconditionFailed(path) from e
        return int(blob.generation)

    def replace(self, path: str, data: bytes, generation: int) -> int:
        blob = self._blob(path)
        try:
            blob.upload_from_string(
                data, if_generation_match=generation, timeout=_TIMEOUT, retry=DEFAULT_RETRY_IF_GENERATION_SPECIFIED
            )
        except gexc.PreconditionFailed as e:
            raise PreconditionFailed(path) from e
        return int(blob.generation)

    def stat(self, path: str) -> ObjectInfo | None:
        blob = self._blob(path)
        try:
            blob.reload(timeout=_TIMEOUT, retry=DEFAULT_RETRY)
        except gexc.NotFound:
            return None
        return ObjectInfo(path, int(blob.generation), blob.updated.timestamp(), int(blob.size))

    def list(self, prefix: str) -> list[ObjectInfo]:
        bucket, key = _split_gs(prefix)
        blobs = self._get_client().list_blobs(
            bucket,
            prefix=key,
            fields="items(name,generation,updated,size),nextPageToken",
            page_size=_LIST_PAGE_SIZE,
            timeout=_TIMEOUT,
            retry=DEFAULT_RETRY,
        )
        return [
            ObjectInfo(f"gs://{bucket}/{b.name}", int(b.generation), b.updated.timestamp(), int(b.size)) for b in blobs
        ]

    def delete(self, path: str) -> None:
        try:
            self._blob(path).delete(timeout=_TIMEOUT, retry=DEFAULT_RETRY)
        except gexc.NotFound:
            pass

    def now(self) -> float:
        return time.time()
