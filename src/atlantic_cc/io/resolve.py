# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pick the :class:`~atlantic_cc.io.store.Store` that serves a path or URL."""

from atlantic_cc.io.fsspec_store import FsspecStore
from atlantic_cc.io.local import LocalStore
from atlantic_cc.io.store import Store

_STORES: dict[str, Store] = {}


def scheme_of(path: str) -> str:
    """The URL scheme of ``path`` (``"gs"``, ``"s3"``, ...), or ``""`` for a local path."""
    return path.split("://", 1)[0] if "://" in path else ""


def store_for(path: str) -> Store:
    """GCS for ``gs://``, the local filesystem for plain paths, fsspec for any other URL.

    One store per scheme and process; each is safe to share between threads and rebuilds its clients after a fork.
    """
    scheme = scheme_of(path)
    if scheme not in _STORES:
        _STORES[scheme] = _new_store(scheme)
    return _STORES[scheme]


def _new_store(scheme: str) -> Store:
    if scheme == "":
        return LocalStore()
    if scheme == "gs":
        from atlantic_cc.io.gcs import GcsStore  # noqa: PLC0415 - google-cloud-storage is the optional `gcs` extra

        return GcsStore()
    return FsspecStore(scheme)
