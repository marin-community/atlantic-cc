# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The release checksums file: ``md5sum`` lines for every published object.

Format (``md5sum -c`` reads it from the release root)::

    <32 hex md5>  v1.0/crawl=CC-MAIN-2013-20/contaminated-00000-of-00001.parquet
    <32 hex md5>  v1.0.manifest.json

Paths are relative to the release root (``projects/atlantic-cc/``) and sorted. The file is built from
the upload done-markers (``{"key", "bytes", "etag"}``, one per uploaded object): every object went up
in a single PUT, so its S3 ETag is the MD5 of its bytes and no data has to be re-read.
"""

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import fsspec

from atlantic_cc.release import storage
from atlantic_cc.release.params import PUBLISHED_PREFIX

MD5_RE = re.compile(r"[0-9a-f]{32}")
HASH_CHUNK_BYTES = 8 << 20


@dataclass(frozen=True)
class UploadRecord:
    key: str  # bucket key, e.g. projects/atlantic-cc/v1.0.manifest.json
    bytes: int
    etag: str


def etag_md5(etag: str) -> str:
    """The MD5 behind a single-part S3 ETag (``"<hex>"``); raises on multipart or KMS ETags."""
    value = etag.strip('"')
    if not MD5_RE.fullmatch(value):
        raise ValueError(f"ETag {etag!r} is not the MD5 of a single-part upload")
    return value


def release_relative(key: str) -> str:
    prefix = f"{PUBLISHED_PREFIX}/"
    if not key.startswith(prefix):
        raise ValueError(f"{key} is not under {prefix}")
    return key.removeprefix(prefix)


def read_done_markers(done_root: str, concurrency: int) -> list[UploadRecord]:
    """Every upload done-marker under *done_root* (``<done_root>/<key>.json``)."""
    fs, base = fsspec.core.url_to_fs(done_root)
    paths = sorted(fs.glob(f"{base}/**/*.json"))
    blobs = fs.cat(paths, batch_size=concurrency) if fs.async_impl else fs.cat(paths)
    records = []
    for path in paths:
        marker = json.loads(blobs[path])
        if marker["key"] != path.removeprefix(f"{base}/").removesuffix(".json"):
            raise ValueError(f"{path} records key {marker['key']}")
        records.append(UploadRecord(marker["key"], marker["bytes"], marker["etag"]))
    return records


def checksums_from_uploads(records: Iterable[UploadRecord]) -> dict[str, str]:
    """``release-relative path -> md5`` from upload records; raises on duplicates or non-MD5 ETags."""
    md5s: dict[str, str] = {}
    for record in records:
        path = release_relative(record.key)
        if path in md5s:
            raise ValueError(f"two upload records for {record.key}")
        md5s[path] = etag_md5(record.etag)
    return md5s


def format_checksums(md5s: dict[str, str]) -> str:
    return "".join(f"{md5s[path]}  {path}\n" for path in sorted(md5s))


def parse_checksums(text: str) -> dict[str, str]:
    """``path -> md5`` from ``md5sum`` output (text ``"  "`` or binary ``" *"`` separators)."""
    md5s = {}
    for line in text.splitlines():
        if not line:
            continue
        md5, sep, path = line[:32], line[32:34], line[34:]
        if not MD5_RE.fullmatch(md5) or sep not in ("  ", " *") or not path:
            raise ValueError(f"not an md5sum line: {line!r}")
        md5s[path] = md5
    return md5s


def file_md5(path: str) -> str:
    digest = hashlib.md5()
    with storage.open_file(path, "rb") as f:
        while chunk := f.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def missing_uploads(records: Sequence[UploadRecord], expected_keys: Iterable[str]) -> list[str]:
    have = {r.key for r in records}
    return sorted(k for k in expected_keys if k not in have)
