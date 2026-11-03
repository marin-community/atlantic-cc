# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Download the release (or some crawls of it) from ``https://data.commoncrawl.org/``, resumably.

Files land under ``<dest>/`` at their release-relative paths (``v1.0/crawl=<crawl>/...``) next to the
manifest, so ``<dest>/v1.0`` can be read as a hive-partitioned dataset and ``md5sum -c`` works from
``<dest>``. A file is downloaded to ``<name>.part`` (resumed with HTTP range requests after an
interruption), checked against its MD5, and only then renamed; finished files are skipped on a rerun.
"""

import json
import logging
import os
import threading
import time
from collections.abc import Collection, Sequence
from concurrent.futures import ThreadPoolExecutor

import requests

from atlantic_cc.release.checksums import MD5_RE, file_md5
from atlantic_cc.release.layout import ReleaseFile, release_files
from atlantic_cc.release.params import Kind, manifest_name
from atlantic_cc.worklist.listings import (
    HTTP_ATTEMPTS,
    HTTP_BACKOFF_MAX_SECONDS,
    HTTP_TIMEOUT_SECONDS,
    RETRY_STATUS,
    fetch_bytes,
)

logger = logging.getLogger(__name__)

STREAM_CHUNK_BYTES = 1 << 20

_session = threading.local()


def _http() -> requests.Session:
    if not hasattr(_session, "value"):
        _session.value = requests.Session()
    return _session.value


def _md5_from_etag(etag: str | None) -> str | None:
    value = (etag or "").strip('"')
    return value if MD5_RE.fullmatch(value) else None


def _stream_to(url: str, part: str) -> str | None:
    """Append the rest of *url* to *part* (range request when part exists); returns the object's ETag MD5."""
    offset = os.path.getsize(part) if os.path.exists(part) else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with _http().get(url, headers=headers, stream=True, timeout=HTTP_TIMEOUT_SECONDS) as response:
        if response.status_code == 416:  # nothing left to fetch
            return _md5_from_etag(_http().head(url, timeout=HTTP_TIMEOUT_SECONDS).headers.get("ETag"))
        if response.status_code in RETRY_STATUS:
            raise requests.ConnectionError(f"HTTP {response.status_code}")
        response.raise_for_status()
        mode = "ab" if response.status_code == 206 else "wb"  # 200: the server ignored the range
        with open(part, mode) as f:
            for chunk in response.iter_content(STREAM_CHUNK_BYTES):
                f.write(chunk)
        return _md5_from_etag(response.headers.get("ETag"))


def download_file(url: str, dest: str, expected_md5: str | None) -> int:
    """Download *url* to *dest* unless it is already there; returns its size (0 if it was there)."""
    if os.path.exists(dest):
        return 0
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = f"{dest}.part"
    for attempt in range(HTTP_ATTEMPTS):
        try:
            etag_md5 = _stream_to(url, part)
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as e:
            logger.warning("%s: %s (attempt %d); resuming", url, e, attempt + 1)
            time.sleep(min(2**attempt, HTTP_BACKOFF_MAX_SECONDS))
            continue
        want = expected_md5 or etag_md5
        got = file_md5(part)
        if want is not None and got != want:
            os.remove(part)
            raise ValueError(f"{url}: md5 {got}, expected {want}")
        size = os.path.getsize(part)
        os.replace(part, dest)
        return size
    raise RuntimeError(f"{url}: gave up after {HTTP_ATTEMPTS} attempts")


def download(
    base_url: str,
    dest: str,
    crawls: Collection[str] | None,
    kinds: Sequence[Kind],
    workers: int,
    checksums: dict[str, str] | None,
    version: str,
) -> dict:
    """Fetch the manifest and the selected files; returns a summary."""
    manifest_bytes = fetch_bytes(f"{base_url}/{manifest_name(version)}", _http())
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, manifest_name(version)), "wb") as f:
        f.write(manifest_bytes)
    files = release_files(json.loads(manifest_bytes), crawls, kinds)
    logger.info("downloading %d files to %s", len(files), dest)

    def fetch(f: ReleaseFile) -> int:
        expected = checksums[f.path] if checksums is not None else None
        return download_file(f"{base_url}/{f.path}", os.path.join(dest, f.path), expected)

    total = done = 0
    with ThreadPoolExecutor(workers) as ex:
        for n_bytes in ex.map(fetch, files):
            total += n_bytes
            done += 1
            if done % 20 == 0 or done == len(files):
                logger.info("%d/%d files, %.2f GB fetched", done, len(files), total / 1e9)
    return {"files": len(files), "bytes_fetched": total, "dest": dest}
