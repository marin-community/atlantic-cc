# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Download a whole Common Crawl WARC into memory (as-run ``_decode_one_warc_stream``, download part).

Up to five attempts. HTTP 429/503 sleep ``5 * 2**attempt`` seconds and retry; any other request
error (including a non-2xx status) also sleeps and retries, except on the last attempt. The body is
read in 1 MiB pieces under a 1200 s wall-clock deadline, and a body shorter or longer than the
announced ``Content-Length`` is retried unless the server applied a ``Content-Encoding``. When every
attempt fails, :class:`DownloadError` is raised; phase A records the WARC as an explicit failure
with zero documents.

The Content-Length check was added during the run (between 2026-09-05 and 2026-09-18). WARCs
processed before it (rung 1 and the 2M ladder, shards 0-8079) had no protection against a truncated
HTTP 200, which would have parsed as fewer records.
"""

import logging
import time

import requests

from atlantic_cc.warc.identity import s3_to_https

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 5
RETRY_BASE_DELAY = 5.0  # seconds, doubled after every attempt
RETRYABLE_STATUS_CODES = frozenset({429, 503})
HTTP_TIMEOUT = 600  # requests' connect/read timeout; a read timeout only fires between bytes
DOWNLOAD_DEADLINE = 1200  # wall-clock cap on reading one body (dribbling servers defeat HTTP_TIMEOUT)
READ_CHUNK_BYTES = 1 << 20


class DownloadError(RuntimeError):
    """Every download attempt failed. The message starts with ``Failed to download``, as in production."""


def check_content_length(response: requests.Response, n_bytes: int, warc_path: str) -> None:
    """Raise a retryable error when the body length differs from ``Content-Length``.

    Skipped when ``Content-Encoding`` is set, because ``Content-Length`` then counts encoded bytes.
    """
    if response.headers.get("Content-Encoding"):
        return
    expected = response.headers.get("Content-Length")
    if expected is not None and int(expected) != n_bytes:
        raise requests.exceptions.RequestException(f"truncated download of {warc_path}: {n_bytes} of {expected} bytes")


def _retry_delay(attempt: int) -> float:
    return RETRY_BASE_DELAY * (2**attempt)


def download_warc(warc_path: str) -> bytes:
    """Fetch ``warc_path`` (``s3://commoncrawl/...``) from data.commoncrawl.org and return its bytes."""
    url = s3_to_https(warc_path)
    for attempt in range(MAX_ATTEMPTS):
        try:
            logger.info("Downloading WARC (attempt %d): %s", attempt + 1, warc_path)
            response = requests.get(url, stream=True, timeout=HTTP_TIMEOUT)
            if response.status_code in RETRYABLE_STATUS_CODES:
                delay = _retry_delay(attempt)
                logger.warning("Rate limited (%d) on %s, retry in %.0fs", response.status_code, warc_path, delay)
                time.sleep(delay)
                continue
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            if attempt < MAX_ATTEMPTS - 1:
                delay = _retry_delay(attempt)
                logger.warning(
                    "Download error on %s (attempt %d): %s. Retrying in %.0fs", warc_path, attempt + 1, e, delay
                )
                time.sleep(delay)
                continue
            raise DownloadError(f"Failed to download {warc_path} after {MAX_ATTEMPTS} attempts: {e}") from e
        try:
            buf = bytearray()
            started = time.monotonic()
            for piece in response.iter_content(chunk_size=READ_CHUNK_BYTES):
                buf.extend(piece)
                if time.monotonic() - started > DOWNLOAD_DEADLINE:
                    raise requests.exceptions.RequestException(
                        f"download exceeded {DOWNLOAD_DEADLINE}s ({len(buf)} bytes so far)"
                    )
            check_content_length(response, len(buf), warc_path)
        except requests.exceptions.RequestException as e:
            if attempt < MAX_ATTEMPTS - 1:
                delay = _retry_delay(attempt)
                logger.warning(
                    "Body read failed on %s (attempt %d): %s. Retrying in %.0fs", warc_path, attempt + 1, e, delay
                )
                time.sleep(delay)
                continue
            raise DownloadError(f"Failed to download {warc_path} after {MAX_ATTEMPTS} attempts: {e}") from e
        return bytes(buf)
    raise DownloadError(f"Failed to download {warc_path} after {MAX_ATTEMPTS} attempts")
