# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a worklist from Common Crawl's public ``warc.paths.gz`` listings for an explicit crawl list.

Listings of finished crawls never change; a CC-NEWS month's listing grows until the month ends, so a
list that includes the current month must say where it stops (``news_last_warc``). CC-SUPPLEMENTAL-2026-22
has no public listing (its WARCs are public, under ``projects/cc-open-athena-test/``), so its paths come
from the frozen v1.0 list, which ships them in this package.
"""

import gzip
import logging
import os
import time
from collections.abc import Callable, Sequence

import requests

from atlantic_cc.warc.identity import snapshot_of
from atlantic_cc.worklist.crawls import NEWS_LAST_WARC, SUPPLEMENTAL_CRAWL, V1_0_CRAWLS, Family, family_of, listing_url
from atlantic_cc.worklist.frozen import (
    V1_0_N_WARCS,
    V1_0_SHA256,
    PathsDigest,
    check_digest,
    crawl_digests,
    parse_paths,
    supplemental_paths,
    v1_0_crawl_digests,
)

logger = logging.getLogger(__name__)

HTTP_ATTEMPTS = 8
HTTP_TIMEOUT_SECONDS = 120
HTTP_BACKOFF_MAX_SECONDS = 60
RETRY_STATUS = {429, 500, 502, 503, 504}


def fetch_bytes(url: str, session: requests.Session | None = None) -> bytes:
    """GET a small object, retrying throttling and transient errors (data.commoncrawl.org answers 503 when busy)."""
    http = session or requests
    for attempt in range(HTTP_ATTEMPTS):
        try:
            response = http.get(url, timeout=HTTP_TIMEOUT_SECONDS)
            if response.status_code not in RETRY_STATUS:
                response.raise_for_status()
                return response.content
            logger.warning("GET %s: HTTP %d (attempt %d)", url, response.status_code, attempt + 1)
        except (requests.ConnectionError, requests.Timeout) as e:
            logger.warning("GET %s: %s (attempt %d)", url, e, attempt + 1)
        time.sleep(min(2**attempt, HTTP_BACKOFF_MAX_SECONDS))
    raise RuntimeError(f"GET {url} failed {HTTP_ATTEMPTS} times")


def fetch_listing(crawl: str) -> list[str]:
    """The bucket-relative WARC paths of a CC-MAIN crawl or CC-NEWS month, in listing order."""
    return parse_paths(gzip.decompress(fetch_bytes(listing_url(crawl))))


def cut_news_month(paths: Sequence[str], last_warc: str) -> list[str]:
    """The month's WARCs up to and including *last_warc* (CC-NEWS file names sort by timestamp)."""
    if last_warc not in paths:
        raise ValueError(f"{last_warc} is not in its month's listing")
    last_name = os.path.basename(last_warc)
    return [p for p in paths if os.path.basename(p) <= last_name]


def build_worklist(
    crawls: Sequence[str],
    news_last_warc: str | None,
    fetch: Callable[[str], list[str]] = fetch_listing,
) -> list[str]:
    """Sorted bucket-relative paths of every WARC of *crawls*.

    Args:
        crawls: crawl names: ``CC-MAIN-YYYY-WW``, ``CC-NEWS-YYYY-MM``, or ``CC-SUPPLEMENTAL-2026-22``.
        news_last_warc: last WARC to include from its CC-NEWS month, or None to take whole months.
        fetch: crawl -> listing paths (the public listing by default).
    """
    paths: list[str] = []
    for crawl in crawls:
        if family_of(crawl) is Family.SUPPLEMENTAL:
            if crawl != SUPPLEMENTAL_CRAWL:
                raise ValueError(f"{crawl} has no public listing and is not part of the frozen v1.0 list")
            got = supplemental_paths()
        else:
            got = fetch(crawl)
            if news_last_warc is not None and snapshot_of(news_last_warc) == crawl:
                got = cut_news_month(got, news_last_warc)
        logger.info("%s: %d WARCs", crawl, len(got))
        paths += got
    if len(set(paths)) != len(paths):
        raise ValueError(f"{len(paths) - len(set(paths))} duplicate WARC paths across the listings")
    return sorted(paths)


def mismatched_crawls(paths: Sequence[str]) -> list[str]:
    """Crawls whose paths differ (count or digest) from the frozen v1.0 list."""
    want = v1_0_crawl_digests()
    got = crawl_digests(paths)
    return sorted(c for c in want.keys() | got.keys() if want.get(c) != got.get(c))


def rebuild_v1_0(fetch: Callable[[str], list[str]] = fetch_listing) -> list[str]:
    """Rebuild the frozen v1.0 list from the public listings and check it crawl by crawl."""
    paths = build_worklist(V1_0_CRAWLS, NEWS_LAST_WARC, fetch)
    bad = mismatched_crawls(paths)
    if bad:
        raise ValueError(f"{len(bad)} crawls differ from the frozen v1.0 list: {bad[:10]}")
    check_digest(paths, PathsDigest(V1_0_N_WARCS, V1_0_SHA256), "rebuilt v1.0 worklist")
    return paths
