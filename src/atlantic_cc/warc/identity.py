# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Identity of a WARC and of its records: ``warc_hash``, ``snapshot``, ``doc_id`` and download URL.

Every function here is part of the release's key space: ``warc_hash`` names the per-WARC output
files, ``doc_id`` is the document key used by dedup and decon, and ``snapshot`` is a release column.
"""

import hashlib
import re

from atlantic_cc import spec

WARC_HASH_LENGTH = 12

# as-run: experiments/baseline_collection/decode_warcs_clean.py:73-75 (final form, deployed with the
# 10-13/10-15 extension shards; earlier bundles matched CC-MAIN only, which named every earlier
# WARC identically).
_SNAPSHOT_RE = re.compile(r"CC-(?:MAIN|SUPPLEMENTAL)-\d{4}-\d{2}")
# CC-NEWS is published daily under crawl-data/CC-NEWS/YYYY/MM/; its snapshot is the month.
_NEWS_SNAPSHOT_RE = re.compile(r"CC-NEWS/(\d{4})/(\d{2})/")

_URN_UUID_PREFIX = "urn:uuid:"


def warc_hash(warc_path: str) -> str:
    """First 12 hex digits of sha256 over the full ``s3://commoncrawl/...`` path."""
    return hashlib.sha256(warc_path.encode()).hexdigest()[:WARC_HASH_LENGTH]


def snapshot_of(warc_path: str) -> str:
    """``CC-NEWS-YYYY-MM`` for CC-NEWS, else the first ``CC-MAIN|SUPPLEMENTAL-YYYY-WW`` in the path, else ""."""
    if m := _NEWS_SNAPSHOT_RE.search(warc_path):
        return f"CC-NEWS-{m.group(1)}-{m.group(2)}"
    m = _SNAPSHOT_RE.search(warc_path)
    return m.group(0) if m else ""


def normalize_record_id(record_id: str) -> str:
    """``<urn:uuid:X>`` -> ``X``: strip whitespace, then the angle brackets, then the ``urn:uuid:`` prefix."""
    rid = record_id.strip()
    if rid.startswith("<") and rid.endswith(">"):
        rid = rid[1:-1]
    if rid.startswith(_URN_UUID_PREFIX):
        rid = rid[len(_URN_UUID_PREFIX) :]
    return rid


def s3_to_https(warc_path: str) -> str:
    """``s3://commoncrawl/<key>`` -> ``https://data.commoncrawl.org/<key>``.

    An ``https://`` URL passes through unchanged and any other string is treated as a bucket key.
    """
    if warc_path.startswith(spec.CC_S3_PREFIX):
        return spec.CC_HTTPS_PREFIX + warc_path[len(spec.CC_S3_PREFIX) :]
    if warc_path.startswith("https://"):
        return warc_path
    return spec.CC_HTTPS_PREFIX + warc_path
