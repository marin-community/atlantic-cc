# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bucket-relative WARC paths and production's base shard formula.

Worklists and the release store paths relative to the Common Crawl bucket, as Common Crawl's own
``warc.paths.gz`` listings do. Production hashed the full ``s3://commoncrawl/...`` URL, both for
``atlantic_cc.warc.identity.warc_hash`` and for the shard formula, so convert with :func:`to_s3_url`
before hashing. Crawl names come from ``atlantic_cc.warc.identity.snapshot_of``.
"""

import hashlib

from atlantic_cc.spec import CC_S3_PREFIX


def to_s3_url(relative_path: str) -> str:
    return CC_S3_PREFIX + relative_path


def to_relative(s3_url: str) -> str:
    """``s3://commoncrawl/<path>`` -> ``<path>``; raises on any other prefix."""
    if not s3_url.startswith(CC_S3_PREFIX):
        raise ValueError(f"not a Common Crawl S3 URL: {s3_url!r}")
    return s3_url.removeprefix(CC_S3_PREFIX)


def hash_shard(s3_url: str, n_shards: int) -> int:
    """Production's frozen shard formula: ``int(sha256(url)[:8], 16) % n_shards``."""
    return int(hashlib.sha256(s3_url.encode()).hexdigest()[:8], 16) % n_shards
