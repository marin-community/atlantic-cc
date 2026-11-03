# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""U-SEL (worklist part): WARC identity and production's shard layout."""

import hashlib

import pytest

from atlantic_cc.warc.identity import warc_hash
from atlantic_cc.worklist import layout
from atlantic_cc.worklist.frozen import supplemental_paths
from atlantic_cc.worklist.identity import hash_shard, to_relative, to_s3_url

# (s3 URL, production's warc_hash), from production's worklist shards 0 and 32,000.
KNOWN = [
    (
        "s3://commoncrawl/crawl-data/CC-MAIN-2013-48/segments/1386164021936/warc/"
        "CC-MAIN-20131204133341-00077-ip-10-33-133-15.ec2.internal.warc.gz",
        "33baad0028fc",
    ),
    (
        "s3://commoncrawl/crawl-data/CC-MAIN-2014-10/segments/1393999636575/warc/"
        "CC-MAIN-20140305060716-00024-ip-10-183-142-35.ec2.internal.warc.gz",
        "e667e3002b51",
    ),
    (
        "s3://commoncrawl/projects/cc-open-athena-test/CC-SUPPLEMENTAL-2026-22/segments/20260521074042/warc/"
        "CC-SUPPLEMENTAL-2026-22-20260521074116-20260521084116-00000.warc.gz",
        "5a4bec92b936",
    ),
]
# sha256(paths)[:16] of production's 19 CC-SUPPLEMENTAL shards (32,000-32,018).
SUPPLEMENTAL_SHARDS = [
    "fecc8a1197b2494f", "ae699df28943dc38", "207930d89ba2d308", "322a664609503082", "388865d95932acde",
    "39c6306ec7e67132", "a6e6a130dc8bedff", "722ed77789be64e5", "4027247df533351a", "e54b3693daefae1f",
    "d7af7358be1f9ca0", "378ed395ea70dc3a", "994923e69944fdc5", "b3626a4dcd89dad9", "d7fff86dd159debd",
    "21c168e0c808e797", "93cf39f1dc4c85ec", "a6e1e1185b336a56", "b9f1f2bcd958274f",
]  # fmt: skip


@pytest.mark.parametrize(("url", "expected"), KNOWN)
def test_warc_hash_of_the_s3_url_matches_production_worklist(url, expected):
    assert warc_hash(url) == expected


def test_relative_paths_round_trip():
    url = KNOWN[0][0]
    assert to_s3_url(to_relative(url)) == url
    with pytest.raises(ValueError, match="not a Common Crawl S3 URL"):
        to_relative("https://data.commoncrawl.org/crawl-data/x.warc.gz")


def test_base_shards_follow_the_hash_modulus():
    # Both WARCs were in production's shard 0.
    assert [hash_shard(url, layout.BASE_SHARDS) for url, _ in KNOWN[:2]] == [0, 0]


def test_round_robin_deals_sorted_paths():
    shards = layout.round_robin(["e", "a", "d", "c", "b", "g", "f"], warcs_per_shard=3)
    assert shards == [["a", "d", "g"], ["b", "e"], ["c", "f"]]


def test_supplemental_extension_reproduces_production_shards():
    shards = layout.round_robin(supplemental_paths(), layout.EXTENSION_WARCS_PER_SHARD)
    digests = [hashlib.sha256("".join(f"{p}\n" for p in sorted(s)).encode()).hexdigest()[:16] for s in shards]
    assert digests == SUPPLEMENTAL_SHARDS


def test_paths_outside_v1_0_crawls_have_no_layout():
    with pytest.raises(ValueError, match="not in a v1.0 crawl"):
        layout.v1_0_layout(["crawl-data/CC-MAIN-2026-43/segments/1/warc/x.warc.gz"])
