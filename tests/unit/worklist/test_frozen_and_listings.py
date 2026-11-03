# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The frozen v1.0 worklist: packaged digests, paths files, and rebuilding from Common Crawl's listings."""

import gzip

import pytest

from atlantic_cc.worklist import frozen, listings
from atlantic_cc.worklist.crawls import (
    NEWS_LAST_WARC,
    SUPPLEMENTAL_CRAWL,
    V1_0_CRAWLS,
    Family,
    family_of,
    listing_url,
    news_months,
)
from atlantic_cc.worklist.frozen import PathsDigest
from atlantic_cc.worklist.layout import V1_0_SHARDS, layout_digest, v1_0_layout

NEWS_OCT = "crawl-data/CC-NEWS/2026/10/CC-NEWS-{}.warc.gz"


def test_packaged_crawl_table_covers_every_v1_0_warc():
    table = frozen.v1_0_crawl_digests()
    assert set(table) == set(V1_0_CRAWLS)
    assert sum(d.n_warcs for d in table.values()) == frozen.V1_0_N_WARCS
    by_family: dict[str, int] = {}
    for crawl, d in table.items():
        by_family[family_of(crawl)] = by_family.get(family_of(crawl), 0) + d.n_warcs
    assert by_family == frozen.V1_0_FAMILY_COUNTS


def test_packaged_supplemental_paths_match_the_crawl_table():
    paths = frozen.supplemental_paths()
    assert frozen.paths_digest(paths) == frozen.v1_0_crawl_digests()[SUPPLEMENTAL_CRAWL]


def test_v1_0_crawls_are_125_main_123_news_months_and_supplemental():
    families = [family_of(c) for c in V1_0_CRAWLS]
    assert (families.count(Family.MAIN), families.count(Family.NEWS), families.count(Family.SUPPLEMENTAL)) == (
        125,
        123,
        1,
    )
    assert news_months((2025, 11), (2026, 2)) == [
        "CC-NEWS-2025-11",
        "CC-NEWS-2025-12",
        "CC-NEWS-2026-01",
        "CC-NEWS-2026-02",
    ]


def test_listing_urls():
    assert listing_url("CC-MAIN-2013-20") == "https://data.commoncrawl.org/crawl-data/CC-MAIN-2013-20/warc.paths.gz"
    assert listing_url("CC-NEWS-2016-08") == "https://data.commoncrawl.org/crawl-data/CC-NEWS/2016/08/warc.paths.gz"
    with pytest.raises(ValueError, match="no public warc.paths.gz"):
        listing_url(SUPPLEMENTAL_CRAWL)


def test_paths_files_round_trip_with_a_stable_gzip(tmp_path):
    paths = ["crawl-data/b.warc.gz", "crawl-data/a.warc.gz"]
    frozen.write_paths_file(paths, str(tmp_path / "x.paths.gz"))
    frozen.write_paths_file(paths, str(tmp_path / "y.paths.gz"))
    assert frozen.read_paths_file(str(tmp_path / "x.paths.gz")) == paths
    assert (tmp_path / "x.paths.gz").read_bytes() == (tmp_path / "y.paths.gz").read_bytes()
    assert gzip.decompress((tmp_path / "x.paths.gz").read_bytes()) == b"crawl-data/b.warc.gz\ncrawl-data/a.warc.gz\n"


def test_loading_a_list_that_is_not_the_frozen_one_fails(tmp_path):
    frozen.write_paths_file(["crawl-data/a.warc.gz"], str(tmp_path / "x.paths.gz"))
    with pytest.raises(ValueError, match="expected"):
        frozen.load_frozen(str(tmp_path / "x.paths.gz"))


def test_the_open_news_month_is_cut_at_its_last_listed_warc():
    month = [NEWS_OCT.format(f"2026101{d}000000-0000{d}") for d in range(1, 6)]
    assert listings.cut_news_month(month, month[2]) == month[:3]
    with pytest.raises(ValueError, match="not in its month's listing"):
        listings.cut_news_month(month[:2], month[2])


def test_build_takes_supplemental_from_the_package_and_cuts_only_the_cut_month():
    listing = {
        "CC-NEWS-2026-09": ["crawl-data/CC-NEWS/2026/09/CC-NEWS-20260901000000-00000.warc.gz"],
        "CC-NEWS-2026-10": [
            NEWS_OCT.format("20261001000000-00000"),
            NEWS_LAST_WARC,
            NEWS_OCT.format("20261020000000-00300"),
        ],
    }
    paths = listings.build_worklist(
        ["CC-NEWS-2026-09", "CC-NEWS-2026-10", SUPPLEMENTAL_CRAWL], NEWS_LAST_WARC, listing.get
    )
    assert len(paths) == 1 + 2 + 4_573
    assert NEWS_OCT.format("20261020000000-00300") not in paths
    assert paths == sorted(paths)


def test_build_refuses_duplicates_and_unlisted_supplemental_crawls():
    with pytest.raises(ValueError, match="duplicate WARC paths"):
        listings.build_worklist(["CC-NEWS-2026-09", "CC-NEWS-2026-08"], None, lambda c: ["crawl-data/same.warc.gz"])
    with pytest.raises(ValueError, match="no public listing"):
        listings.build_worklist(["CC-SUPPLEMENTAL-2027-01"], None, lambda c: [])


def test_a_changed_listing_is_reported_by_crawl():
    table = frozen.v1_0_crawl_digests()
    paths = frozen.supplemental_paths()[:-1]
    assert listings.mismatched_crawls(paths) == sorted(set(table))
    assert table[SUPPLEMENTAL_CRAWL] != frozen.paths_digest(paths)


@pytest.mark.network
def test_public_listings_match_the_packaged_digests():
    for crawl in ("CC-NEWS-2016-08", "CC-MAIN-2026-17"):
        assert frozen.paths_digest(sorted(listings.fetch_listing(crawl))) == frozen.v1_0_crawl_digests()[crawl]
    october = listings.cut_news_month(listings.fetch_listing("CC-NEWS-2026-10"), NEWS_LAST_WARC)
    assert frozen.paths_digest(sorted(october)) == frozen.v1_0_crawl_digests()["CC-NEWS-2026-10"]


@pytest.mark.network
@pytest.mark.slow
def test_rebuild_from_public_listings_gives_the_frozen_list_and_layout():
    paths = listings.rebuild_v1_0()  # raises unless every crawl and the whole list match
    shards = v1_0_layout(paths)
    assert len(shards) == V1_0_SHARDS
    assert frozen.paths_digest(paths) == PathsDigest(frozen.V1_0_N_WARCS, frozen.V1_0_SHA256)
    assert layout_digest(shards) == frozen.V1_0_LAYOUT_SHA256
