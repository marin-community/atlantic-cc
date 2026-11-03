# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The explicit crawl list of the Atlantic-CC v1.0 input, and where Common Crawl lists each crawl's WARCs.

"All of Common Crawl" is not a reproducible definition (new crawls keep appearing), so v1.0 is pinned to
the crawls below. Every WARC of each listed crawl is included, except for the open CC-NEWS month
(October 2026), which stops at :data:`NEWS_LAST_WARC`, the newest file listed when it was added.
"""

from enum import StrEnum

from atlantic_cc.spec import CC_HTTPS_PREFIX

# The 119 CC-MAIN crawls with a WARC listing on 2026-04-10, when the base worklist was frozen
# (index.commoncrawl.org/collinfo.json listed 122; CC-MAIN-2008-2009, 2009-2010 and 2012 have no WARCs).
BASE_MAIN_CRAWLS = (
    "CC-MAIN-2013-20",
    "CC-MAIN-2013-48",
    "CC-MAIN-2014-10",
    "CC-MAIN-2014-15",
    "CC-MAIN-2014-23",
    "CC-MAIN-2014-35",
    "CC-MAIN-2014-41",
    "CC-MAIN-2014-42",
    "CC-MAIN-2014-49",
    "CC-MAIN-2014-52",
    "CC-MAIN-2015-06",
    "CC-MAIN-2015-11",
    "CC-MAIN-2015-14",
    "CC-MAIN-2015-18",
    "CC-MAIN-2015-22",
    "CC-MAIN-2015-27",
    "CC-MAIN-2015-32",
    "CC-MAIN-2015-35",
    "CC-MAIN-2015-40",
    "CC-MAIN-2015-48",
    "CC-MAIN-2016-07",
    "CC-MAIN-2016-18",
    "CC-MAIN-2016-22",
    "CC-MAIN-2016-26",
    "CC-MAIN-2016-30",
    "CC-MAIN-2016-36",
    "CC-MAIN-2016-40",
    "CC-MAIN-2016-44",
    "CC-MAIN-2016-50",
    "CC-MAIN-2017-04",
    "CC-MAIN-2017-09",
    "CC-MAIN-2017-13",
    "CC-MAIN-2017-17",
    "CC-MAIN-2017-22",
    "CC-MAIN-2017-26",
    "CC-MAIN-2017-30",
    "CC-MAIN-2017-34",
    "CC-MAIN-2017-39",
    "CC-MAIN-2017-43",
    "CC-MAIN-2017-47",
    "CC-MAIN-2017-51",
    "CC-MAIN-2018-05",
    "CC-MAIN-2018-09",
    "CC-MAIN-2018-13",
    "CC-MAIN-2018-17",
    "CC-MAIN-2018-22",
    "CC-MAIN-2018-26",
    "CC-MAIN-2018-30",
    "CC-MAIN-2018-34",
    "CC-MAIN-2018-39",
    "CC-MAIN-2018-43",
    "CC-MAIN-2018-47",
    "CC-MAIN-2018-51",
    "CC-MAIN-2019-04",
    "CC-MAIN-2019-09",
    "CC-MAIN-2019-13",
    "CC-MAIN-2019-18",
    "CC-MAIN-2019-22",
    "CC-MAIN-2019-26",
    "CC-MAIN-2019-30",
    "CC-MAIN-2019-35",
    "CC-MAIN-2019-39",
    "CC-MAIN-2019-43",
    "CC-MAIN-2019-47",
    "CC-MAIN-2019-51",
    "CC-MAIN-2020-05",
    "CC-MAIN-2020-10",
    "CC-MAIN-2020-16",
    "CC-MAIN-2020-24",
    "CC-MAIN-2020-29",
    "CC-MAIN-2020-34",
    "CC-MAIN-2020-40",
    "CC-MAIN-2020-45",
    "CC-MAIN-2020-50",
    "CC-MAIN-2021-04",
    "CC-MAIN-2021-10",
    "CC-MAIN-2021-17",
    "CC-MAIN-2021-21",
    "CC-MAIN-2021-25",
    "CC-MAIN-2021-31",
    "CC-MAIN-2021-39",
    "CC-MAIN-2021-43",
    "CC-MAIN-2021-49",
    "CC-MAIN-2022-05",
    "CC-MAIN-2022-21",
    "CC-MAIN-2022-27",
    "CC-MAIN-2022-33",
    "CC-MAIN-2022-40",
    "CC-MAIN-2022-49",
    "CC-MAIN-2023-06",
    "CC-MAIN-2023-14",
    "CC-MAIN-2023-23",
    "CC-MAIN-2023-40",
    "CC-MAIN-2023-50",
    "CC-MAIN-2024-10",
    "CC-MAIN-2024-18",
    "CC-MAIN-2024-22",
    "CC-MAIN-2024-26",
    "CC-MAIN-2024-30",
    "CC-MAIN-2024-33",
    "CC-MAIN-2024-38",
    "CC-MAIN-2024-42",
    "CC-MAIN-2024-46",
    "CC-MAIN-2024-51",
    "CC-MAIN-2025-05",
    "CC-MAIN-2025-08",
    "CC-MAIN-2025-13",
    "CC-MAIN-2025-18",
    "CC-MAIN-2025-21",
    "CC-MAIN-2025-26",
    "CC-MAIN-2025-30",
    "CC-MAIN-2025-33",
    "CC-MAIN-2025-38",
    "CC-MAIN-2025-43",
    "CC-MAIN-2025-47",
    "CC-MAIN-2025-51",
    "CC-MAIN-2026-04",
    "CC-MAIN-2026-08",
    "CC-MAIN-2026-12",
)
# Added 2026-10-13: the CC-MAIN crawls released after the April freeze.
LATE_MAIN_CRAWLS = (
    "CC-MAIN-2026-17",
    "CC-MAIN-2026-21",
    "CC-MAIN-2026-25",
    "CC-MAIN-2026-30",
    "CC-MAIN-2026-34",
    "CC-MAIN-2026-39",
)
# Added 2026-10-13: Common Crawl's science-focused crawl. It has no public warc.paths.gz, so its
# 4,573 WARC paths ship with this package (data/v1.0.supplemental.warc.paths.gz).
SUPPLEMENTAL_CRAWL = "CC-SUPPLEMENTAL-2026-22"
# Added 2026-10-15: every CC-NEWS month from the first (2016-08) to the one open at the time.
NEWS_FIRST_MONTH = (2016, 8)
NEWS_LAST_MONTH = (2026, 10)
NEWS_LAST_WARC = "crawl-data/CC-NEWS/2026/10/CC-NEWS-20261015151252-00272.warc.gz"


class Family(StrEnum):
    MAIN = "CC-MAIN"
    NEWS = "CC-NEWS"
    SUPPLEMENTAL = "CC-SUPPLEMENTAL"


def family_of(crawl: str) -> Family:
    for family in Family:
        if crawl.startswith(f"{family}-"):
            return family
    raise ValueError(f"unknown crawl family: {crawl!r}")


def news_months(first: tuple[int, int], last: tuple[int, int]) -> list[str]:
    """``CC-NEWS-YYYY-MM`` names from *first* to *last* (inclusive)."""
    months = []
    year, month = first
    while (year, month) <= last:
        months.append(f"CC-NEWS-{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


NEWS_CRAWLS = tuple(news_months(NEWS_FIRST_MONTH, NEWS_LAST_MONTH))
V1_0_CRAWLS = BASE_MAIN_CRAWLS + LATE_MAIN_CRAWLS + (SUPPLEMENTAL_CRAWL,) + NEWS_CRAWLS


def listing_url(crawl: str) -> str:
    """Common Crawl's public ``warc.paths.gz`` for a CC-MAIN crawl or a CC-NEWS month."""
    family = family_of(crawl)
    if family is Family.MAIN:
        return f"{CC_HTTPS_PREFIX}crawl-data/{crawl}/warc.paths.gz"
    if family is Family.NEWS:
        year, month = crawl.removeprefix("CC-NEWS-").split("-")
        return f"{CC_HTTPS_PREFIX}crawl-data/CC-NEWS/{year}/{month}/warc.paths.gz"
    raise ValueError(f"{crawl} has no public warc.paths.gz; its paths come from the frozen worklist")
