# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Production's shard layout of the v1.0 worklist (34,628 shards).

Shards were the unit of work for extraction; they do not affect any output except the row order of
the release's internal WARC path map. The layout was built once and then extended three times:

* shards 0-31,999: the 119 base CC-MAIN crawls, ``shard = int(sha256(url)[:8], 16) % 32000``;
* each extension sorted its new WARCs by URL and dealt them round-robin into
  ``ceil(n / 250)`` new shards appended after the last one: CC-SUPPLEMENTAL-2026-22 (shards
  32,000-32,018), the six late 2026 CC-MAIN crawls (32,019-34,418), and CC-NEWS (34,419-34,627).

Each shard lists its WARCs sorted by URL.
"""

import hashlib
from collections.abc import Iterable, Sequence
from enum import StrEnum

from atlantic_cc.warc.identity import snapshot_of
from atlantic_cc.worklist.crawls import BASE_MAIN_CRAWLS, LATE_MAIN_CRAWLS, SUPPLEMENTAL_CRAWL, Family, family_of
from atlantic_cc.worklist.identity import hash_shard, to_s3_url

BASE_SHARDS = 32_000
EXTENSION_WARCS_PER_SHARD = 250
V1_0_SHARDS = 34_628


class LayoutGroup(StrEnum):
    BASE = "base"
    SUPPLEMENTAL = "supplemental"
    LATE_MAIN = "late_main"
    NEWS = "news"


# First shard of each extension, in the order production appended them.
V1_0_EXTENSION_FIRST_SHARDS = {
    LayoutGroup.SUPPLEMENTAL: 32_000,
    LayoutGroup.LATE_MAIN: 32_019,
    LayoutGroup.NEWS: 34_419,
}

_BASE = frozenset(BASE_MAIN_CRAWLS)
_LATE_MAIN = frozenset(LATE_MAIN_CRAWLS)


def layout_group(relative_path: str) -> LayoutGroup:
    crawl = snapshot_of(relative_path)
    if crawl in _BASE:
        return LayoutGroup.BASE
    if crawl == SUPPLEMENTAL_CRAWL:
        return LayoutGroup.SUPPLEMENTAL
    if crawl in _LATE_MAIN:
        return LayoutGroup.LATE_MAIN
    if crawl and family_of(crawl) is Family.NEWS:
        return LayoutGroup.NEWS
    raise ValueError(f"{relative_path} is not in a v1.0 crawl")


def round_robin(paths: list[str], warcs_per_shard: int) -> list[list[str]]:
    """Production's ``extend``: deal the sorted paths into ``ceil(n / warcs_per_shard)`` shards."""
    n_new = -(-len(paths) // warcs_per_shard)
    shards: list[list[str]] = [[] for _ in range(n_new)]
    for i, path in enumerate(sorted(paths)):
        shards[i % n_new].append(path)
    return shards


def v1_0_layout(relative_paths: Iterable[str]) -> list[list[str]]:
    """The 34,628 production shards of the v1.0 worklist, each sorted, in shard order.

    Paths are bucket-relative; the shard formula hashes the ``s3://commoncrawl/`` URL, and sorting by
    URL or by relative path gives the same order.
    """
    groups: dict[LayoutGroup, list[str]] = {group: [] for group in LayoutGroup}
    for path in relative_paths:
        groups[layout_group(path)].append(path)
    shards: list[list[str]] = [[] for _ in range(BASE_SHARDS)]
    for path in groups[LayoutGroup.BASE]:
        shards[hash_shard(to_s3_url(path), BASE_SHARDS)].append(path)
    for group, first in V1_0_EXTENSION_FIRST_SHARDS.items():
        if len(shards) != first:
            raise ValueError(f"the {group} extension would start at shard {len(shards)}; production used {first}")
        shards += round_robin(groups[group], EXTENSION_WARCS_PER_SHARD)
    if len(shards) != V1_0_SHARDS:
        raise ValueError(f"{len(shards)} shards; production had {V1_0_SHARDS}")
    return [sorted(shard) for shard in shards]


def layout_digest(shards: Sequence[Sequence[str]]) -> str:
    r"""sha256 of ``"<shard>\t<path>\n"`` lines in shard order (pinned as ``frozen.V1_0_LAYOUT_SHA256``)."""
    return hashlib.sha256("".join(f"{s}\t{p}\n" for s, paths in enumerate(shards) for p in paths).encode()).hexdigest()
