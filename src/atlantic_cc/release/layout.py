# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The published file list, derived from the manifest.

A crawl's files are fully named by its per-kind file counts, so the manifest alone gives every key,
in the order the paths files list them (crawls sorted; each crawl's contaminated file before its
parts).
"""

from collections.abc import Collection, Sequence
from dataclasses import dataclass

from atlantic_cc.release.params import PUBLISHED_PREFIX, Kind, part_file_name


@dataclass(frozen=True)
class ReleaseFile:
    crawl: str
    kind: Kind
    path: str  # relative to the release root, e.g. v1.0/crawl=CC-NEWS-2016-08/part-00000-of-00001.parquet

    @property
    def key(self) -> str:
        """The object key in the Common Crawl bucket."""
        return f"{PUBLISHED_PREFIX}/{self.path}"


def release_files(manifest: dict, crawls: Collection[str] | None, kinds: Sequence[Kind]) -> list[ReleaseFile]:
    """Every file of the selected crawls (all when *crawls* is None) and kinds, in paths-file order."""
    version = manifest["version"]
    unknown = set(crawls or ()) - manifest["crawls"].keys()
    if unknown:
        raise ValueError(f"no such crawls in the {version} manifest: {sorted(unknown)}")
    files = []
    for crawl, entry in sorted(manifest["crawls"].items()):
        if crawls is not None and crawl not in crawls:
            continue
        for kind in sorted(kinds):
            n = entry[kind]["files"]
            files += [
                ReleaseFile(crawl, kind, f"{version}/crawl={crawl}/{part_file_name(kind, i, n)}") for i in range(n)
            ]
    return files
