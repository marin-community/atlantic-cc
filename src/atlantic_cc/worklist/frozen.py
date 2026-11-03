# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The frozen v1.0 input worklist: 8,582,115 WARC paths, and the files that pin it.

The list is published as ``v1.0.input.warc.paths.gz`` in Common Crawl's paths-file format: one
bucket-relative path per line (``crawl-data/...`` or ``projects/...``), here sorted. Its identity is
the sha256 of the *uncompressed* text, since gzip bytes vary with the zlib build.

Two small files ship with this package so the list can be rebuilt and checked without downloading it:

* ``data/v1.0.crawls.tsv``: per crawl, the WARC count and the sha256 of its sorted paths;
* ``data/v1.0.supplemental.warc.paths.gz``: the 4,573 CC-SUPPLEMENTAL-2026-22 paths, which Common
  Crawl does not list publicly.
"""

import csv
import gzip
import hashlib
import io
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib import resources

import fsspec

from atlantic_cc.warc.identity import snapshot_of

V1_0_FILE_NAME = "v1.0.input.warc.paths.gz"
V1_0_N_WARCS = 8_582_115
# sha256 of the uncompressed v1.0 list (sorted, one path per line, "\n"-terminated).
V1_0_SHA256 = "101505dd737241a05afb03338b6e2db6175a40651d57538a7c4db0fd4576961a"
# sha256 of "<shard>\t<path>\n" over the production shard layout (layout.v1_0_layout), in shard order.
V1_0_LAYOUT_SHA256 = "7dfd37b8722b7f398c1e061d8368e33bbaee927112b825b402a9a73cb54726b9"
V1_0_FAMILY_COUNTS = {"CC-MAIN": 8_525_398, "CC-NEWS": 52_144, "CC-SUPPLEMENTAL": 4_573}

_DATA = resources.files("atlantic_cc.worklist") / "data"
_CRAWLS_TSV = "v1.0.crawls.tsv"
_SUPPLEMENTAL_PATHS = "v1.0.supplemental.warc.paths.gz"


@dataclass(frozen=True)
class PathsDigest:
    n_warcs: int
    sha256: str


def paths_text(paths: Iterable[str]) -> bytes:
    return "".join(f"{p}\n" for p in paths).encode()


def paths_digest(paths: Sequence[str]) -> PathsDigest:
    return PathsDigest(len(paths), hashlib.sha256(paths_text(paths)).hexdigest())


def parse_paths(text: bytes) -> list[str]:
    return [line.strip() for line in text.decode().split("\n") if line.strip()]


def read_paths_file(path: str) -> list[str]:
    """Lines of a paths file (gzipped when the name ends in ``.gz``), local or remote via fsspec."""
    with fsspec.open(path, "rb", compression="infer") as f:
        return parse_paths(f.read())


def write_paths_file(paths: Sequence[str], dest: str) -> None:
    """Write a gzipped paths file with a zero gzip timestamp."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9, mtime=0) as gz:
        gz.write(paths_text(paths))
    with fsspec.open(dest, "wb", auto_mkdir=True) as f:
        f.write(buf.getvalue())


def check_digest(paths: Sequence[str], expected: PathsDigest, what: str) -> None:
    got = paths_digest(paths)
    if got != expected:
        raise ValueError(f"{what}: got {got.n_warcs} paths / sha256 {got.sha256}, expected {expected}")


def load_frozen(path: str) -> list[str]:
    """Read the published v1.0 worklist and check it is exactly the frozen list."""
    paths = read_paths_file(path)
    check_digest(paths, PathsDigest(V1_0_N_WARCS, V1_0_SHA256), path)
    return paths


def crawl_digests(paths: Iterable[str]) -> dict[str, PathsDigest]:
    """Per crawl, the count and digest of its sorted paths."""
    by_crawl: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        by_crawl[snapshot_of(path)].append(path)
    return {crawl: paths_digest(sorted(ps)) for crawl, ps in sorted(by_crawl.items())}


def format_crawl_table(digests: dict[str, PathsDigest]) -> str:
    rows = ["crawl\tn_warcs\tsha256"] + [f"{c}\t{d.n_warcs}\t{d.sha256}" for c, d in sorted(digests.items())]
    return "".join(f"{row}\n" for row in rows)


def v1_0_crawl_digests() -> dict[str, PathsDigest]:
    """The packaged per-crawl counts and digests of the v1.0 list (249 crawls)."""
    text = (_DATA / _CRAWLS_TSV).read_text()
    return {
        row["crawl"]: PathsDigest(int(row["n_warcs"]), row["sha256"])
        for row in csv.DictReader(io.StringIO(text), delimiter="\t")
    }


def supplemental_paths() -> list[str]:
    """The packaged CC-SUPPLEMENTAL-2026-22 WARC paths (no public listing exists)."""
    return parse_paths(gzip.decompress((_DATA / _SUPPLEMENTAL_PATHS).read_bytes()))
