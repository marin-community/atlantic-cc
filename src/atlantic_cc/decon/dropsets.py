# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Boilerplate drop sets: phrases that are eval phrases but also common web boilerplate.

Port of ``dropsets``/``_count_source`` and the drop-list readers of Marin
``experiments/fast_curation/decon_8m.py`` (tag ``lpv11-8m-as-run``). The final drop set D is the union
of four lists (``_drop_hashes``):

* per-crawl (local): per source, bloom-hit phrases in >= ``max(5, int(0.005 * n))`` of the source's
  first ``n <= 5,000`` sampled docs;
* global: summed document frequency over every source's first <= 1,000,000 sampled docs >= 50, in
  >= 3 sources;
* pooldf1000: phrases in >= 1,000 pool docs across >= 3 crawls (``pool_totals``);
* codedf100: phrases in >= 100 StarCoderData files (``code_totals``).

A "source" is a crawl (``snapshot``) with every CC-NEWS month merged into ``CC-NEWS``.
"""

import concurrent.futures
import logging
import multiprocessing
import os
from collections import Counter, deque
from collections.abc import Callable, Container, Iterable, Iterator
from dataclasses import dataclass
from itertools import islice

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from atlantic_cc.decon.matching import _bloom_hash, _extract_features
from atlantic_cc.decon.params import (
    CODE_DROP_MIN_DF,
    COUNT_PARTITIONS,
    GLOBAL_DF_COMMON_MIN_ABS,
    GLOBAL_DF_COMMON_MIN_SOURCES,
    GLOBAL_DF_SAMPLE_DOCS,
    MARK_NGRAM,
    POOL_DROP_MIN_CRAWLS,
    POOL_DROP_MIN_DF,
    SOURCE_DF_COMMON_FRAC,
    SOURCE_DF_COMMON_MIN_ABS,
    SOURCE_DF_SAMPLE_DOCS,
)

logger = logging.getLogger(__name__)

DROP_FILE = "drop.parquet"
COUNTS_DIR = "_counts"
GLOBAL_SOURCE = "_global"
PART_PREFETCH = 16


def drop_set_source(snapshot: str) -> str:
    """Drop-set source of a doc: its crawl (``CC-MAIN-2014-35``), with all CC-NEWS months as one source."""
    return "CC-NEWS" if snapshot.startswith("CC-NEWS") else snapshot


@dataclass
class SourceCounts:
    """Document frequencies of bloom-hit eval phrases in one source's sample prefix."""

    local: Counter[int]
    """Over the first ``SOURCE_DF_SAMPLE_DOCS`` non-empty docs."""
    glob: Counter[int]
    """Over the first ``GLOBAL_DF_SAMPLE_DOCS`` non-empty docs."""
    docs: int
    local_docs: int


def count_source(parts: Iterable[list[str | None]], bf: Container[int]) -> SourceCounts:
    """Count, per phrase, the sampled docs containing it, for one source.

    *parts* yields each sample part's texts, parts in name order. Mirrors the as-run
    ``_count_source`` loop exactly: empty texts are skipped and not counted, each doc counts a phrase
    once, only bloom hits are counted, and Counter insertion order (which fixes the row order of the
    written tables) follows the same set iteration.
    """
    local: Counter[int] = Counter()
    glob: Counter[int] = Counter()
    n = n_local = 0
    for texts in parts:
        if n >= GLOBAL_DF_SAMPLE_DOCS:
            break
        for text in texts:
            if n >= GLOBAL_DF_SAMPLE_DOCS:
                break
            if not text:
                continue
            features = {_bloom_hash(f) for f in _extract_features(text, MARK_NGRAM)}
            hits = {h for h in features if h in bf}
            glob.update(hits)
            if n < SOURCE_DF_SAMPLE_DOCS:
                local.update(hits)
            n += 1
            n_local += n <= SOURCE_DF_SAMPLE_DOCS
    return SourceCounts(local=local, glob=glob, docs=n, local_docs=n_local)


def local_threshold(local_docs: int) -> int:
    return max(SOURCE_DF_COMMON_MIN_ABS, int(SOURCE_DF_COMMON_FRAC * local_docs))


def local_drop_hashes(counts: SourceCounts) -> list[int]:
    """The source-local drop list, in the as-run (Counter) order."""
    threshold = local_threshold(counts.local_docs)
    return [h for h, c in counts.local.items() if c >= threshold]


def counts_table(counts: SourceCounts) -> pa.Table:
    """``_counts/<source>.parquet``: ``hash uint64, df int64`` over the global prefix."""
    return pa.table(
        {
            "hash": pa.array(list(counts.glob.keys()), pa.uint64()),
            "df": pa.array(list(counts.glob.values()), pa.int64()),
        }
    )


def drop_table(hashes: Iterable[int]) -> pa.Table:
    return pa.table({"hash": pa.array(list(hashes), pa.uint64())})


def global_drop_hashes(count_tables: list[pa.Table]) -> pa.Array:
    """Phrases with summed df >= 50 over all sources' tables and present in >= 3 of them."""
    grouped = pa.concat_tables(count_tables).group_by("hash").aggregate([("df", "sum"), ("df", "count")])
    common = grouped.filter(
        pc.and_(
            pc.greater_equal(grouped["df_sum"], GLOBAL_DF_COMMON_MIN_ABS),
            pc.greater_equal(grouped["df_count"], GLOBAL_DF_COMMON_MIN_SOURCES),
        )
    )
    return common["hash"]


def pool_common_hashes(pool_totals: Iterable[pa.Table], min_df: int = POOL_DROP_MIN_DF) -> set[int]:
    """Phrases in >= *min_df* pool docs across >= ``POOL_DROP_MIN_CRAWLS`` crawls (``pool_totals`` parts)."""
    hashes: set[int] = set()
    for table in pool_totals:
        keep = pc.and_(pc.greater_equal(table["df"], min_df), pc.greater_equal(table["crawls"], POOL_DROP_MIN_CRAWLS))
        hashes.update(table.filter(keep).column("hash").to_pylist())
    return hashes


def code_common_hashes(code_totals: Iterable[pa.Table], min_df: int = CODE_DROP_MIN_DF) -> set[int]:
    """Phrases in >= *min_df* StarCoderData files (``code_totals`` parts)."""
    hashes: set[int] = set()
    for table in code_totals:
        hashes.update(table.filter(pc.greater_equal(table["df"], min_df)).column("hash").to_pylist())
    return hashes


def read_parquet(path: str, columns: list[str] | None = None) -> pa.Table:
    with fsspec.open(path, "rb") as fh:
        return pq.read_table(fh, columns=columns)


def write_parquet(path: str, table: pa.Table) -> None:
    with fsspec.open(path, "wb") as fh:
        pq.write_table(table, fh, compression="zstd")


def read_texts(path: str) -> list[str | None]:
    return read_parquet(path, columns=["text"]).column("text").to_pylist()


def prefetched(paths: Iterable[str], read: Callable[[str], list], depth: int = PART_PREFETCH) -> Iterator[list]:
    """Yield ``read(path)`` in order, reading up to *depth* paths ahead on threads."""
    it = iter(paths)
    with concurrent.futures.ThreadPoolExecutor(depth) as ex:
        pending = deque(ex.submit(read, p) for p in islice(it, depth))
        try:
            while pending:
                result = pending.popleft().result()
                nxt = next(it, None)
                if nxt is not None:
                    pending.append(ex.submit(read, nxt))
                yield result
        finally:
            for future in pending:
                future.cancel()


def sample_parts(sample_dir: str, source: str) -> list[str]:
    """A source's sample parts in name order (``<sample_dir>/<source>/part-NNNNNN.parquet``)."""
    fs, root = fsspec.core.url_to_fs(sample_dir.rstrip("/"))
    protocol = sample_dir.split("://", 1)[0] + "://" if "://" in sample_dir else ""
    return sorted(protocol + p for p in fs.glob(f"{root}/{source}/*.parquet"))


def drop_set_union(local_and_global: Iterable[Iterable[int]], pool: set[int], code: set[int]) -> np.ndarray:
    """D as a sorted uint64 array."""
    union: set[int] = set().union(*(set(h) for h in local_and_global)) | pool | code
    return np.array(sorted(union), dtype=np.uint64)


def drop_tables_dirs(root: str) -> list[str]:
    """Every ``<root>/*/drop.parquet`` (127 source-local lists and ``_global``), sorted."""
    fs, base = fsspec.core.url_to_fs(root.rstrip("/"))
    protocol = root.split("://", 1)[0] + "://" if "://" in root else ""
    return sorted(protocol + p for p in fs.glob(f"{base}/*/{DROP_FILE}"))


def partition_paths(root: str, name: str) -> list[str]:
    return [f"{root.rstrip('/')}/{name}/part={part:03d}.parquet" for part in range(COUNT_PARTITIONS)]


def sample_sources(sample_dir: str) -> list[str]:
    """Source names of a drop-set sample (its subdirectories), sorted."""
    fs, root = fsspec.core.url_to_fs(sample_dir.rstrip("/"))
    return sorted(os.path.basename(p.rstrip("/")) for p in fs.ls(root, detail=False) if fs.isdir(p))


_WORKER: dict = {}


def write_source_drop_set(sample_dir: str, source: str, out_root: str) -> SourceCounts:
    """Count one source of a sample and write ``_counts/<source>.parquet`` and ``<source>/drop.parquet``."""
    counts = count_source(prefetched(sample_parts(sample_dir, source), read_texts), _WORKER["bloom"])
    write_parquet(f"{out_root}/{COUNTS_DIR}/{source}.parquet", counts_table(counts))
    write_parquet(f"{out_root}/{source}/{DROP_FILE}", drop_table(local_drop_hashes(counts)))
    return counts


def _source_task(args: tuple[str, str, str]) -> tuple[str, int, int]:
    sample_dir, source, out_root = args
    counts = write_source_drop_set(sample_dir, source, out_root)
    logger.info("%s: %d docs (%d local), %d phrases", source, counts.docs, counts.local_docs, len(counts.glob))
    return source, counts.docs, counts.local_docs


def run_dropsets(sample_dir: str, bloom: Container[int], out_root: str, workers: int) -> int:
    """As-run ``dropsets`` step: every source's local list and counts, then the global list.

    Returns the size of the global list. *bloom* is shared with the fork-based workers.
    """
    out_root = out_root.rstrip("/")
    sources = sample_sources(sample_dir)
    _WORKER["bloom"] = bloom
    tasks = [(sample_dir, s, out_root) for s in sources]
    with multiprocessing.get_context("fork").Pool(max(1, min(workers, len(tasks)))) as pool:
        for source, docs, local_docs in pool.imap_unordered(_source_task, tasks):
            logger.info("done %s (%d docs, %d local)", source, docs, local_docs)
    with concurrent.futures.ThreadPoolExecutor(32) as ex:
        tables = list(ex.map(read_parquet, [f"{out_root}/{COUNTS_DIR}/{s}.parquet" for s in sources]))
    common = global_drop_hashes(tables)
    write_parquet(f"{out_root}/{GLOBAL_SOURCE}/{DROP_FILE}", drop_table(common.to_pylist()))
    return len(common)
