# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Restrict the release to a list of WARCs: per-WARC document, character and (optionally) token counts,
and optionally the matching documents themselves.

Port of production's ``experiments/fast_curation/subset_release_to_warcs.py``. Release rows are grouped
by source WARC (about ten WARCs per 10,000-row group), so each file is scanned cheaply: read the
``warc_path`` column (a few hundred bytes per row group), pick the row groups holding a target WARC,
and read ``text`` (and, when exporting, the other columns) for those row groups only. Only the crawls
of the listed WARCs are visited. Over HTTPS the reads are byte-range requests, so a subset of a few
thousand WARCs downloads a small fraction of the release.

Work units are chunks of ``FILES_PER_CHUNK`` release files; ``part=I/N`` processes every N-th chunk and
writes ``{output}/chunks/chunk-XXXXX.json`` (existing chunks are skipped, so an interrupted run resumes).
``merge`` sums the chunks into ``{output}/summary.json``. With ``export``, each chunk's matched clean
documents are written as ``{export}/data-XXXXX-of-NNNNN.parquet`` in the release schema, and that file
is the chunk's done marker.
"""

import collections
import json
import logging
import tempfile
from collections import deque
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from atlantic_cc.release import storage
from atlantic_cc.release.params import RELEASE_SCHEMA, WARC_PATH_FIELD, Kind, part_file_name
from atlantic_cc.spec import CC_S3_PREFIX

logger = logging.getLogger(__name__)

FILES_PER_CHUNK = 20
WRITE_THREADS = 8
TEXT_FIELD = "text"
WARC_PATH_PREFIXES = ("crawl-data/", "projects/")

# Set once per worker process by _init_worker.
_TARGETS: pa.Array | None = None
_TOKENIZER = None
_EXPORT = False


def normalize_warc(line: str) -> str:
    """A WARC list line as a bucket-relative path; accepts ``s3://commoncrawl/`` URLs too."""
    path = line.strip().removeprefix(CC_S3_PREFIX)
    if not path.startswith(WARC_PATH_PREFIXES):
        raise ValueError(f"not a Common Crawl WARC path: {line!r}")
    return path


def load_warc_list(path: str) -> list[str]:
    with storage.open_file(path, "r", compression="infer") as f:
        return [normalize_warc(line) for line in f if line.strip()]


def chunk_files(files: Sequence, files_per_chunk: int = FILES_PER_CHUNK) -> list[list]:
    return [list(files[i : i + files_per_chunk]) for i in range(0, len(files), files_per_chunk)]


def _init_worker(targets: list[str], tokenizer_dir: str | None, export: bool) -> None:
    global _TARGETS, _TOKENIZER, _EXPORT
    _TARGETS = pa.array(targets, type=pa.string())
    _EXPORT = export
    if tokenizer_dir is not None:
        # Imported only when counting tokens: transformers takes tens of seconds to import (it pulls in
        # torch), which every worker process would otherwise pay.
        from transformers import AutoTokenizer  # noqa: PLC0415

        _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_dir)


def stage_tokenizer(name: str) -> str:
    """Download the tokenizer once in the parent; workers load the local copy (no Hub calls per worker)."""
    from transformers import AutoTokenizer  # noqa: PLC0415  # slow import, see _init_worker

    local_dir = tempfile.mkdtemp(prefix="tokenizer_")
    AutoTokenizer.from_pretrained(name, cache_dir=tempfile.mkdtemp(prefix="hf_cache_")).save_pretrained(local_dir)
    return local_dir


def _count_tokens(texts: list[str]) -> list[int]:
    if _TOKENIZER is None:
        return [0] * len(texts)
    encodings = _TOKENIZER.backend_tokenizer.encode_batch(texts, add_special_tokens=False)
    return [len(e.ids) for e in encodings]


def scan_file(crawl: str, kind: str, path: str) -> dict:
    """Per-WARC ``[docs, chars, tokens]`` for the target WARCs in one release file."""
    result: dict = {"path": path, "crawl": crawl, "kind": kind, "row_groups": 0, "row_groups_read": 0, "warcs": {}}
    with storage.open_file(path, "rb", cache_type="none") as f:
        pf = pq.ParquetFile(f, pre_buffer=True)
        result["row_groups"] = pf.metadata.num_row_groups
        warcs = pf.read(columns=[WARC_PATH_FIELD]).column(0)
        hits = np.flatnonzero(pc.is_in(warcs, value_set=_TARGETS).to_numpy(zero_copy_only=False))
        if hits.size == 0:
            return result
        ends = np.cumsum([pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)])
        groups = sorted(set(np.searchsorted(ends, hits, side="right").tolist()))
        export = _EXPORT and kind == Kind.CLEAN.value
        columns = RELEASE_SCHEMA.names if export else [WARC_PATH_FIELD, TEXT_FIELD]
        table = pf.read_row_groups(groups, columns=columns)
    result["row_groups_read"] = len(groups)
    table = table.filter(pc.is_in(table.column(WARC_PATH_FIELD), value_set=_TARGETS))
    warc_col = table.column(WARC_PATH_FIELD).to_pylist()
    texts = table.column(TEXT_FIELD).to_pylist()
    if export:
        result["table"] = table
    per_warc: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0])
    for warc, text, n_tokens in zip(warc_col, texts, _count_tokens(texts), strict=True):
        entry = per_warc[warc]
        entry[0] += 1
        entry[1] += len(text)
        entry[2] += n_tokens
    result["warcs"] = dict(per_warc)
    return result


def _scan_file_star(args: tuple[str, str, str]) -> dict:
    return scan_file(*args)


def chunk_path(output: str, index: int) -> str:
    return f"{output}/chunks/chunk-{index:05d}.json"


def export_path(export: str, index: int, n_chunks: int) -> str:
    return f"{export}/data-{index:05d}-of-{n_chunks:05d}.parquet"


def _done_path(output: str, export: str | None, index: int, n_chunks: int) -> str:
    return export_path(export, index, n_chunks) if export else chunk_path(output, index)


def _write_export(path: str, files: list[dict]) -> int:
    """Write the chunk's matched clean documents as one Parquet file; returns the row count."""
    tables = [f.pop("table") for f in files if "table" in f]
    table = pa.concat_tables(tables) if tables else RELEASE_SCHEMA.empty_table()
    with storage.open_file(path, "wb") as fh:
        pq.write_table(table, fh, compression="zstd")
    return table.num_rows


def _finish_chunk(output: str, export: str | None, index: int, n_chunks: int, scanned: list[dict]) -> dict:
    """Write a chunk's export (if exporting) and then its stats JSON; returns the stats."""
    docs = sum(v[0] for f in scanned for v in f["warcs"].values())
    exported = _write_export(export_path(export, index, n_chunks), scanned) if export else None
    summary = {
        "chunk": index,
        "files": scanned,
        "docs": docs,
        "exported_docs": exported,
        "row_groups": sum(f["row_groups"] for f in scanned),
        "row_groups_read": sum(f["row_groups_read"] for f in scanned),
    }
    with storage.open_file(chunk_path(output, index), "w") as fh:
        json.dump(summary, fh)
    return summary


def run_part(
    output: str,
    export: str | None,
    chunks: list[list[tuple[str, str, str]]],
    part: int,
    parts: int,
    workers: int,
    targets: list[str],
    tokenizer_dir: str | None,
) -> None:
    """Scan this part's unfinished chunks, keeping every worker busy, and write each chunk as it completes."""
    todo = [
        (i, chunk)
        for i, chunk in enumerate(chunks)
        if i % parts == part and not storage.exists(_done_path(output, export, i, len(chunks)))
    ]
    logger.info("part %d/%d: %d chunks to do (%d files)", part, parts, len(todo), sum(len(c) for _, c in todo))
    files = [f for _, chunk in todo for f in chunk]
    with (
        ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker, initargs=(targets, tokenizer_dir, export is not None)
        ) as pool,
        ThreadPoolExecutor(max_workers=WRITE_THREADS) as writers,
    ):
        results = pool.map(_scan_file_star, files, chunksize=1)
        pending: deque = deque()
        for n, (index, chunk) in enumerate(todo):
            scanned = [next(results) for _ in chunk]
            pending.append((n, index, writers.submit(_finish_chunk, output, export, index, len(chunks), scanned)))
            while len(pending) >= WRITE_THREADS or (pending and pending[0][2].done()):
                m, idx, fut = pending.popleft()
                summary = fut.result()
                logger.info(
                    "chunk %d (%d/%d): %d docs, %d/%d row groups read",
                    idx,
                    m + 1,
                    len(todo),
                    summary["docs"],
                    summary["row_groups_read"],
                    summary["row_groups"],
                )
        for _, _, fut in pending:
            fut.result()


def merge(output: str, release: str, n_chunks: int, targets: list[str], tokenizer: str | None) -> dict:
    """Sum every chunk into summary.json; raises if a chunk is missing."""
    kinds = {k.value: [0, 0, 0] for k in Kind}
    crawls: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0])
    per_warc: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0])
    row_groups = row_groups_read = files = 0
    for index in range(n_chunks):
        chunk = storage.read_json(chunk_path(output, index))
        row_groups += chunk["row_groups"]
        row_groups_read += chunk["row_groups_read"]
        files += len(chunk["files"])
        for f in chunk["files"]:
            for warc, counts in f["warcs"].items():
                for acc in (kinds[f["kind"]], crawls[f["crawl"]], per_warc[warc]):
                    for j in range(3):
                        acc[j] += counts[j]

    def row(counts: list[int]) -> dict:
        docs, chars, tokens = counts
        return {
            "docs": docs,
            "chars": chars,
            "tokens": tokens if tokenizer else None,
            "chars_per_token": chars / tokens if tokenizer and tokens else None,
        }

    total = [sum(k[j] for k in kinds.values()) for j in range(3)]
    target_set = set(targets)
    unexpected = sorted(set(per_warc) - target_set)
    if unexpected:
        raise ValueError(f"{len(unexpected)} matched WARCs are not in the list, e.g. {unexpected[:3]}")
    summary = {
        "release": release,
        "tokenizer": tokenizer,
        "warcs_in_list": len(targets),
        "warcs_with_docs": len(per_warc),
        "warcs_without_docs": sorted(target_set - set(per_warc)),
        "files_scanned": files,
        "row_groups": row_groups,
        "row_groups_read": row_groups_read,
        "total": row(total),
        "by_kind": {k: row(v) for k, v in kinds.items()},
        "by_crawl": {k: row(v) for k, v in sorted(crawls.items())},
        "per_warc": {k: per_warc[k] for k in sorted(per_warc)},
    }
    with storage.open_file(f"{output}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    return summary


def subset_files(release: str, manifest: dict, crawls: set[str]) -> list[tuple[str, str, str]]:
    """``(crawl, kind, url)`` for every release file of the given crawls: crawls sorted, parts before the
    contaminated file (production's chunking order)."""
    version = manifest["version"]
    files = []
    for crawl in sorted(crawls):
        entry = manifest["crawls"][crawl]
        for kind in Kind:
            n = entry[kind.value]["files"]
            for i in range(n):
                files.append((crawl, kind.value, f"{release}/{version}/crawl={crawl}/{part_file_name(kind, i, n)}"))
    return files
