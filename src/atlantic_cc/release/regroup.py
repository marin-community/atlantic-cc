# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regroup the deduplicated pool into the release layout: one hive partition per crawl.

Port of production's ``experiments/fast_curation/release_regroup.py`` (Marin tag ``lpv11-8m-as-run``)
with Zephyr replaced by a local process pool; record handling, file naming, compression settings and
ordering are unchanged. Four resumable steps:

1. ``warc_paths``: ``<crawl>/<warc_hash> -> warc_path`` for every WARC of the worklist (dedup records
   carry only the hash).
2. ``pieces``: each batch of ``FILES_PER_TASK`` dedup output files becomes one Parquet piece per
   ``(crawl, kind)`` under ``{stage}/pieces/crawl=<crawl>/kind=<kind>/batch-NNNNNN.parquet``; records the
   decontamination filter removed are tagged with the evals they matched and go to ``kind=contaminated``.
   A batch's ``_done`` marker is written last.
3. ``parts``: each crawl's pieces of one kind are packed, in batch order, into parts of at least
   ``part_mb`` piece-megabytes (so every part count is known before writing), then written with zstd
   level 9 and 10,000-row row groups. Existing parts are skipped.
4. ``manifest``: per-crawl record and file counts from the part footers, checked against the
   decontamination totals, plus the two paths files.

Record order: within a ``(crawl, kind)``, records follow batch order, i.e. the dedup output order,
which groups records by source WARC in ascending ``warc_hash`` order and keeps each WARC's record
order.
"""

import gzip
import json
import logging
import multiprocessing as mp
import os
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq

from atlantic_cc.release import storage
from atlantic_cc.release.params import (
    COLUMNS,
    DECON_EXPORT_FILE,
    DECON_FIELD,
    DECON_LOOKUP_FILE,
    DONE_DIR,
    FILES_PER_TASK,
    LOOKUP_ROW_GROUP_ROWS,
    PARTS_PLAN_FILE,
    PIECES_DIR,
    PUBLISHED_PREFIX,
    READ_THREADS,
    RELEASE_SCHEMA,
    REMOVED_DOCS_FILE,
    WARC_PATH_FIELD,
    WARC_PATHS_FILE,
    WRITE_BATCH_ROWS,
    Kind,
    clean_paths_name,
    manifest_name,
    part_file_name,
    paths_name,
)
from atlantic_cc.warc.identity import snapshot_of, warc_hash
from atlantic_cc.worklist.frozen import V1_0_N_WARCS, V1_0_SHA256, PathsDigest, paths_digest, read_paths_file
from atlantic_cc.worklist.identity import to_s3_url
from atlantic_cc.worklist.layout import v1_0_layout

logger = logging.getLogger(__name__)


def release_schema(drop: Sequence[str]) -> pa.Schema:
    return pa.schema([f for f in RELEASE_SCHEMA if f.name not in drop])


def release_row(record: dict, warc_path: str, matches: list[str], schema: pa.Schema) -> dict:
    row = {name: record.get(COLUMNS[name]) for name in schema.names if name in COLUMNS}
    row[WARC_PATH_FIELD] = warc_path
    row[DECON_FIELD] = matches
    return row


def read_jsonl(path: str) -> Iterator[dict]:
    with storage.open_file(path, "rb") as raw, gzip.open(raw, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


# ---- warc paths -------------------------------------------------------------------------------------


def warc_key(crawl: str, warc_hash: str) -> str:
    return f"{crawl}/{warc_hash}"


def warc_paths_table(shards: Sequence[Sequence[str]]) -> pa.Table:
    """``key = <crawl>/<warc_hash>`` and the bucket-relative ``warc_path``, in shard order."""
    keys, relative = [], []
    for shard in shards:
        for path in shard:
            keys.append(warc_key(snapshot_of(path), warc_hash(to_s3_url(path))))
            relative.append(path)
    if len(set(keys)) != len(keys):
        raise ValueError(f"{len(keys) - len(set(keys))} (crawl, warc_hash) collisions in the worklist")
    return pa.table({"key": keys, WARC_PATH_FIELD: relative})


def worklist_shards(worklist: str) -> list[list[str]]:
    """The worklist in production's shard order when it is the v1.0 list, else as one sorted shard.

    Only the row order of ``warc_paths.parquet`` depends on this; the release does not.
    """
    paths = read_paths_file(worklist)
    if paths_digest(paths) == PathsDigest(V1_0_N_WARCS, V1_0_SHA256):
        return v1_0_layout(paths)
    return [sorted(paths)]


def write_warc_paths(worklist: str, stage: str) -> None:
    table = warc_paths_table(worklist_shards(worklist))
    with storage.open_file(f"{stage}/{WARC_PATHS_FILE}", "wb") as fh:
        pq.write_table(table, fh, compression="zstd")
    logger.info("warc paths: %d WARCs", table.num_rows)


def load_warc_paths(stage: str) -> dict[str, str]:
    table = storage.read_parquet(f"{stage}/{WARC_PATHS_FILE}")
    return dict(zip(table.column("key").to_pylist(), table.column(WARC_PATH_FIELD).to_pylist(), strict=True))


# ---- decontamination lookup -----------------------------------------------------------------------


def write_decon_lookup(decon_run: str, stage: str) -> None:
    """Copy the removed docs' ``source_file, doc_id, matched_evals`` into small row groups, sorted by file.

    The filter's ``removed_docs.parquet`` has two huge row groups; small ones let each batch read only
    the rows of its own source files.
    """
    table = storage.read_parquet(f"{decon_run}/{REMOVED_DOCS_FILE}", columns=["source_file", "doc_id", "matched_evals"])
    table = table.sort_by("source_file")
    with storage.open_file(f"{stage}/{DECON_LOOKUP_FILE}", "wb") as fh:
        pq.write_table(table, fh, compression="zstd", row_group_size=LOOKUP_ROW_GROUP_ROWS)
    logger.info("decon lookup: %d removed docs", table.num_rows)


def decon_matches(stage: str, source_files: Sequence[str]) -> dict[str, list[str]]:
    """``doc_id -> matched evals`` for every doc the decon filter removed from *source_files*."""
    table = storage.read_parquet(
        f"{stage}/{DECON_LOOKUP_FILE}",
        columns=["doc_id", "matched_evals"],
        filters=[("source_file", "in", [os.path.basename(p) for p in source_files])],
    )
    return dict(zip(table.column("doc_id").to_pylist(), table.column("matched_evals").to_pylist(), strict=True))


# ---- pieces -----------------------------------------------------------------------------------------


def piece_path(stage: str, crawl: str, kind: str, batch_id: int) -> str:
    return f"{stage}/{PIECES_DIR}/crawl={crawl}/kind={kind}/batch-{batch_id:06d}.parquet"


def done_marker_path(stage: str, batch_id: int) -> str:
    return f"{stage}/{PIECES_DIR}/{DONE_DIR}/batch-{batch_id:06d}.json"


class PieceWriter:
    """Streams one batch's records into one Parquet piece per (crawl, kind)."""

    def __init__(self, stage: str, schema: pa.Schema, batch_id: int):
        self.stage, self.schema, self.batch_id = stage, schema, batch_id
        self.buffers: dict[tuple[str, Kind], list[dict]] = defaultdict(list)
        self.writers: dict[tuple[str, Kind], tuple] = {}

    def add(self, row: dict, kind: Kind) -> None:
        key = (row["crawl"], kind)
        buffer = self.buffers[key]
        buffer.append(row)
        if len(buffer) >= WRITE_BATCH_ROWS:
            self._flush(key)

    def _flush(self, key: tuple[str, Kind]) -> None:
        if key not in self.writers:
            crawl, kind = key
            fh = storage.open_file(piece_path(self.stage, crawl, kind, self.batch_id), "wb").open()
            self.writers[key] = (fh, pq.ParquetWriter(fh, self.schema, compression="zstd"))
        self.writers[key][1].write_table(pa.Table.from_pylist(self.buffers.pop(key), schema=self.schema))

    def close(self) -> list[str]:
        for key in list(self.buffers):
            self._flush(key)
        for fh, writer in self.writers.values():
            writer.close()
            fh.close()
        return sorted(f"{crawl}/{kind}" for crawl, kind in self.writers)


def build_batch(
    stage: str, schema: pa.Schema, batch_id: int, paths: Sequence[str], warc_path_of: dict[str, str]
) -> dict:
    """Write the pieces of one batch, then its done marker."""
    removed = decon_matches(stage, paths)
    writer = PieceWriter(stage, schema, batch_id)
    n = n_contaminated = 0
    for path in paths:
        for record in read_jsonl(path):
            matches = removed.get(record[COLUMNS["record_id"]], [])
            warc_path = warc_path_of[warc_key(record["snapshot"], record["warc_hash"])]
            row = release_row(record, warc_path, matches, schema)
            writer.add(row, Kind.CONTAMINATED if matches else Kind.CLEAN)
            n += 1
            n_contaminated += bool(matches)
    if n_contaminated != len(removed):
        raise RuntimeError(f"batch {batch_id}: {len(removed)} removed docs but {n_contaminated} tagged")
    outputs = writer.close()
    # Written last: a marker means every piece of this batch is complete.
    storage.write_bytes(
        done_marker_path(stage, batch_id),
        json.dumps({"files": list(paths), "records": n, "contaminated": n_contaminated, "pieces": outputs}).encode(),
    )
    return {"batch": batch_id, "records": n, "contaminated": n_contaminated}


def plan_batches(source_files: Sequence[str]) -> list[tuple[int, list[str]]]:
    """``(batch_id, files)`` over the sorted dedup output files, ``FILES_PER_TASK`` files per batch."""
    files = sorted(source_files)
    return [(i // FILES_PER_TASK, files[i : i + FILES_PER_TASK]) for i in range(0, len(files), FILES_PER_TASK)]


_WARC_PATH_OF: dict[str, str] = {}


def _init_pieces_worker(stage: str) -> None:
    global _WARC_PATH_OF
    _WARC_PATH_OF = load_warc_paths(stage)


def _pieces_task(stage: str, drop: Sequence[str], batch: tuple[int, list[str]]) -> dict:
    batch_id, paths = batch
    return build_batch(stage, release_schema(drop), batch_id, paths, _WARC_PATH_OF)


def pieces(
    source: str,
    decon_run: str,
    stage: str,
    drop: Sequence[str],
    workers: int,
    *,
    limit_files: int | None = None,
    batch_ids: Sequence[int] | None = None,
    shard: int = 0,
    num_shards: int = 1,
) -> None:
    """Build every unfinished batch (this shard's share of them, or only *batch_ids*)."""
    files = sorted(storage.glob(f"{source}/*.jsonl.gz"))[:limit_files]
    batches = plan_batches(files)
    done = {os.path.basename(p) for p in storage.glob(f"{stage}/{PIECES_DIR}/{DONE_DIR}/*.json")}
    wanted = set(batch_ids) if batch_ids is not None else None
    todo = [
        b
        for b in batches
        if os.path.basename(done_marker_path(stage, b[0])) not in done
        and b[0] % num_shards == shard
        and (wanted is None or b[0] in wanted)
    ]
    logger.info("pieces: %d of %d batches to build (%d source files)", len(todo), len(batches), len(files))
    if not todo:
        return
    if not storage.exists(f"{stage}/{DECON_LOOKUP_FILE}"):
        write_decon_lookup(decon_run, stage)
    if not storage.exists(f"{stage}/{WARC_PATHS_FILE}"):
        raise FileNotFoundError(f"{stage}/{WARC_PATHS_FILE} is missing; run the warc-paths step first")
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=mp.get_context("spawn"), initializer=_init_pieces_worker, initargs=(stage,)
    ) as ex:
        futures = [ex.submit(_pieces_task, stage, list(drop), b) for b in todo]
        for n, future in enumerate(futures, 1):
            result = future.result()
            logger.info("pieces: batch %d done (%d/%d): %s", result["batch"], n, len(todo), result)


# ---- parts ------------------------------------------------------------------------------------------


def glob_pieces(stage: str) -> list[tuple[str, int]]:
    # One listing with sizes; a size() call per piece is ~750k sequential requests.
    return storage.glob_sizes(f"{stage}/{PIECES_DIR}/crawl=*/kind=*/*.parquet")


def plan_parts(pieces: Iterable[tuple[str, int]], stage: str, part_bytes: int) -> list[dict]:
    """Group each crawl's pieces of one kind, in batch order, into parts of about ``part_bytes``.

    A part closes as soon as its pieces reach ``part_bytes``, so every part but a crawl's last is at
    least that large (in piece bytes).
    """
    groups_by_key: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    prefix = f"{stage}/{PIECES_DIR}/"
    for path, n_bytes in pieces:
        crawl_dir, kind_dir = path.removeprefix(prefix).split("/")[:2]
        key = (crawl_dir.removeprefix("crawl="), kind_dir.removeprefix("kind="))
        groups_by_key[key].append((path, n_bytes))
    plan = []
    for (crawl, kind), items in sorted(groups_by_key.items()):
        groups, current, size = [], [], 0
        for path, n_bytes in sorted(items):
            current.append(path)
            size += n_bytes
            if size >= part_bytes:
                groups.append(current)
                current, size = [], 0
        if current:
            groups.append(current)
        plan += [
            {"crawl": crawl, "kind": kind, "part": i, "of": len(groups), "pieces": g} for i, g in enumerate(groups)
        ]
    return plan


def part_key(out: str, version: str, part: dict) -> str:
    return f"{out}/{version}/crawl={part['crawl']}/{part_file_name(part['kind'], part['part'], part['of'])}"


def write_part(part: dict, out: str, version: str, zstd_level: int) -> dict:
    """Concatenate a part's pieces and write it; a part is written whole, so an existing file is final."""
    table = pa.concat_tables([storage.read_parquet(p) for p in part["pieces"]])
    with storage.open_file(part_key(out, version, part), "wb") as fh:
        pq.write_table(table, fh, compression="zstd", compression_level=zstd_level, row_group_size=WRITE_BATCH_ROWS)
    return {"crawl": part["crawl"], "kind": part["kind"], "part": part["part"], "records": table.num_rows}


def parts(
    out: str,
    version: str,
    stage: str,
    part_mb: int,
    zstd_level: int,
    workers: int,
    *,
    plan_path: str | None = None,
    shard: int = 0,
    num_shards: int = 1,
) -> None:
    plan = plan_parts(glob_pieces(stage), stage, part_mb * 1_000_000)
    with storage.open_file(plan_path or f"{stage}/{PARTS_PLAN_FILE}", "w") as fh:
        json.dump(plan, fh)
    existing = set(storage.glob(f"{out}/{version}/crawl=*/*.parquet"))
    todo = [p for i, p in enumerate(plan) if i % num_shards == shard and part_key(out, version, p) not in existing]
    logger.info("parts: %d of %d parts to write over %d crawls", len(todo), len(plan), len({p["crawl"] for p in plan}))
    if not todo:
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        futures = [ex.submit(write_part, p, out, version, zstd_level) for p in todo]
        for n, future in enumerate(futures, 1):
            logger.info("parts: %d/%d %s", n, len(todo), future.result())


# ---- manifest ---------------------------------------------------------------------------------------


def manifest_document(
    plan: Sequence[dict], rows: Sequence[int], version: str, decon_export: dict | None, decon_run: str | None
) -> dict:
    """Per-crawl record and file counts; raises if the totals disagree with the decontamination run."""
    crawls: dict[str, dict] = defaultdict(lambda: {str(k): {"records": 0, "files": 0} for k in Kind})
    for part, n in zip(plan, rows, strict=True):
        crawls[part["crawl"]][part["kind"]]["records"] += n
        crawls[part["crawl"]][part["kind"]]["files"] += 1
    totals = {str(k): sum(c[k]["records"] for c in crawls.values()) for k in Kind}
    result = {"version": version, "records": sum(rows), "by_kind": totals, "crawls": dict(sorted(crawls.items()))}
    if decon_export is not None:
        summary = decon_export["summary"]
        expected = {Kind.CLEAN: summary["decon/docs_kept"], Kind.CONTAMINATED: summary["decon/docs_removed"]}
        for kind, n in expected.items():
            if totals[kind] != n:
                raise RuntimeError(f"release has {totals[kind]} {kind} records but the decon run has {n}")
        result["decontamination"] = {"method": decon_export["method"], "run": decon_run}
    return result


def manifest_bytes(document: dict) -> bytes:
    return json.dumps(document, indent=1).encode()


def published_keys(plan: Sequence[dict], version: str) -> list[str]:
    return [part_key(PUBLISHED_PREFIX, version, p) for p in plan]


def paths_files(plan: Sequence[dict], version: str, gzip_mtime: int | None) -> dict[str, bytes]:
    """The two gzipped paths files; ``gzip_mtime=None`` stamps the current time, as production did."""
    published = published_keys(plan, version)
    clean = [key for key, p in zip(published, plan, strict=True) if p["kind"] == Kind.CLEAN]
    return {
        name: gzip.compress("".join(f"{k}\n" for k in keys).encode(), mtime=gzip_mtime)
        for name, keys in ((paths_name(version), published), (clean_paths_name(version), clean))
    }


def part_rows(out: str, version: str, plan: Sequence[dict]) -> list[int]:
    with ThreadPoolExecutor(READ_THREADS) as ex:
        return list(ex.map(storage.parquet_rows, [part_key(out, version, p) for p in plan]))


def manifest(
    out: str,
    version: str,
    stage: str,
    decon_run: str | None,
    *,
    plan_path: str | None = None,
    gzip_mtime: int | None = None,
) -> dict:
    """Write ``{out}/{version}.manifest.json`` and the paths files next to ``{out}/{version}/``."""
    plan = storage.read_json(plan_path or f"{stage}/{PARTS_PLAN_FILE}")
    rows = part_rows(out, version, plan)
    export = storage.read_json(f"{decon_run}/{DECON_EXPORT_FILE}") if decon_run else None
    document = manifest_document(plan, rows, version, export, decon_run)
    # Hive readers need {version}/ to hold only crawl= partitions, so these sit one level above it.
    with storage.open_file(f"{out}/{manifest_name(version)}", "w") as fh:
        json.dump(document, fh, indent=1)
    for name, data in paths_files(plan, version, gzip_mtime).items():
        storage.write_bytes(f"{out}/{name}", data)
    logger.info(
        "manifest: %d records (%s) over %d crawls, %d files",
        sum(rows),
        document["by_kind"],
        len(document["crawls"]),
        len(plan),
    )
    return document
