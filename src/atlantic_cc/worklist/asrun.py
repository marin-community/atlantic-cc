# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Maintainer tools: export the frozen worklist and its verification aids from production's records.

Production kept the worklist as 34,628 shard parquets (``warc_path`` = full ``s3://commoncrawl/`` URL,
``warc_hash``) plus ``_index.parquet`` (``shard``, ``n_warcs``, ``region``), a per-shard ``catalog/`` of
per-WARC counters, and an audit of the WARCs that produced no output. These live in Marin's GCS
buckets; reproducers use the exported files instead.
"""

import hashlib
import io
import json
import logging
from collections.abc import Callable

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq

from atlantic_cc.warc.identity import warc_hash
from atlantic_cc.worklist.frozen import paths_text, write_paths_file
from atlantic_cc.worklist.identity import to_relative, to_s3_url
from atlantic_cc.worklist.layout import layout_digest, v1_0_layout

logger = logging.getLogger(__name__)

ASRUN_ROOT = "gs://marin-us-central1/documents/fast_curation/lpv11_fastpipe_v2_1_fused-2a076ddfcb"
# Shards 0-403 ran through the two-phase spec first; their catalog lives in that spec's namespace.
ASRUN_TWO_PHASE_ROOT = "gs://marin-us-central1/documents/fast_curation/lpv11_fastpipe_v2_1-944ca6bc38"
TWO_PHASE_SHARDS = 404
EMPTY_WARCS_AUDIT = "_reopened/audit-empty-warcs.json"
CATALOG_COUNTERS = (
    "a_n_in",
    "a_n_prefilter_dropped",
    "a_n_extract_empty",
    "a_n_extract_crashed",
    "n_presurvivors",
    "n_hi_accept",
    "n_band",
    "n_kept",
)
CATALOG_SCHEMA = pa.schema(
    [
        ("shard", pa.int32()),
        ("warc_path", pa.string()),
        ("warc_hash", pa.string()),
        *[(name, pa.int64()) for name in CATALOG_COUNTERS],
        ("device", pa.string()),
        ("n_devices", pa.int64()),
    ]
)
EMPTY_WARCS_HEADER = ("warc_path", "warc_hash", "shard", "http_status", "bytes")


def _cat_parquets(root: str, names: list[str], concurrency: int) -> list[pa.Table]:
    fs, base = fsspec.core.url_to_fs(root)
    paths = [f"{base}/{name}" for name in names]
    blobs = fs.cat(paths, batch_size=concurrency)
    return [pq.read_table(io.BytesIO(blobs[p])) for p in paths]


def read_asrun_shards(shards_root: str, concurrency: int) -> list[list[str]]:
    """Production's shard lists as bucket-relative paths, checking hashes and the index counts."""
    (index,) = _cat_parquets(shards_root, ["_index.parquet"], 1)
    n_warcs = index.column("n_warcs").to_pylist()
    if index.column("shard").to_pylist() != list(range(len(n_warcs))):
        raise ValueError("the shard index is not dense")
    tables = _cat_parquets(shards_root, [f"shard-{s:05d}.parquet" for s in range(len(n_warcs))], concurrency)
    shards = []
    for s, (table, expected) in enumerate(zip(tables, n_warcs, strict=True)):
        urls = table.column("warc_path").to_pylist()
        hashes = table.column("warc_hash").to_pylist()
        if len(urls) != expected:
            raise ValueError(f"shard {s}: {len(urls)} WARCs, index says {expected}")
        for url, h in zip(urls, hashes, strict=True):
            if warc_hash(url) != h:
                raise ValueError(f"shard {s}: warc_hash {h} does not match {url}")
        shards.append([to_relative(url) for url in urls])
    return shards


def export_frozen(shards_root: str, out: str, concurrency: int) -> dict:
    """Write the sorted v1.0 list from production's shards; check the layout is reproduced exactly."""
    shards = read_asrun_shards(shards_root, concurrency)
    paths = sorted(p for shard in shards for p in shard)
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate WARC paths in the worklist")
    rebuilt = v1_0_layout(paths)
    mismatched = [s for s, (a, b) in enumerate(zip(shards, rebuilt, strict=True)) if a != b]
    if mismatched:
        raise ValueError(f"{len(mismatched)} shards differ from the rebuilt layout, e.g. {mismatched[:5]}")
    write_paths_file(paths, out)
    summary = {
        "n_warcs": len(paths),
        "n_shards": len(shards),
        "sha256": hashlib.sha256(paths_text(paths)).hexdigest(),
        "layout_sha256": layout_digest(shards),
    }
    logger.info("frozen worklist: %s", summary)
    return summary


def export_empty_warcs(audit_path: str, out: str) -> int:
    """The WARCs that produced no output, as a TSV sorted by path (``bytes`` is -1 when not HTTP 200)."""
    with fsspec.open(audit_path, "r") as f:
        rows = json.load(f)
    lines = ["\t".join(EMPTY_WARCS_HEADER)]
    for row in sorted(rows, key=lambda r: r["path"]):
        if warc_hash(row["path"]) != row["warc_hash"]:
            raise ValueError(f"bad audit row {row}")
        lines.append(f"{to_relative(row['path'])}\t{row['warc_hash']}\t{row['shard']}\t{row['status']}\t{row['bytes']}")
    with fsspec.open(out, "wt", compression="infer", auto_mkdir=True) as f:
        f.write("".join(f"{line}\n" for line in lines))
    return len(rows)


CATALOG_READ_BATCH = 2_000  # catalog files fetched per round


def _catalog_rows(table: pa.Table, shard: int, path_of: dict[str, tuple[int, str]]) -> pa.Table:
    hashes = table.column("warc_hash").to_pylist()
    located = [path_of[h] for h in hashes]
    if any(s != shard for s, _ in located):
        raise ValueError(f"catalog shard {shard} lists WARCs of other shards")
    columns = {
        "shard": pa.array([shard] * len(hashes), pa.int32()),
        "warc_path": pa.array([p for _, p in located], pa.string()),
        "warc_hash": pa.array(hashes, pa.string()),
    }
    for field in CATALOG_SCHEMA:
        if field.name not in columns:
            present = field.name in table.column_names
            columns[field.name] = (
                table.column(field.name).cast(field.type) if present else pa.nulls(len(hashes), field.type)
            )
    return pa.table(columns, schema=CATALOG_SCHEMA)


def export_catalog(
    shards: list[list[str]],
    catalog_root: Callable[[int], str],
    out: str,
    concurrency: int,
) -> dict:
    """One row per WARC with production's per-WARC counters and the device that scored it, sorted by path.

    Shards 0-403 ran the two-phase spec, whose catalog has no phase-A (``a_*``) counters; those are null.
    """
    path_of = {warc_hash(to_s3_url(p)): (s, p) for s, ps in enumerate(shards) for p in ps}
    by_root: dict[str, list[int]] = {}
    for s in range(len(shards)):
        by_root.setdefault(catalog_root(s), []).append(s)
    tables = []
    for root, shard_ids in by_root.items():
        for i in range(0, len(shard_ids), CATALOG_READ_BATCH):
            batch = shard_ids[i : i + CATALOG_READ_BATCH]
            raw = _cat_parquets(root, [f"catalog/shard-{s:05d}.parquet" for s in batch], concurrency)
            tables += [_catalog_rows(t, s, path_of) for s, t in zip(batch, raw, strict=True)]
            logger.info("catalog: %d/%d shards read from %s", i + len(batch), len(shard_ids), root)
    table = pa.concat_tables(tables).sort_by("warc_path")
    distinct = len(set(table.column("warc_hash").to_pylist()))
    if distinct != table.num_rows:
        raise ValueError(f"{table.num_rows - distinct} duplicate catalog rows")
    with fsspec.open(out, "wb", auto_mkdir=True) as f:
        pq.write_table(table, f, compression="zstd", compression_level=9)
    summary = {"rows": table.num_rows, "missing": len(path_of) - table.num_rows}
    logger.info("catalog: %s", summary)
    return summary
