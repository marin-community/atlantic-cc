# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-pass decontamination: flag documents with the v4 rule and name the evals they match.

The as-run pipeline marked every document (one required phrase, no drop set) and then re-scored only
the mark's candidates with the filter rule. Every drop hash is an eval phrase and the filter rule is
stricter than the mark, so scoring every document once with the filter rule gives the same result
(``docs/decon.md``). Membership can be the ``full-val`` bloom or the exact set S' of every eval phrase
the mark matched; on the released text both give identical decisions and matched hashes.

Inputs: release parquet (``record_id``, ``text``, ``decon_matches``) or deduplicated ``*.jsonl.gz``
records (``doc_id``, ``text``). Output per input file: a parquet of the flagged documents
(``id``, ``matched_hashes``, ``decon_matches``) and a JSON summary.
"""

import gzip
import json
import logging
import multiprocessing
import os
from collections.abc import Container, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq

from atlantic_cc.decon.matching import _document_overlap_and_matches
from atlantic_cc.decon.names import decon_matches
from atlantic_cc.decon.params import FILTER_NGRAM, MARK_NGRAM

logger = logging.getLogger(__name__)

FLAGGED_SCHEMA = pa.schema(
    [("id", pa.string()), ("matched_hashes", pa.list_(pa.uint64())), ("decon_matches", pa.list_(pa.string()))]
)
RELEASE_BATCH_ROWS = 4_096


class InputKind(StrEnum):
    RELEASE = "release"
    """Atlantic-CC release parquet: ``record_id``, ``text``, ``decon_matches``."""
    DEDUP = "dedup"
    """Deduplicated pool shards (``deduped_c075/*.jsonl.gz``): ``doc_id``, ``text``."""


def removal_matches(text: str, membership: Container[int], drop_hashes: frozenset[int]) -> list[int]:
    """Matched eval phrases that flag *text* (empty: clean). As-run ``removal_matches`` with ``Scoring.V4``."""
    return _document_overlap_and_matches(text, membership, FILTER_NGRAM, drop_hashes)[1]


def mark_matches(text: str, membership: Container[int]) -> tuple[float, list[int]]:
    """As-run mark: one required phrase, no drop set. Returns ``(max_overlap, matched_hashes)``."""
    return _document_overlap_and_matches(text, membership, MARK_NGRAM)


def iter_release(path: str) -> Iterator[tuple[str, str, list[str]]]:
    """``(record_id, text, decon_matches)`` of a release parquet file, in file order."""
    with fsspec.open(path, "rb") as fh:
        parquet = pq.ParquetFile(fh)
        for batch in parquet.iter_batches(
            columns=["record_id", "text", "decon_matches"], batch_size=RELEASE_BATCH_ROWS
        ):
            yield from zip(
                batch.column("record_id").to_pylist(),
                batch.column("text").to_pylist(),
                batch.column("decon_matches").to_pylist(),
                strict=True,
            )


def iter_dedup(path: str) -> Iterator[tuple[str, str, None]]:
    """``(doc_id, text, None)`` of a deduplicated ``*.jsonl.gz`` shard, in file order."""
    with fsspec.open(path, "rb") as raw, gzip.open(raw, "rt", encoding="utf-8") as lines:
        for line in lines:
            record = json.loads(line)
            yield record["doc_id"], record["text"], None


@dataclass
class FileResult:
    """Per-file outcome; ``mismatched_ids`` lists release records whose stored ``decon_matches`` differ."""

    path: str
    docs: int = 0
    flagged: int = 0
    matched_hashes_without_names: int = 0
    stored_flagged: int = 0
    mismatched: int = 0
    mismatched_ids: list[str] = field(default_factory=list)


def decon_records(
    records: Iterator[tuple[str, str, Sequence[str] | None]],
    membership: Container[int],
    drop_hashes: frozenset[int],
    names_of: Mapping[int, Sequence[str]],
    result: FileResult,
) -> list[dict]:
    """Flag records; returns the flagged rows and fills *result* (comparing to stored names if given)."""
    rows = []
    for record_id, text, stored in records:
        result.docs += 1
        matched = removal_matches(text, membership, drop_hashes)
        names = decon_matches(matched, names_of)
        result.matched_hashes_without_names += sum(h not in names_of for h in matched)
        if matched:
            result.flagged += 1
            rows.append({"id": record_id, "matched_hashes": matched, "decon_matches": names})
        if stored is None:
            continue
        result.stored_flagged += bool(stored)
        if list(stored) != names:
            result.mismatched += 1
            result.mismatched_ids.append(record_id)
    return rows


_STATE: dict = {}


def output_name(path: str) -> str:
    """Output stem of an input file; keeps a hive partition directory (``crawl=X/part-...``) so names stay unique."""
    base = os.path.basename(path)
    for suffix in (".jsonl.gz", ".parquet"):
        base = base.removesuffix(suffix)
    parent = os.path.basename(os.path.dirname(path))
    return f"{parent}/{base}" if "=" in parent else base


def _decon_file(args: tuple[str, InputKind, str]) -> dict:
    path, kind, out_dir = args
    records = iter_release(path) if kind is InputKind.RELEASE else iter_dedup(path)
    result = FileResult(path=path)
    rows = decon_records(records, _STATE["membership"], _STATE["drop"], _STATE["names"], result)
    name = output_name(path)
    with fsspec.open(f"{out_dir}/{name}.decon.parquet", "wb") as fh:
        pq.write_table(pa.Table.from_pylist(rows, schema=FLAGGED_SCHEMA), fh, compression="zstd")
    summary = asdict(result)
    with fsspec.open(f"{out_dir}/{name}.decon.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    return summary


def decon_files(
    paths: list[str],
    kind: InputKind,
    out_dir: str,
    membership: Container[int],
    drop_hashes: frozenset[int],
    names_of: Mapping[int, Sequence[str]],
    workers: int,
) -> list[dict]:
    """Flag every file of *paths* with a fork-based process pool sharing *membership* copy-on-write."""
    _STATE.update(membership=membership, drop=drop_hashes, names=names_of)
    names = [output_name(p) for p in paths]
    if len(set(names)) != len(names):
        raise ValueError("input files map to the same output name; pass files from distinct directories")
    tasks = [(p, kind, out_dir.rstrip("/")) for p in paths]
    if workers <= 1:
        return [_decon_file(t) for t in tasks]
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        return list(pool.imap(_decon_file, tasks))
