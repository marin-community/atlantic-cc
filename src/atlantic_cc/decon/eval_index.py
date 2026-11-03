# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Eval phrase index and bloom from a staged eval corpus, plus digests to verify a rebuilt corpus.

Ports ``bloom``/``bloom_val``/``_record_features`` of Marin ``decon_8m.py`` and ``_discover_eval_files`` of
``marin.datakit.decon`` (tag ``lpv11-8m-as-run``). A staged corpus is a tree of parquet files with
``id, text`` records: ``full`` = ``aa/<benchmark>/<split>.parquet`` + ``lmh/<task>/eval.parquet``, ``val`` =
``<name>/eval.parquet``. Each record contributes the set of its eval-side features
(:func:`atlantic_cc.decon.matching._extract_features` with the mark n-gram config); the index has one row
per (record, distinct feature).

Digests (shipped so a rebuilder can find exactly which records drifted, without shipping eval text):
per record ``(file, id, sha256(text))``; per file the record count, the sha256 over its ``id\\tsha256``
lines, and the sha256 of its sorted distinct feature hashes.
"""

import hashlib
import os
from collections.abc import Iterable, Iterator

import dupekit_native
import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from atlantic_cc.decon.matching import _bloom_hash, _extract_features
from atlantic_cc.decon.params import BLOOM_HEADROOM, FALSE_POSITIVE_RATE, MARK_NGRAM

INDEX_SCHEMA = pa.schema([("hash", pa.uint64()), ("eval_id", pa.string())])
DIGEST_SCHEMA = pa.schema([("file", pa.string()), ("id", pa.string()), ("text_sha256", pa.binary(32))])
INDEX_BATCH_ROWS = 1_000_000

# lm-eval tasks excluded from the decontamination corpus at read time (as-run
# ``prepare_eval_corpus.DECON_EXCLUDED_EVAL_TASKS``): their "documents" are ordinary public text.
DECON_EXCLUDED_EVAL_TASKS: frozenset[str] = frozenset(
    {
        "code2text_go",
        "code2text_java",
        "code2text_javascript",
        "code2text_php",
        "code2text_python",
        "code2text_ruby",
        "jsonschema_bench_easy",
        "jsonschema_bench_medium",
        "jsonschema_bench_hard",
        "swde",
        "realtoxicityprompts",
        "wikitext",
        "lambada_openai",
        "lambada_standard",
        "lambada_openai_cloze_yaml",
        "lambada_standard_cloze_yaml",
    }
)


def eval_record_files(root: str, excluded: frozenset[str] = DECON_EXCLUDED_EVAL_TASKS) -> list[str]:
    """Parquet record files under *root*: recursive, hidden directories and excluded task dirs skipped."""
    fs, resolved = fsspec.core.url_to_fs(root.rstrip("/"))
    protocol = root.split("://")[0] if "://" in root else ""
    out = []
    for walk_root, _dirs, files in fs.walk(resolved):
        rel = os.path.relpath(walk_root, resolved)
        if rel != "." and any(p.startswith(".") for p in rel.split(os.sep)):
            continue
        if os.path.basename(walk_root.rstrip("/")) in excluded:
            continue
        for name in files:
            if name.startswith(".") or not name.endswith(".parquet"):
                continue
            full = os.path.join(walk_root, name)
            out.append(f"{protocol}://{full}" if protocol else full)
    return sorted(out)


def iter_records(path: str) -> Iterator[tuple[str, str]]:
    """``(eval_id, text)`` of the non-empty records of one file, in file order (as-run id fallback)."""
    with fsspec.open(path, "rb") as fh:
        table = pq.read_table(fh)
    for idx, record in enumerate(table.to_pylist()):
        text = record.get("text")
        if text:
            yield str(record.get("id") or f"{path}::{idx}"), str(text)


def record_hashes(text: str) -> set[int]:
    return {_bloom_hash(f) for f in _extract_features(text, MARK_NGRAM)}


def index_table(rows: list[tuple[int, str]]) -> pa.Table:
    hashes, ids = zip(*rows, strict=True)
    return pa.table({"hash": pa.array(hashes, pa.uint64()), "eval_id": pa.array(ids, pa.string())}, schema=INDEX_SCHEMA)


def full_val_capacity(full_index_rows: int, val_index_rows: int) -> int:
    """As-run ``bloom_val`` sizing: ``int((full rows + val rows) * 1.25)`` (167,087,048 for v1.0)."""
    return int((full_index_rows + val_index_rows) * BLOOM_HEADROOM)


def _splitmix64(x: np.ndarray) -> np.ndarray:
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def _id_key(eval_id: str) -> int:
    return int.from_bytes(hashlib.blake2b(eval_id.encode(), digest_size=8).digest(), "little")


class IndexDigest:
    """Order-independent digest of a multiset of ``(hash, eval_id)`` rows (row count, wrapping sum, xor)."""

    def __init__(self) -> None:
        self.rows = 0
        self._sum = np.uint64(0)
        self._xor = np.uint64(0)
        self._ids: dict[str, int] = {}

    def update(self, hashes: np.ndarray, eval_ids: Iterable[str]) -> None:
        keys = np.fromiter((self._ids.setdefault(i, _id_key(i)) for i in eval_ids), dtype=np.uint64, count=hashes.size)
        mixed = _splitmix64(hashes.astype(np.uint64) ^ _splitmix64(keys))
        self._sum = np.uint64((int(self._sum) + int(mixed.sum(dtype=np.uint64))) % 2**64)
        self._xor ^= np.bitwise_xor.reduce(mixed) if mixed.size else np.uint64(0)
        self.rows += int(hashes.size)

    def update_batch(self, batch: pa.RecordBatch) -> None:
        ids = batch.column("eval_id").dictionary_encode()
        keys = np.array([_id_key(i) for i in ids.dictionary.to_pylist()], dtype=np.uint64)
        idx = ids.indices.to_numpy(zero_copy_only=False)
        mixed = _splitmix64(batch.column("hash").to_numpy().astype(np.uint64) ^ _splitmix64(keys[idx]))
        self._sum = np.uint64((int(self._sum) + int(mixed.sum(dtype=np.uint64))) % 2**64)
        self._xor ^= np.bitwise_xor.reduce(mixed) if mixed.size else np.uint64(0)
        self.rows += batch.num_rows

    def as_dict(self) -> dict:
        return {"rows": self.rows, "sum": f"{int(self._sum):016x}", "xor": f"{int(self._xor):016x}"}


def file_digest(path: str, label: str) -> tuple[dict, list[tuple[str, str, bytes]], list[tuple[str, np.ndarray]]]:
    """Digests of one eval file and its per-record hash sets.

    Returns ``(file summary, per-record (file, id, sha256(text)) rows, per-record (id, hashes))``.
    """
    lines = hashlib.sha256()
    records: list[tuple[str, str, bytes]] = []
    hash_sets: list[tuple[str, np.ndarray]] = []
    distinct: set[int] = set()
    for eval_id, text in iter_records(path):
        text_sha = hashlib.sha256(text.encode()).digest()
        lines.update(f"{eval_id}\t{text_sha.hex()}\n".encode())
        records.append((label, eval_id, text_sha))
        hashes = record_hashes(text)
        distinct |= hashes
        hash_sets.append((eval_id, np.fromiter(hashes, dtype=np.uint64, count=len(hashes))))
    summary = {
        "file": label,
        "records": len(records),
        "records_sha256": lines.hexdigest(),
        "index_rows": int(sum(h.size for _, h in hash_sets)),
        "distinct_features": len(distinct),
        "features_sha256": hashlib.sha256(np.array(sorted(distinct), dtype="<u8").tobytes()).hexdigest(),
    }
    return summary, records, hash_sets


def val_record_files(val_root: str) -> list[str]:
    fs, root = fsspec.core.url_to_fs(val_root.rstrip("/"))
    protocol = val_root.split("://", 1)[0] + "://" if "://" in val_root else ""
    return sorted(protocol + p for p in fs.glob(f"{root}/*/eval.parquet"))


def _write_index(path: str, records: Iterator[tuple[str, set[int]]], on_hash=None) -> int:
    """Write ``(hash, eval_id)`` rows, one per distinct feature per record (as-run batching)."""
    rows = 0
    batch: list[tuple[int, str]] = []
    with fsspec.open(path, "wb") as fh, pq.ParquetWriter(fh, INDEX_SCHEMA) as writer:
        for eval_id, hashes in records:
            for h in hashes:
                if on_hash is not None:
                    on_hash(h)
                batch.append((h, eval_id))
            if len(batch) >= INDEX_BATCH_ROWS:
                writer.write_table(index_table(batch))
                rows += len(batch)
                batch = []
        if batch:
            writer.write_table(index_table(batch))
            rows += len(batch)
    return rows


def _record_features(files: list[str]) -> Iterator[tuple[str, set[int]]]:
    for path in files:
        for eval_id, text in iter_records(path):
            yield eval_id, record_hashes(text)


def build_eval_index(full_root: str, val_root: str, out_dir: str) -> dict:
    """As-run ``bloom`` + ``bloom_val`` for the ``full-val`` eval set.

    Writes ``eval_hash_index.parquet`` (full corpus), ``val_index.parquet`` and ``filter.bin`` (the
    ``full-val`` bloom, capacity ``int((full rows + val rows) * 1.25)``, p = 1e-9).
    """
    full_files = eval_record_files(full_root)
    val_files = val_record_files(val_root)
    full_rows = _write_index(f"{out_dir}/eval_hash_index.parquet", _record_features(full_files))
    val_items = sum(len(hashes) for _, hashes in _record_features(val_files))
    capacity = full_val_capacity(full_rows, val_items)
    bf = dupekit_native.Bloom(capacity, FALSE_POSITIVE_RATE)
    for _, hashes in _record_features(full_files):
        for h in hashes:
            bf.add(h)
    val_rows = _write_index(f"{out_dir}/val_index.parquet", _record_features(val_files), on_hash=bf.add)
    data = bf.save_bytes()
    with fsspec.open(f"{out_dir}/filter.bin", "wb") as fh:
        fh.write(data)
    return {
        "full_files": len(full_files),
        "val_files": len(val_files),
        "full_index_rows": full_rows,
        "val_index_rows": val_rows,
        "capacity": capacity,
        "bloom_sha256": hashlib.sha256(data).hexdigest(),
    }


def corpus_digests(full_root: str, val_root: str) -> tuple[list[dict], pa.Table]:
    """Per-file digests and the per-record ``(file, id, text_sha256)`` table of a staged corpus."""
    labelled = [(p, "full/" + p.removeprefix(full_root.rstrip("/") + "/")) for p in eval_record_files(full_root)]
    labelled += [(p, "val/" + p.removeprefix(val_root.rstrip("/") + "/")) for p in val_record_files(val_root)]
    summaries, records = [], []
    for path, label in labelled:
        summary, rows, _ = file_digest(path, label)
        summaries.append(summary)
        records.extend(rows)
    table = pa.table(
        {
            "file": pa.array([r[0] for r in records], pa.string()),
            "id": pa.array([r[1] for r in records], pa.string()),
            "text_sha256": pa.array([r[2] for r in records], pa.binary(32)),
        },
        schema=DIGEST_SCHEMA,
    )
    return summaries, table


def compare_file_digests(rebuilt: list[dict], shipped: list[dict]) -> dict:
    """Which eval files of a rebuild differ from the as-run corpus (by records and by features)."""
    a = {d["file"]: d for d in rebuilt}
    b = {d["file"]: d for d in shipped}
    return {
        "files_rebuilt": len(a),
        "files_as_run": len(b),
        "missing": sorted(b.keys() - a.keys()),
        "extra": sorted(a.keys() - b.keys()),
        "records_differ": sorted(f for f in a.keys() & b.keys() if a[f]["records_sha256"] != b[f]["records_sha256"]),
        "features_differ": sorted(f for f in a.keys() & b.keys() if a[f]["features_sha256"] != b[f]["features_sha256"]),
    }


def compare_record_digests(rebuilt: pa.Table, shipped: pa.Table, max_examples: int = 20) -> dict:
    """Records whose text differs (same ``file``, ``id``), plus records only on one side."""

    def keyed(table: pa.Table) -> dict[tuple[str, str], bytes]:
        files = table["file"].cast(pa.string()).to_pylist()
        return dict(
            zip(zip(files, table["id"].to_pylist(), strict=True), table["text_sha256"].to_pylist(), strict=True)
        )

    a, b = keyed(rebuilt), keyed(shipped)
    changed = sorted(k for k in a.keys() & b.keys() if a[k] != b[k])
    return {
        "records_rebuilt": len(a),
        "records_as_run": len(b),
        "changed": len(changed),
        "missing": len(b.keys() - a.keys()),
        "extra": len(a.keys() - b.keys()),
        "examples_changed": [list(k) for k in changed[:max_examples]],
    }
