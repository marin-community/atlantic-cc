# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reading kept files and encoding output records (as-run ``rdedup/docs.py:140-221`` and ``stages.py:699-714``).

An output record is the kept row minus ``input_ids``/``n_tokens``, in kept column order, with the compacted text,
``modernbert_prob`` NaN as ``None``, then ``exact_copies`` and ``fuzzy_cluster_size``. Records are msgspec JSON
lines (float32 widened to double, shortest round-trip repr; any other NaN/Inf is ``null``). Each kept file's
surviving rows become one gzip member (level 6, mtime 0); the gzip header's OS byte comes from the zlib build
(03 on Linux), so compare outputs decompressed across platforms.
"""

import gzip
import io
import math

import msgspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from atlantic_cc.dedup.params import DROP_COLUMNS, GZIP_LEVEL
from atlantic_cc.dedup.text import compact_whitespace
from atlantic_cc.io.store import Store
from atlantic_cc.schemas import KEPT_SCHEMA

_JSON = msgspec.json.Encoder()
# What a kept file holds once the token columns are dropped; output records carry exactly these fields, in order.
KEPT_DEDUP_SCHEMA = pa.schema([field for field in KEPT_SCHEMA if field.name not in DROP_COLUMNS])


def read_kept(store: Store, path: str, columns: list[str] | None = None) -> pa.Table:
    """One ``kept/`` parquet file without ``DROP_COLUMNS``. A full read must have the kept schema."""
    table = pq.read_table(io.BytesIO(store.read(path)), columns=columns)
    table = table.drop_columns([name for name in DROP_COLUMNS if name in table.column_names])
    if columns is None and not table.schema.equals(KEPT_DEDUP_SCHEMA):
        raise ValueError(f"{path}: schema {table.schema} is not the kept schema {KEPT_DEDUP_SCHEMA}")
    return table


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value


def output_records(table: pa.Table, rows: np.ndarray) -> tuple[list[dict], list[str]]:
    """Output dicts for ``rows`` of a kept table and their compacted texts."""
    taken = table.take(pa.array(rows, type=pa.int64()))
    records = taken.to_pylist()
    texts = []
    for record in records:
        record["text"] = compact_whitespace(record["text"])
        if "modernbert_prob" in record:
            record["modernbert_prob"] = _finite_or_none(record["modernbert_prob"])
        texts.append(record["text"])
    return records, texts


def encode_jsonl(records: list[dict]) -> bytes:
    """JSONL bytes as the as-run writer produced them (msgspec; NaN encodes as null)."""
    return b"".join(_JSON.encode(r) + b"\n" for r in records)


def gzip_member(payload: bytes) -> bytes:
    return gzip.compress(payload, compresslevel=GZIP_LEVEL, mtime=0)
