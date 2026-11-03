# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""U-DEDUP rule 7 (records): output record bytes, field order and gzip framing match the as-run writer."""

import gzip
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from atlantic_cc.dedup import records
from atlantic_cc.dedup.params import CLUSTER_SIZE_FIELD, EXACT_COPIES_FIELD
from atlantic_cc.io.local import LocalStore
from atlantic_cc.schemas import KEPT_SCHEMA

GOLDEN = Path(__file__).parents[2] / "golden" / "dedup"
ROWS = [
    ("a", 1e-05, "plain text", 0.9802150130271912, float("nan")),
    ("b", float("inf"), "line\u2028sep\u2029par \x00 nul \x1f unit", float("nan"), 0.5),
    ("c", 1e16, 'ΟΔΥΣΣΕΥΣ İ ß 日本語 😀 "quoted" back\\slash', float("-inf"), 1.0),
    ("d", 0.1, "x" + " " * 300 + "y", 0.0, 0.0048),
]


@pytest.fixture(scope="module")
def golden():
    return json.loads((GOLDEN / "asrun_goldens.json").read_text())["records"]


def _kept_table() -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "doc_id": doc_id,
                "url": f"https://example.com/{doc_id}",
                "warc_hash": "0123456789ab",
                "snapshot": "CC-MAIN-2024-10",
                "fasttext_score": score,
                "text": text,
                "n_tokens": 7,
                "pooled_prob": pooled,
                "modernbert_prob": terminal,
            }
            for doc_id, score, text, pooled, terminal in ROWS
        ],
        schema=KEPT_SCHEMA,
    )


def _encoded(table: pa.Table) -> tuple[bytes, list[str]]:
    out, texts = records.output_records(table, np.arange(table.num_rows))
    for i, record in enumerate(out):
        record[EXACT_COPIES_FIELD] = i + 1
        record[CLUSTER_SIZE_FIELD] = 4_557_202_931 if i == 2 else 1
    return records.encode_jsonl(out), texts


def test_record_bytes_match_the_as_run_encoder(tmp_path, golden):
    path = tmp_path / "data-0123456789ab.parquet"
    pq.write_table(_kept_table(), path)
    payload, texts = _encoded(records.read_kept(LocalStore(), str(path)))
    assert payload.hex() == golden["jsonl_hex"]
    assert [len(t) for t in texts] == golden["text_lengths"]
    first = json.loads(payload.splitlines()[0])
    assert list(first) == [
        "doc_id",
        "url",
        "warc_hash",
        "snapshot",
        "fasttext_score",
        "text",
        "pooled_prob",
        "modernbert_prob",
        "exact_copies",
        "fuzzy_cluster_size",
    ]
    # float32 scores are widened to double and written in shortest round-trip form; NaN modernbert_prob is null.
    assert first["modernbert_prob"] is None and first["fasttext_score"] == float(np.float32(1e-05)) != 1e-05


def test_gzip_members_have_a_zero_mtime_and_concatenate():
    first, second = records.gzip_member(b'{"a":1}\n'), records.gzip_member(b'{"b":2}\n')
    # 1f8b, deflate, no flags, mtime 0, xfl 0; the OS byte depends on the zlib build (03 Linux, 13 macOS).
    assert first[:9] == bytes.fromhex("1f8b08000000000000")
    assert gzip.decompress(first + second) == b'{"a":1}\n{"b":2}\n'


def test_full_reads_require_the_kept_schema(tmp_path):
    path = tmp_path / "data-000000000000.parquet"
    pq.write_table(_kept_table().drop_columns(["pooled_prob"]), path)
    with pytest.raises(ValueError, match="kept schema"):
        records.read_kept(LocalStore(), str(path))
    assert records.read_kept(LocalStore(), str(path), ["text"]).num_rows == len(ROWS)
