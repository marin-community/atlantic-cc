# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Kept and tombstone files in production's exact format (fixture F3, 4ff1f9108b5a)."""

import gzip
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from atlantic_cc.io.local import LocalStore
from atlantic_cc.pipeline.outputs import (
    kept_parquet_bytes,
    kept_path,
    tombstones_jsonl_gz_bytes,
    tombstones_path,
    write_warc_outputs,
)
from atlantic_cc.schemas import KEPT_SCHEMA
from tests.parity.r2_asrun import FIXTURES, fetch_asrun

DATA = Path(__file__).parent / "data"
# Production's F3 kept file holds web text, so it is not committed: it is fetched from Marin's GCS once.
ASRUN_F3_DIR = Path(__file__).parents[3] / ".cache" / "asrun" / "f3"
ASRUN_F3_KEPT = ASRUN_F3_DIR / "kept-4ff1f9108b5a.parquet"
GZIP_FLAGS_BYTE = 3  # FNAME, FCOMMENT, ... bits; production's files carry no file name


@pytest.mark.gcs
def test_kept_file_is_byte_identical_to_production():
    if not ASRUN_F3_KEPT.exists():
        fetch_asrun(FIXTURES["F3"], str(ASRUN_F3_DIR))
    asrun = ASRUN_F3_KEPT.read_bytes()
    table = pq.read_table(io.BytesIO(asrun))
    assert table.num_rows == 2830
    assert kept_parquet_bytes(table) == asrun


def test_tombstone_file_rewrites_to_production_bytes():
    asrun = (DATA / "f3_4ff1f9108b5a_tombstones.jsonl.gz").read_bytes()
    rows = [json.loads(line) for line in gzip.decompress(asrun).decode().splitlines()]

    ours = tombstones_jsonl_gz_bytes(rows)

    assert gzip.decompress(ours) == gzip.decompress(asrun)
    assert ours[GZIP_FLAGS_BYTE] == asrun[GZIP_FLAGS_BYTE] == 0


def test_rewriting_a_warc_without_drops_removes_its_old_tombstones(tmp_path):
    store, root = LocalStore(), str(tmp_path)
    dropped = [{"doc_id": "d0", "pooled_prob": 0.01, "modernbert_prob": None}]
    write_warc_outputs(store, root, "abc123def456", KEPT_SCHEMA.empty_table(), dropped)

    write_warc_outputs(store, root, "abc123def456", KEPT_SCHEMA.empty_table(), [])

    assert store.stat(tombstones_path(root, "abc123def456")) is None
    assert pq.read_table(kept_path(root, "abc123def456")).num_rows == 0
