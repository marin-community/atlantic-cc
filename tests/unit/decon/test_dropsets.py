# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drop-set rules (docs/decon.md): per-source counting, local and global thresholds, the union D."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from atlantic_cc.decon import dropsets as ds
from atlantic_cc.decon.artifacts import DROP_HASHES_SHA256, load_drop_hashes, sha256_uint64
from atlantic_cc.decon.matching import _bloom_hash, _extract_features
from atlantic_cc.decon.params import DROP_HASH_COUNT, MARK_NGRAM

BOILER = "All rights reserved by the company and its partners worldwide since nineteen ninety nine"
OTHER = "Completely different words that never appear in any evaluation set at all or anywhere else"


def features(text: str) -> set[int]:
    return {_bloom_hash(f) for f in _extract_features(text, MARK_NGRAM)}


def test_drop_set_source_groups_cc_news_months():
    assert ds.drop_set_source("CC-NEWS-2024-05") == "CC-NEWS"
    assert ds.drop_set_source("CC-MAIN-2014-35") == "CC-MAIN-2014-35"
    assert ds.drop_set_source("CC-SUPPLEMENTAL-2026-22") == "CC-SUPPLEMENTAL-2026-22"


@pytest.mark.parametrize(("docs", "threshold"), [(0, 5), (999, 5), (1_199, 5), (1_200, 6), (5_000, 25)])
def test_local_threshold(docs, threshold):
    assert ds.local_threshold(docs) == threshold


def test_count_source_counts_bloom_hits_once_per_nonempty_doc_over_prefixes(monkeypatch):
    monkeypatch.setattr(ds, "SOURCE_DF_SAMPLE_DOCS", 2)
    monkeypatch.setattr(ds, "GLOBAL_DF_SAMPLE_DOCS", 4)
    bloom = features(BOILER)
    twice = f"{BOILER}\n\n{BOILER}"
    parts = [[twice, "", None, OTHER], [BOILER, BOILER], [BOILER]]
    counts = ds.count_source(iter(parts), bloom)
    assert (counts.docs, counts.local_docs) == (4, 2)
    assert set(counts.glob) == bloom  # OTHER contributes nothing: not in the bloom
    assert set(counts.glob.values()) == {3}  # docs 1, 3 and 4 (the doc repeating it counts once)
    assert set(counts.local.values()) == {1}  # only the first two docs are local


def test_docs_after_the_global_prefix_are_not_counted(monkeypatch):
    monkeypatch.setattr(ds, "GLOBAL_DF_SAMPLE_DOCS", 2)
    bloom = features(BOILER) | features(OTHER)
    counts = ds.count_source(iter([[BOILER], [BOILER], [OTHER], [OTHER]]), bloom)
    assert counts.docs == 2
    assert set(counts.glob) == features(BOILER)


def test_local_drop_list_and_counts_table_follow_counter_order():
    counts = ds.SourceCounts(
        local=__import__("collections").Counter({9: 30, 4: 2, 7: 25}),
        glob=__import__("collections").Counter({9: 40, 4: 2}),
        docs=5_000,
        local_docs=5_000,
    )
    assert ds.local_drop_hashes(counts) == [9, 7]
    table = ds.counts_table(counts)
    assert table.schema == pa.schema([("hash", pa.uint64()), ("df", pa.int64())])
    assert table.to_pylist() == [{"hash": 9, "df": 40}, {"hash": 4, "df": 2}]


def test_global_list_needs_df_50_and_three_sources():
    def table(rows):
        return pa.table(
            {"hash": pa.array([h for h, _ in rows], pa.uint64()), "df": pa.array([d for _, d in rows], pa.int64())}
        )

    tables = [table([(1, 20), (2, 48), (3, 49)]), table([(1, 20), (2, 1), (3, 1)]), table([(1, 10), (2, 1)])]
    assert sorted(ds.global_drop_hashes(tables).to_pylist()) == [1, 2]


def test_pool_and_code_lists():
    pool = pa.table({"hash": pa.array([1, 2, 3], pa.uint64()), "df": [1_000, 999, 5_000], "crawls": [3, 9, 2]})
    code = pa.table({"hash": pa.array([4, 5], pa.uint64()), "df": [100, 99], "languages": [1, 1]})
    assert ds.pool_common_hashes([pool]) == {1}
    assert ds.code_common_hashes([code]) == {4}
    union = ds.drop_set_union([[7, 1], [8]], {1}, {4})
    assert union.dtype == np.uint64 and union.tolist() == [1, 4, 7, 8]


def test_packaged_drop_hashes_match_the_pinned_digest():
    drop = load_drop_hashes()
    assert len(drop) == DROP_HASH_COUNT
    assert sha256_uint64(np.array(sorted(drop), dtype=np.uint64)) == DROP_HASHES_SHA256


def test_run_dropsets_writes_local_counts_and_global_lists(tmp_path):
    sample = tmp_path / "sample"
    for source, n in (("CC-MAIN-2014-23", 30), ("CC-MAIN-2015-11", 30), ("CC-NEWS", 30)):
        (sample / source).mkdir(parents=True)
        for part in range(2):
            texts = [f"{BOILER}\n\n{OTHER}" if i % 2 else OTHER for i in range(n)]
            pq.write_table(
                pa.table({"id": [str(i) for i in range(n)], "text": texts}),
                sample / source / f"part-{part:06d}.parquet",
            )
    out = tmp_path / "dropsets"
    bloom = features(BOILER)
    assert ds.run_dropsets(str(sample), bloom, str(out), workers=2) == len(bloom)  # 3 sources x 30 docs >= 50
    for source in ("CC-MAIN-2014-23", "CC-MAIN-2015-11", "CC-NEWS"):
        counts = pq.read_table(out / "_counts" / f"{source}.parquet").to_pylist()
        assert {r["df"] for r in counts} == {30}
        assert set(pq.read_table(out / source / "drop.parquet")["hash"].to_pylist()) == bloom  # 30 >= 5
    assert set(pq.read_table(out / "_global" / "drop.parquet")["hash"].to_pylist()) == bloom
