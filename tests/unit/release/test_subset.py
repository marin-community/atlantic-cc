# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""U-REL: restricting a release to a WARC list, reading text only for row groups that hold a target."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from atlantic_cc.release import subset
from atlantic_cc.release.params import RELEASE_SCHEMA, VERSION
from atlantic_cc.warc.identity import snapshot_of

A = "crawl-data/CC-MAIN-2013-20/segments/1/warc/a.warc.gz"
B = "crawl-data/CC-MAIN-2013-20/segments/1/warc/b.warc.gz"
C = "crawl-data/CC-MAIN-2013-20/segments/1/warc/c.warc.gz"
N = "crawl-data/CC-NEWS/2016/08/CC-NEWS-20160826124520-00000.warc.gz"


def _rows(warcs: list[str], crawl: str, matches: list[str]) -> list[dict]:
    return [
        {
            "record_id": f"{w.rsplit('/', 1)[1]}#{i}",
            "url": f"https://x/{i}",
            "crawl": crawl,
            "warc_path": w,
            "text": "x" * (i + 1),
            "fasttext_score": 0.5,
            "pooled_prob": 0.9,
            "modernbert_prob": None,
            "exact_copies": 1,
            "fuzzy_cluster_size": 1,
            "decon_matches": matches,
        }
        for i, w in enumerate(warcs)
    ]


@pytest.fixture
def release(tmp_path) -> str:
    """A tiny release whose files have 2-row row groups, so row-group pruning is visible."""
    files = {
        "crawl=CC-MAIN-2013-20/part-00000-of-00001.parquet": _rows([A, A, A, B, B, C, C, C], "CC-MAIN-2013-20", []),
        "crawl=CC-MAIN-2013-20/contaminated-00000-of-00001.parquet": _rows([B], "CC-MAIN-2013-20", ["mmlu"]),
        "crawl=CC-NEWS-2016-08/part-00000-of-00001.parquet": _rows([N, N], "CC-NEWS-2016-08", []),
    }
    for name, rows in files.items():
        path = tmp_path / VERSION / name
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows, schema=RELEASE_SCHEMA), path, row_group_size=2)
    manifest = {
        "version": VERSION,
        "crawls": {
            "CC-MAIN-2013-20": {"part": {"records": 8, "files": 1}, "contaminated": {"records": 1, "files": 1}},
            "CC-NEWS-2016-08": {"part": {"records": 2, "files": 1}, "contaminated": {"records": 0, "files": 0}},
        },
    }
    (tmp_path / f"{VERSION}.manifest.json").write_text(json.dumps(manifest))
    return str(tmp_path)


def test_warc_lists_accept_relative_paths_and_s3_urls():
    assert subset.normalize_warc(f"s3://commoncrawl/{B}\n") == B
    assert subset.normalize_warc(B) == B
    with pytest.raises(ValueError, match="not a Common Crawl WARC path"):
        subset.normalize_warc("https://example.com/b.warc.gz")


def test_files_of_the_target_crawls_list_parts_before_the_contaminated_file(release):
    manifest = json.loads(open(f"{release}/{VERSION}.manifest.json").read())
    files = subset.subset_files(release, manifest, {"CC-MAIN-2013-20"})
    assert [(c, k, p.rsplit("/", 1)[1]) for c, k, p in files] == [
        ("CC-MAIN-2013-20", "part", "part-00000-of-00001.parquet"),
        ("CC-MAIN-2013-20", "contaminated", "contaminated-00000-of-00001.parquet"),
    ]


def test_scan_reads_only_the_row_groups_holding_a_target(release):
    subset._init_worker([B], None, False)
    result = subset.scan_file(
        "CC-MAIN-2013-20", "part", f"{release}/{VERSION}/crawl=CC-MAIN-2013-20/part-00000-of-00001.parquet"
    )
    assert (result["row_groups"], result["row_groups_read"]) == (4, 2)
    assert result["warcs"] == {B: [2, 4 + 5, 0]}  # two docs, texts of 4 and 5 characters, no tokenizer


def test_part_and_merge_count_docs_and_export_clean_rows(release, tmp_path):
    targets = [B, N]
    manifest = json.loads(open(f"{release}/{VERSION}.manifest.json").read())
    chunks = subset.chunk_files(subset.subset_files(release, manifest, {snapshot_of(w) for w in targets}))
    output, export = str(tmp_path / "out"), str(tmp_path / "export")
    subset.run_part(output, export, chunks, 0, 1, 1, targets, None)
    summary = subset.merge(output, release, len(chunks), targets, None)
    assert summary["total"]["docs"] == 2 + 1 + 2
    assert summary["by_kind"]["contaminated"]["docs"] == 1
    assert summary["per_warc"] == {B: [3, 4 + 5 + 1, 0], N: [2, 1 + 2, 0]}
    assert summary["warcs_without_docs"] == []
    exported = pq.read_table(subset.export_path(export, 0, len(chunks)))
    assert exported.schema.equals(RELEASE_SCHEMA)
    news = N.rsplit("/", 1)[1]
    assert sorted(exported.column("record_id").to_pylist()) == [f"{news}#0", f"{news}#1", "b.warc.gz#3", "b.warc.gz#4"]
