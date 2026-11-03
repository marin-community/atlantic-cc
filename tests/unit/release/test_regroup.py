# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""U-REL: the regroup steps on a synthetic dedup output (schema, kind split, decon tagging, part planning,
manifest totals, paths files, hive read)."""

import gzip
import json
import os

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from atlantic_cc.release import regroup
from atlantic_cc.release.params import RELEASE_SCHEMA, VERSION, Kind
from tests.unit.release import synthetic


def _rows(path: str) -> list[dict]:
    # Not pq.read_table(path): it infers a dictionary `crawl` partition from crawl=... and fails to merge
    # it with the `crawl` column (see test_release_tree_reads_as_a_hive_dataset).
    return pq.ParquetFile(path).read().to_pylist()


def test_release_row_renames_fields_and_adds_path_and_matches():
    record = synthetic.record(0, 1, synthetic.WARCS[2])
    row = regroup.release_row(record, synthetic.WARCS[2], ["mmlu"], RELEASE_SCHEMA)
    assert row["record_id"] == "doc-0-1"
    assert row["crawl"] == "CC-NEWS-2016-08"
    assert row["warc_path"] == synthetic.WARCS[2]
    assert row["decon_matches"] == ["mmlu"]
    assert row["modernbert_prob"] is None
    assert set(row) == set(RELEASE_SCHEMA.names)


def test_warc_paths_map_crawl_and_hash_to_relative_path(built):
    table = pq.read_table(f"{built.stage}/warc_paths.parquet")
    mapping = dict(zip(table.column("key").to_pylist(), table.column("warc_path").to_pylist(), strict=True))
    first = synthetic.record(0, 0, synthetic.WARCS[0])
    assert mapping[regroup.warc_key(first["snapshot"], first["warc_hash"])] == synthetic.WARCS[0]
    assert len(mapping) == len(synthetic.WARCS)


def test_pieces_split_records_by_crawl_and_kind_in_input_order(built):
    by_piece = {}
    for crawl in ("CC-MAIN-2013-20", "CC-MAIN-2026-17", "CC-NEWS-2016-08"):
        for kind in Kind:
            path = regroup.piece_path(built.stage, crawl, kind, 0)
            if os.path.exists(path):
                by_piece[(crawl, kind)] = _rows(path)
    expected: dict = {}
    for r in built.records:
        kind = Kind.CONTAMINATED if r["doc_id"] in synthetic.REMOVED else Kind.CLEAN
        expected.setdefault((r["snapshot"], kind), []).append(r["doc_id"])
    assert {k: [row["record_id"] for row in rows] for k, rows in by_piece.items()} == expected
    for (_, kind), rows in by_piece.items():
        for row in rows:
            want = synthetic.REMOVED[row["record_id"]][1] if kind is Kind.CONTAMINATED else []
            assert row["decon_matches"] == want
    assert pq.read_schema(regroup.piece_path(built.stage, "CC-NEWS-2016-08", Kind.CLEAN, 0)).equals(RELEASE_SCHEMA)


def test_done_marker_records_batch_totals(built):
    marker = json.loads(open(regroup.done_marker_path(built.stage, 0)).read())
    assert marker["records"] == len(built.records)
    assert marker["contaminated"] == len(synthetic.REMOVED)
    assert marker["pieces"] == [
        "CC-MAIN-2013-20/contaminated",
        "CC-MAIN-2013-20/part",
        "CC-MAIN-2026-17/part",
        "CC-NEWS-2016-08/contaminated",
        "CC-NEWS-2016-08/part",
    ]


def test_pieces_raise_when_a_removed_doc_is_not_found(tmp_path):
    fx = synthetic.build(tmp_path)
    removed = pq.read_table(f"{fx.decon_run}/removed_docs.parquet")
    extra = {"source_file": "data-000000-000.jsonl.gz", "doc_id": "not-in-the-file", "matched_evals": ["mmlu"]}
    pq.write_table(
        pa.Table.from_pylist([*removed.to_pylist(), extra], schema=removed.schema),
        f"{fx.decon_run}/removed_docs.parquet",
    )
    regroup.write_warc_paths(fx.worklist, fx.stage)
    with pytest.raises(RuntimeError, match="3 removed docs but 2 tagged"):
        regroup.pieces(fx.source, fx.decon_run, fx.stage, [], workers=1)


def test_batches_take_fifty_sorted_files_each():
    files = [f"gs://b/d/data-{i:06d}-000.jsonl.gz" for i in reversed(range(120))]
    batches = regroup.plan_batches(files)
    assert [(b, len(fs)) for b, fs in batches] == [(0, 50), (1, 50), (2, 20)]
    assert batches[1][1][0] == "gs://b/d/data-000050-000.jsonl.gz"


def test_plan_closes_a_part_once_its_pieces_reach_the_target():
    stage = "gs://b/stage"

    def piece(crawl, kind, batch):
        return f"{stage}/pieces/crawl={crawl}/kind={kind}/batch-{batch:06d}.parquet"

    listing = [
        (piece("CC-MAIN-2013-20", "part", 3), 400),
        (piece("CC-MAIN-2013-20", "part", 1), 600),
        (piece("CC-MAIN-2013-20", "part", 2), 500),
        (piece("CC-MAIN-2013-20", "part", 7), 999),
        (piece("CC-MAIN-2013-20", "contaminated", 2), 5),
        (piece("CC-NEWS-2016-08", "part", 9), 10),
    ]
    plan = regroup.plan_parts(listing, stage, part_bytes=1000)
    summary = [
        (p["crawl"], p["kind"], p["part"], p["of"], [x.rsplit("-", 1)[1][:6] for x in p["pieces"]]) for p in plan
    ]
    assert summary == [
        ("CC-MAIN-2013-20", "contaminated", 0, 1, ["000002"]),
        # 600 + 500 >= 1000 closes part 0; 400 + 999 closes part 1; nothing is left for a third part.
        ("CC-MAIN-2013-20", "part", 0, 2, ["000001", "000002"]),
        ("CC-MAIN-2013-20", "part", 1, 2, ["000003", "000007"]),
        ("CC-NEWS-2016-08", "part", 0, 1, ["000009"]),
    ]


def test_parts_concatenate_pieces_with_release_settings(built):
    part = f"{built.out}/{VERSION}/crawl=CC-MAIN-2013-20/part-00000-of-00001.parquet"
    md = pq.ParquetFile(part).metadata
    assert md.row_group(0).column(0).compression == "ZSTD"
    assert pq.read_schema(part).equals(RELEASE_SCHEMA)
    assert [r["record_id"] for r in _rows(part)] == [
        r["doc_id"]
        for r in built.records
        if r["snapshot"] == "CC-MAIN-2013-20" and r["doc_id"] not in synthetic.REMOVED
    ]


def test_manifest_counts_records_and_files_per_crawl_and_kind(built):
    manifest = json.loads(open(f"{built.out}/{VERSION}.manifest.json").read())
    assert manifest["records"] == 12
    assert manifest["by_kind"] == {"part": 10, "contaminated": 2}
    assert manifest["crawls"] == {
        "CC-MAIN-2013-20": {"part": {"records": 6, "files": 1}, "contaminated": {"records": 1, "files": 1}},
        "CC-MAIN-2026-17": {"part": {"records": 2, "files": 1}, "contaminated": {"records": 0, "files": 0}},
        "CC-NEWS-2016-08": {"part": {"records": 2, "files": 1}, "contaminated": {"records": 1, "files": 1}},
    }
    assert manifest["decontamination"] == {"method": "synthetic decon", "run": built.decon_run}


def test_manifest_raises_when_totals_disagree_with_the_decon_run(built):
    plan = json.loads(open(f"{built.stage}/parts_plan.json").read())
    rows = regroup.part_rows(built.out, VERSION, plan)
    export = dict(synthetic.EXPORT, summary={"decon/docs_kept": 11, "decon/docs_removed": 1})
    with pytest.raises(RuntimeError, match="release has 10 part records but the decon run has 11"):
        regroup.manifest_document(plan, rows, VERSION, export, built.decon_run)


def test_paths_files_list_keys_contaminated_first_and_clean_only(built):
    every = gzip.decompress(open(f"{built.out}/{VERSION}.parquet.paths.gz", "rb").read()).decode().split()
    clean = gzip.decompress(open(f"{built.out}/{VERSION}.clean.parquet.paths.gz", "rb").read()).decode().split()
    assert every == [
        "projects/atlantic-cc/v1.0/crawl=CC-MAIN-2013-20/contaminated-00000-of-00001.parquet",
        "projects/atlantic-cc/v1.0/crawl=CC-MAIN-2013-20/part-00000-of-00001.parquet",
        "projects/atlantic-cc/v1.0/crawl=CC-MAIN-2026-17/part-00000-of-00001.parquet",
        "projects/atlantic-cc/v1.0/crawl=CC-NEWS-2016-08/contaminated-00000-of-00001.parquet",
        "projects/atlantic-cc/v1.0/crawl=CC-NEWS-2016-08/part-00000-of-00001.parquet",
    ]
    assert clean == [k for k in every if "/part-" in k]
    assert not os.path.exists(f"{built.out}/{VERSION}/{VERSION}.manifest.json")


def test_paths_files_gzip_timestamp_is_a_parameter(built):
    plan = json.loads(open(f"{built.stage}/parts_plan.json").read())
    stamped = regroup.paths_files(plan, VERSION, 1_793_254_392)
    data = stamped[f"{VERSION}.parquet.paths.gz"]
    assert int.from_bytes(data[4:8], "little") == 1_793_254_392
    assert stamped == regroup.paths_files(plan, VERSION, 1_793_254_392)


def test_release_tree_reads_as_a_hive_dataset(built):
    dataset = ds.dataset(f"{built.out}/{VERSION}", format="parquet", partitioning="hive")
    table = dataset.to_table(filter=ds.field("crawl") == "CC-NEWS-2016-08")
    assert table.num_rows == 3
    assert set(table.column("crawl").to_pylist()) == {"CC-NEWS-2016-08"}
    assert pq.read_table(f"{built.out}/{VERSION}", partitioning=None).num_rows == 12


def test_read_table_default_partitioning_conflicts_with_the_crawl_column(built):
    """Documented in docs/release_format.md: pass partitioning=None (or use pyarrow.dataset / ParquetFile)."""
    with pytest.raises(pa.ArrowTypeError, match="Field crawl has incompatible types"):
        pq.read_table(f"{built.out}/{VERSION}")


def test_rerun_skips_finished_batches_and_parts(built):
    before = os.path.getmtime(regroup.piece_path(built.stage, "CC-NEWS-2016-08", Kind.CLEAN, 0))
    regroup.pieces(built.source, built.decon_run, built.stage, [], workers=1)
    regroup.parts(built.out, VERSION, built.stage, part_mb=1000, zstd_level=9, workers=1)
    assert os.path.getmtime(regroup.piece_path(built.stage, "CC-NEWS-2016-08", Kind.CLEAN, 0)) == before
