# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A tiny synthetic dedup output + decontamination run, shaped like production's inputs to the regroup."""

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from atlantic_cc.warc.identity import snapshot_of, warc_hash
from atlantic_cc.worklist.frozen import write_paths_file
from atlantic_cc.worklist.identity import to_s3_url

WARCS = (
    "crawl-data/CC-MAIN-2013-20/segments/1/warc/CC-MAIN-a-00000.warc.gz",
    "crawl-data/CC-MAIN-2013-20/segments/1/warc/CC-MAIN-a-00001.warc.gz",
    "crawl-data/CC-NEWS/2016/08/CC-NEWS-20160826124520-00000.warc.gz",
    "crawl-data/CC-MAIN-2026-17/segments/2/warc/CC-MAIN-b-00000.warc.gz",
)
# doc_id -> (dedup file, evals the decontamination filter matched)
REMOVED = {
    "doc-0-1": ("data-000000-000.jsonl.gz", ["paloma_4chan"]),
    "doc-1-1": ("data-000001-000.jsonl.gz", ["mmlu", "paloma_c4_100_domains"]),
}
EXPORT = {
    "method": "synthetic decon",
    "summary": {"decon/docs_kept": 10, "decon/docs_removed": 2},
}


@dataclass(frozen=True)
class Fixture:
    root: Path
    worklist: str
    source: str
    decon_run: str
    stage: str
    out: str
    records: list[dict]  # every dedup record, in file order


def record(i_file: int, i_row: int, warc: str) -> dict:
    return {
        "doc_id": f"doc-{i_file}-{i_row}",
        "url": f"https://example.com/{i_file}/{i_row}",
        "warc_hash": warc_hash(to_s3_url(warc)),
        "snapshot": snapshot_of(warc),
        "fasttext_score": 0.5 + i_row / 100,
        "text": f"text of document {i_file}-{i_row}\nsecond line",
        "pooled_prob": 0.9,
        "modernbert_prob": None if i_row % 2 else 0.75,
        "exact_copies": 1 + i_row,
        "fuzzy_cluster_size": 4_557_202_931 if i_row == 0 else 1,
    }


def build(root: Path) -> Fixture:
    """Two dedup files of 6 records each (crawls interleaved, as in rdedup output) and 2 removed docs.

    Per crawl: CC-MAIN-2013-20 6 clean + 1 contaminated, CC-MAIN-2026-17 2 clean, CC-NEWS-2016-08 2 clean +
    1 contaminated.
    """
    source = root / "deduped_c075"
    source.mkdir(parents=True)
    records = []
    for i_file in range(2):
        rows = [record(i_file, i_row, WARCS[(i_file + i_row) % len(WARCS)]) for i_row in range(6)]
        records += rows
        payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
        (source / f"data-{i_file:06d}-000.jsonl.gz").write_bytes(gzip.compress(payload, mtime=0))
    decon_run = root / "decon_run"
    decon_run.mkdir()
    removed = pa.table(
        {
            "source_file": [f for f, _ in REMOVED.values()],
            "doc_id": list(REMOVED),
            "matched_evals": [m for _, m in REMOVED.values()],
        }
    )
    pq.write_table(removed, decon_run / "removed_docs.parquet")
    (decon_run / "export.json").write_text(json.dumps(EXPORT))
    worklist = root / "worklist.paths.gz"
    write_paths_file(sorted(WARCS), str(worklist))
    return Fixture(
        root, str(worklist), str(source), str(decon_run), str(root / "stage"), str(root / "release"), records
    )
