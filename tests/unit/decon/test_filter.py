# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-pass filter over release parquet and deduplicated jsonl.gz shards."""

import gzip
import json

import pyarrow as pa
import pyarrow.parquet as pq

from atlantic_cc.decon.filter import FLAGGED_SCHEMA, InputKind, decon_files
from atlantic_cc.decon.matching import _bloom_hash, _extract_features
from atlantic_cc.decon.params import MARK_NGRAM

LEAK = (
    "A train leaves the station at nine travelling sixty miles per hour while a second train leaves "
    "the opposite station at ten travelling eighty miles per hour toward the first"
)
FILLER = (
    "The community garden opened its new greenhouse this spring with volunteers planting tomatoes "
    "peppers and herbs along the southern wall near the old brick shed"
)


def phrases(text: str) -> set[int]:
    return {_bloom_hash(f) for f in _extract_features(text, MARK_NGRAM)}


def test_release_files_are_flagged_named_and_compared(tmp_path):
    membership = phrases(LEAK)
    names_of = {h: ["gsm8k", "mmlu"] for h in membership}
    texts = [f"Homework\n\n{LEAK}", FILLER, f"{FILLER}\n\n{LEAK}", LEAK.upper()]
    stored = [["gsm8k", "mmlu"], [], ["mmlu"], []]  # the third row disagrees on purpose
    table = pa.table(
        {
            "record_id": [f"r{i}" for i in range(4)],
            "text": texts,
            "decon_matches": pa.array(stored, pa.list_(pa.string())),
        }
    )
    pq.write_table(table, tmp_path / "part-00000-of-00001.parquet")
    out = tmp_path / "out"
    [summary] = decon_files(
        [str(tmp_path / "part-00000-of-00001.parquet")],
        InputKind.RELEASE,
        str(out),
        membership,
        frozenset(),
        names_of,
        1,
    )
    assert (summary["docs"], summary["flagged"], summary["stored_flagged"]) == (4, 2, 2)
    assert summary["mismatched_ids"] == ["r2"]
    flagged = pq.read_table(out / "part-00000-of-00001.decon.parquet")
    assert flagged.schema == FLAGGED_SCHEMA
    assert flagged["id"].to_pylist() == ["r0", "r2"]
    assert flagged["decon_matches"].to_pylist() == [["gsm8k", "mmlu"], ["gsm8k", "mmlu"]]
    assert json.loads((out / "part-00000-of-00001.decon.json").read_text())["flagged"] == 2


def test_dedup_shards_and_drop_hashes(tmp_path):
    membership = phrases(LEAK)
    path = tmp_path / "data-000000-000.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for i, text in enumerate([f"Homework\n\n{LEAK}", FILLER]):
            fh.write(json.dumps({"doc_id": f"d{i}", "text": text, "snapshot": "CC-MAIN-2014-23"}) + "\n")
    out = tmp_path / "out"
    [kept] = decon_files([str(path)], InputKind.DEDUP, str(out), membership, frozenset(membership), {}, 2)
    assert kept["flagged"] == 0
    [flagged] = decon_files([str(path)], InputKind.DEDUP, str(out), membership, frozenset(), {}, 1)
    assert (flagged["docs"], flagged["flagged"], flagged["matched_hashes_without_names"]) == (2, 1, len(membership))
    assert pq.read_table(out / "data-000000-000.decon.parquet")["decon_matches"].to_pylist() == [[]]
