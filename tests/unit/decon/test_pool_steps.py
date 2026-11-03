# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The as-run pool steps (mark -> candidate counts -> pool counts -> two-pass filter -> manifest) on a tiny pool.

Checks the decontamination identities (docs/decon.md) end to end: the two-pass filter equals the
single-pass rule, S' (candidate totals) is the union of mark matches, and replaying with S' equals
replaying with the bloom.
"""

import gzip
import json
import random

import dupekit_native
import pyarrow as pa
import pyarrow.parquet as pq

from atlantic_cc.decon import pool_steps
from atlantic_cc.decon.dropsets import partition_paths
from atlantic_cc.decon.filter import mark_matches, removal_matches
from atlantic_cc.decon.matching import _bloom_hash, _extract_features
from atlantic_cc.decon.params import FALSE_POSITIVE_RATE, MARK_NGRAM
from tests.unit.decon import synthetic


def _write_pool(root, docs: list[str]) -> list[dict]:
    rng = random.Random(3)
    records = [
        {
            "doc_id": f"d{i}",
            "text": t,
            "url": f"http://x/{i}",
            "snapshot": rng.choice(["CC-MAIN-2014-23", "CC-NEWS-2016-08"]),
        }
        for i, t in enumerate(docs)
    ]
    root.mkdir(parents=True)
    for shard in range(2):
        with gzip.open(root / f"data-00000{shard}-000.jsonl.gz", "wt", encoding="utf-8") as fh:
            for r in records[shard::2]:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return records


def test_pool_steps_reproduce_the_single_pass_rule(tmp_path, monkeypatch):
    evals = synthetic.eval_texts()
    phrase_hashes = sorted({_bloom_hash(f) for t in evals for f in _extract_features(t, MARK_NGRAM)})
    bf = dupekit_native.Bloom(100_000, FALSE_POSITIVE_RATE)
    for h in phrase_hashes:
        bf.add(h)
    drop = frozenset(phrase_hashes[::5])
    records = _write_pool(tmp_path / "pool", synthetic.documents(n=200))
    by_id = {r["doc_id"]: r for r in records}

    attributes, boilerplate = tmp_path / "attributes", tmp_path / "boilerplate"
    pool_steps.mark(str(tmp_path / "pool"), str(attributes), bf, workers=2)
    rows = [
        r for p in sorted((attributes / "outputs" / "main").glob("*.parquet")) for r in pq.read_table(p).to_pylist()
    ]
    assert len(rows) == len(records)
    for row in rows:
        score, matched = mark_matches(by_id[row["id"]]["text"], bf)
        assert (row["contaminated"], row["max_overlap"], row["matched_hashes"]) == (bool(matched), score, matched)

    pool_steps.candidate_counts(str(attributes), str(boilerplate), workers=2)
    totals = pa.concat_tables([pq.read_table(p) for p in partition_paths(str(boilerplate), "candidate_totals")])
    eval_hits = set(totals["hash"].to_pylist())
    assert eval_hits == {h for row in rows for h in row["matched_hashes"]}

    monkeypatch.setattr(pool_steps, "TRACK_MIN_CANDIDATE_DF", 1)
    summary = pool_steps.pool_counts(str(tmp_path / "pool"), str(boilerplate), workers=2)
    assert summary["tracked_phrases"] > 0
    pool_totals = pa.concat_tables([pq.read_table(p) for p in partition_paths(str(boilerplate), "pool_totals")])
    h0, df0 = pool_totals["hash"][0].as_py(), pool_totals["df"][0].as_py()
    assert df0 == sum(h0 in {_bloom_hash(f) for f in _extract_features(r["text"], MARK_NGRAM)} for r in records)

    out, removed = tmp_path / "decon", tmp_path / "removed"
    pool_steps.filter_pool(str(tmp_path / "pool"), str(attributes), str(out), str(removed), bf, drop, workers=2)
    sidecars = pa.concat_tables([pq.read_table(p) for p in sorted(removed.glob("*.parquet"))])
    two_pass = dict(zip(sidecars["doc_id"].to_pylist(), sidecars["matched_hashes"].to_pylist(), strict=True))
    single_pass = {r["doc_id"]: m for r in records if (m := removal_matches(r["text"], bf, drop))}
    replay = {r["doc_id"]: m for r in records if (m := removal_matches(r["text"], eval_hits, drop))}
    assert two_pass == single_pass == replay
    assert single_pass  # the synthetic docs do contain contamination
    kept = sum(1 for p in out.glob("*.jsonl.gz") for _ in gzip.open(p, "rt"))
    assert kept == len(records) - len(two_pass)

    index = tmp_path / "index.parquet"
    pairs = [(h, f"task{i % 3}-test-{i}") for i, h in enumerate(phrase_hashes)]
    pq.write_table(
        pa.table({"hash": pa.array([h for h, _ in pairs], pa.uint64()), "eval_id": [e for _, e in pairs]}), index
    )
    pool_steps.manifest(str(removed), [str(index)], str(tmp_path / "removed_docs.parquet"))
    manifest = pq.read_table(tmp_path / "removed_docs.parquet")
    assert manifest.num_rows == len(two_pass)
    assert all(names and names == sorted(names) for names in manifest["matched_evals"].to_pylist())


def test_sample_is_seeded_and_split_by_source(tmp_path):
    records = _write_pool(tmp_path / "pool", [f"doc {i} text here" for i in range(40)])
    for name in ("a", "b"):
        pool_steps.sample(str(tmp_path / "pool"), str(tmp_path / name), n_files=2, per_file=7, workers=1)
    parts = sorted(p.relative_to(tmp_path / "a") for p in (tmp_path / "a").rglob("*.parquet"))
    assert {p.parts[0] for p in parts} <= {"CC-MAIN-2014-23", "CC-NEWS"}
    for p in parts:
        assert pq.read_table(tmp_path / "a" / p).equals(pq.read_table(tmp_path / "b" / p))
    sampled = sum(pq.read_table(tmp_path / "a" / p).num_rows for p in parts)
    assert sampled == 14 and len(records) == 40
