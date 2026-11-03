# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The port reproduces the as-run Marin code exactly on synthetic inputs (golden from the as-run code).

``tests/golden/decon/asrun_synthetic.json`` was produced by ``generate_asrun_goldens.py`` with the as-run
``decon_v4``/``decon_8m`` modules; this test recomputes everything with ``atlantic_cc.decon``.
"""

import hashlib
import json
from pathlib import Path

import dupekit_native
import pyarrow.parquet as pq
import pytest

from atlantic_cc.decon import dropsets as ds
from atlantic_cc.decon.matching import _bloom_hash, _document_overlap_and_matches, _extract_features
from atlantic_cc.decon.names import eval_name
from atlantic_cc.decon.params import FALSE_POSITIVE_RATE, FILTER_NGRAM, MARK_NGRAM
from tests.unit.decon import synthetic


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, separators=(",", ":")).encode()).hexdigest()


GOLDEN = Path(__file__).parents[2] / "golden" / "decon" / "asrun_synthetic.json"


@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN.read_text())


@pytest.fixture(scope="module")
def bloom(golden):
    features = [f for t in synthetic.eval_texts() for f in _extract_features(t, MARK_NGRAM)]
    assert digest(features) == golden["eval_features_sha256"]
    bf = dupekit_native.Bloom(golden["bloom_capacity"], FALSE_POSITIVE_RATE)
    phrase_hashes = sorted({_bloom_hash(f) for f in features})
    assert len(phrase_hashes) == golden["eval_phrases"]
    for h in phrase_hashes:
        bf.add(h)
    return bf, frozenset(phrase_hashes[:: golden["drop_every"]])


def test_document_decisions_match_as_run(golden, bloom):
    bf, drop = bloom
    matching = [
        {
            "filter": list(_document_overlap_and_matches(d, bf, FILTER_NGRAM, drop)),
            "mark": list(_document_overlap_and_matches(d, bf, MARK_NGRAM)),
        }
        for d in synthetic.documents()
    ]
    assert matching == golden["matching"]
    assert digest(matching) == golden["matching_sha256"]


def test_drop_set_counts_match_as_run_row_for_row(golden, bloom, tmp_path):
    bf, _ = bloom
    sources = synthetic.write_sample(tmp_path / "sample")
    tables = []
    for source in sources:
        counts = ds.count_source(ds.prefetched(ds.sample_parts(str(tmp_path / "sample"), source), ds.read_texts), bf)
        table = ds.counts_table(counts)
        tables.append(table)
        expected = golden["dropsets"][source]
        assert (counts.docs, counts.local_docs) == (expected["docs"], expected["local_docs"])
        assert digest([table["hash"].to_pylist(), table["df"].to_pylist()]) == expected["counts_sha256"]
        assert ds.local_drop_hashes(counts) == expected["drop"]
    assert sorted(ds.global_drop_hashes(tables).to_pylist()) == golden["global"]


def test_run_dropsets_matches_as_run(golden, bloom, tmp_path):
    bf, _ = bloom
    sources = synthetic.write_sample(tmp_path / "sample")
    out = tmp_path / "out"
    ds.run_dropsets(str(tmp_path / "sample"), bf, str(out), workers=2)
    for source in sources:
        assert pq.read_table(out / source / "drop.parquet")["hash"].to_pylist() == golden["dropsets"][source]["drop"]
    assert sorted(pq.read_table(out / "_global" / "drop.parquet")["hash"].to_pylist()) == golden["global"]


def test_eval_names_match_as_run(golden):
    assert {i: eval_name(i) for i in golden["eval_names"]} == golden["eval_names"]
