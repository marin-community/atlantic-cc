# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate ``asrun_synthetic.json`` by running the as-run Marin decon code on the synthetic inputs.

Run inside a Marin checkout at tag ``lpv11-8m-as-run`` (the code is read, never modified)::

    cd marin && uv run python <atlantic-cc>/tests/golden/decon/generate_asrun_goldens.py <atlantic-cc>
"""

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import dupekit
import pyarrow as pa
import pyarrow.compute as pc

ATLANTIC_CC = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ATLANTIC_CC / "tests" / "unit" / "decon"))

import synthetic  # noqa: E402
from experiments.fast_curation import decon_8m as asrun  # noqa: E402
from experiments.fast_curation.decon_v4 import (  # noqa: E402
    _bloom_hash,
    _document_overlap_and_matches,
    _extract_features,
)

BLOOM_CAPACITY = 100_000
EVAL_IDS = [
    "a-test-1",
    "b-dev-0001",
    "c-train-x",
    "d-0123456789abcdef",
    "e-0123456789abcde",
    "f-validation-ff",
    "g--3",
    "-4",
]


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    evals = synthetic.eval_texts()
    features = [f for t in evals for f in _extract_features(t, asrun.MARK_NGRAM)]
    phrase_hashes = sorted({_bloom_hash(f) for f in features})
    bf = dupekit.Bloom(BLOOM_CAPACITY, asrun.FALSE_POSITIVE_RATE)
    for h in phrase_hashes:
        bf.add(h)
    drop = frozenset(phrase_hashes[::5])

    docs = synthetic.documents()
    matching = [
        {
            "filter": list(_document_overlap_and_matches(d, bf, asrun.FILTER_NGRAM, drop)),
            "mark": list(_document_overlap_and_matches(d, bf, asrun.MARK_NGRAM)),
        }
        for d in docs
    ]

    with tempfile.TemporaryDirectory() as tmp:
        pool, sample = Path(tmp) / "pool", Path(tmp) / "sample"
        sources = synthetic.write_sample(sample)
        bloom_path = Path(asrun._bloom_path(str(pool), asrun.EvalSet.FULL_VAL))
        bloom_path.parent.mkdir(parents=True)
        bloom_path.write_bytes(bf.save_bytes())
        count = asrun._count_source(str(pool), str(sample), [asrun.EvalSet.FULL_VAL])
        per_source = {row["source"]: row for row in count(iter(sources), None)}
        root = asrun._dropsets_dir(str(pool), asrun.EvalSet.FULL_VAL)
        tables = []
        for source in sources:
            counts = asrun._read_parquet(f"{root}/_counts/{source}.parquet")
            tables.append(counts)
            drop_list = asrun._read_parquet(f"{root}/{source}/drop.parquet")["hash"].to_pylist()
            per_source[source] |= {
                "counts_rows": counts.num_rows,
                "counts_sha256": digest([counts["hash"].to_pylist(), counts["df"].to_pylist()]),
                "drop": drop_list,
            }
        grouped = pa.concat_tables(tables).group_by("hash").aggregate([("df", "sum"), ("df", "count")])
        common = grouped.filter(
            pc.and_(
                pc.greater_equal(grouped["df_sum"], asrun.GLOBAL_DF_COMMON_MIN_ABS),
                pc.greater_equal(grouped["df_count"], asrun.GLOBAL_DF_COMMON_MIN_SOURCES),
            )
        )
        global_list = sorted(common["hash"].to_pylist())

    golden = {
        "generated_by": "tests/golden/decon/generate_asrun_goldens.py at Marin tag lpv11-8m-as-run",
        "bloom_capacity": BLOOM_CAPACITY,
        "eval_features_sha256": digest(features),
        "eval_phrases": len(phrase_hashes),
        "drop_every": 5,
        "matching_sha256": digest(matching),
        "filter_flagged": sum(bool(m["filter"][1]) for m in matching),
        "mark_flagged": sum(bool(m["mark"][1]) for m in matching),
        "matching": matching,
        "dropsets": per_source,
        "global": global_list,
        "eval_names": {i: asrun.eval_name(i) for i in EVAL_IDS},
    }
    out = ATLANTIC_CC / "tests" / "golden" / "decon" / "asrun_synthetic.json"
    out.write_text(json.dumps(golden, indent=None, separators=(",", ":")) + "\n")
    summary = {k: golden[k] for k in ("eval_phrases", "filter_flagged", "mark_flagged")}
    print(out, out.stat().st_size, summary, len(global_list))


if __name__ == "__main__":
    main()
