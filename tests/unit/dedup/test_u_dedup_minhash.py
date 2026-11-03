# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""U-DEDUP rule 2: dupekit CleanText -> MinHash(286, 5, 42) -> LSH(26) against the as-run values and the oracle."""

import hashlib
import json
import random
from pathlib import Path

import dupekit_native as dupekit
import numpy as np
import pyarrow as pa
import pytest

from atlantic_cc.dedup import oracle, text
from atlantic_cc.dedup.minhash import minhash_buckets
from tests.unit.dedup.test_u_dedup_text import GOLDEN_TEXTS

ORACLE_TEXTS = [
    "Hello, World! This is a MinHash test.",
    "abc",
    "",
    "!!!",
    "ΟΔΥΣΣΕΥΣ ΟΔΟΣ Σ naïve café — 日本語のテキスト 😀",
    "İstanbul ıi İI ß straße",
    "tab\tnew\nline   nbsp\u00a0 ideographic\u3000 x",
    "\x1c\x1d\x1e\x1f kept by Rust, whitespace to Python",
    "The quick brown fox jumps over the lazy dog. " * 20,
    "  leading and trailing  ",
]

GOLDEN = Path(__file__).parents[2] / "golden" / "dedup"


@pytest.fixture(scope="module")
def rules():
    return json.loads((GOLDEN / "rules.json").read_text())


@pytest.fixture(scope="module")
def multipliers():
    return oracle.coefficients()


@pytest.mark.parametrize("name", sorted(GOLDEN_TEXTS))
def test_buckets_match_the_as_run_values(rules, name):
    has_signature, buckets = minhash_buckets([text.compact_whitespace(GOLDEN_TEXTS[name])])
    assert has_signature.tolist() == [True]
    assert buckets.dtype == np.uint64 and buckets.shape == (1, 26)
    assert buckets[0][:3].tolist() == rules[name]["buckets_first3"]
    assert hashlib.sha256(np.ascontiguousarray(buckets[0]).tobytes()).hexdigest() == rules[name]["buckets_sha256"]


def test_native_pipeline_equals_the_pure_python_oracle(multipliers):
    rng = random.Random(1)
    alphabet = "abcdefghij ΣσςİıßÆ日本 \t\n.,!?"
    texts = ORACLE_TEXTS + ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, 120))) for _ in range(40)]
    _, native = minhash_buckets(texts)
    assert [oracle.buckets(t, multipliers) for t in texts] == native.tolist()


def test_clean_text_matches_rust():
    cleaned = dupekit.transform(
        pa.record_batch({"text": pa.array(ORACLE_TEXTS, type=pa.large_string())}),
        [dupekit.Transformation.CleanText(input_col="text", output_col="clean_text")],
    )
    assert cleaned.column("clean_text").to_pylist() == [oracle.clean_text(t) for t in ORACLE_TEXTS]


def test_band_hashes_are_not_salted_with_the_band_index():
    rng = np.random.default_rng(0)
    first = rng.integers(0, 2**63, 286, dtype=np.uint64)
    second = rng.integers(0, 2**63, 286, dtype=np.uint64)
    second[55:66] = first[0:11]  # band 5 of `second` holds band 0 of `first`
    signatures = pa.array([first.tolist(), second.tolist()], type=pa.list_(pa.uint64()))
    result = dupekit.transform(
        pa.record_batch({"signature": signatures}),
        [dupekit.Transformation.MinHashLSH(input_col="signature", output_col="buckets", num_bands=26)],
    )
    a, b = result.column("buckets").to_pylist()
    assert b[5] == a[0]
    assert oracle.lsh_buckets(second.tolist())[5] == oracle.lsh_buckets(first.tolist())[0] == a[0]


def test_texts_shorter_than_one_shingle_hash_the_whole_cleaned_string(multipliers):
    # "!!!" cleans to "", which still gets a signature: the hash of the empty string.
    _, buckets = minhash_buckets(["!!!", "", "abcd"])
    assert buckets[0].tolist() == buckets[1].tolist() == oracle.buckets("", multipliers)
    assert buckets[2].tolist() == oracle.lsh_buckets(oracle.signature("abcd", multipliers))
