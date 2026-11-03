# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""U-DEDUP rule 1: blank filter, whitespace compaction, content ids and doc keys (B section 6(a) item 1)."""

import json
import random
from pathlib import Path

import dupekit_native as dupekit
import numpy as np
import pytest

from atlantic_cc.dedup import text

GOLDEN_TEXTS = {
    "hello": "hello world",
    "fox": "The quick brown fox jumps over the lazy dog.",
    "short": "abcd",
    "empty_after_clean": "!!!",
    "ws129": "a" + " " * 129 + "b",
    "ws128": "a" + " " * 128 + "b",
    "mixed_ws": "a" + "\t\n" * 100 + "b",
    "fs_sep": "a" + "\x1f" * 200 + "b",
}

GOLDEN = Path(__file__).parents[2] / "golden" / "dedup"


@pytest.fixture(scope="module")
def rules():
    return json.loads((GOLDEN / "rules.json").read_text())


BLANK_CASES = [
    (None, "None"),
    ("", "''"),
    (" ", "' '"),
    ("\u3000", "'\\u3000'"),
    ("\x1c", "'\\x1c'"),
    ("x", "'x'"),
]


@pytest.mark.parametrize(("value", "key"), BLANK_CASES)
def test_blank_text_is_dropped_by_python_strip(rules, value, key):
    assert text.is_kept_text(value) is rules["is_kept"][key]


@pytest.mark.parametrize("name", sorted(GOLDEN_TEXTS))
def test_compacted_length_and_content_id_match_the_as_run_values(rules, name):
    compacted = text.compact_whitespace(GOLDEN_TEXTS[name])
    assert len(compacted) == rules[name]["compacted_len"]
    assert format(text.content_id(compacted), "032x") == rules[name]["content_id"]


def test_whitespace_runs_keep_their_first_128_original_characters():
    assert text.compact_whitespace("a" + " " * 128 + "b") == "a" + " " * 128 + "b"
    assert text.compact_whitespace("a" + " " * 129 + "b") == "a" + " " * 128 + "b"
    assert text.compact_whitespace("a" + "\t\n" * 100 + "b") == "a" + "\t\n" * 64 + "b"
    # \x1f is whitespace to Python's re and str (not to Rust), and compaction follows Python.
    assert text.compact_whitespace("a" + "\x1f" * 200 + "b") == "a" + "\x1f" * 128 + "b"
    # U+200B is not whitespace anywhere.
    assert text.compact_whitespace("a" + "\u200b" * 300 + "b") == "a" + "\u200b" * 300 + "b"


def test_two_texts_that_compact_alike_share_a_content_id():
    assert text.content_id(text.compact_whitespace("a" + " " * 500 + "b")) == text.content_id("a" + " " * 128 + "b")
    assert format(text.content_id("hello world"), "032x") == "df8d09e93f874900a99b8775cc15b6c7"


def test_normalize_file_keeps_row_indices_of_non_blank_texts():
    run = " " * 300
    rows, normalized, compacted = text.normalize_file([None, "", "  \n\t ", "keep me", f"a{run}b", "keep me"])
    assert rows.tolist() == [3, 4, 5]
    assert normalized == ["keep me", "a" + " " * 128 + "b", "keep me"]
    assert compacted == 1


def test_doc_keys_pack_file_and_row(rules):
    keys = text.doc_keys(3, np.array([7, (1 << 24) - 1]))
    assert int(keys[0]) == rules["doc_key"]
    assert text.key_file(keys).tolist() == [3, 3]
    assert text.key_row(keys).tolist() == [7, (1 << 24) - 1]


def test_order_words_put_a_prefix_before_its_extensions():
    strings = ["12", "123", "13", "9", "1", "129", "0"]
    k1, k2, k3 = text.order_words(strings)
    assert [strings[i] for i in np.lexsort((k3, k2, k1))] == sorted(strings)


def test_canonical_order_keys_sort_like_the_decimal_strings_of_the_node_hash():
    rng = random.Random(0)
    ids = [rng.getrandbits(128) for _ in range(3000)]
    hi = np.array([i >> 64 for i in ids], dtype=np.uint64)
    lo = np.array([i & (2**64 - 1) for i in ids], dtype=np.uint64)
    k1, k2, k3 = text.canonical_order_keys(hi, lo)
    by_keys = np.lexsort((k3, k2, k1)).tolist()
    by_strings = sorted(range(len(ids)), key=lambda i: str(dupekit.hash_xxh3_128(f"source_000|{ids[i]:032x}".encode())))
    assert by_keys == by_strings
