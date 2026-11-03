# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""U-DECON items 1-5 and 8-9: the v4 matching rule, pinned with known answers from the as-run code."""

import random

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from atlantic_cc.decon import matching
from atlantic_cc.decon.filter import mark_matches, removal_matches
from atlantic_cc.decon.matching import (
    NGramConfig,
    _bloom_hash,
    _document_overlap_and_matches,
    _extract_features,
    _extract_ngrams,
    _iter_paragraphs,
)
from atlantic_cc.decon.params import FILTER_NGRAM, MARK_NGRAM

LEAK = (
    "A train leaves the station at nine travelling sixty miles per hour while a second train leaves "
    "the opposite station at ten travelling eighty miles per hour toward the first"
)
FILLER = (
    "The community garden opened its new greenhouse this spring with volunteers planting tomatoes "
    "peppers and herbs along the southern wall near the old brick shed"
)
MCQ = "Which planet is largest?\n\nA. Mars\n\nB. Jupiter\n\nC. Venus\n\nD. Mercury\n\nAnswer: B"


def words(prefix: str, n: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


E13 = words("ev", 13)  # exactly one 13-gram
E14 = words("ev", 14)  # two 13-grams
FILL = words("zz", 40)


def phrases(*texts: str, ngram: NGramConfig = MARK_NGRAM) -> set[int]:
    """Eval-side phrase set, built exactly like the as-run blooms and index."""
    return {_bloom_hash(f) for t in texts for f in _extract_features(t, ngram)}


def removed(doc: str, membership: set[int], drop: frozenset[int] = frozenset()) -> bool:
    return bool(removal_matches(doc, membership, drop))


# ---- 1. hash ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("feature", "expected"),
    [
        ("a b c", 3516195607690045812),
        ("Lana Del Rey sings", 7287079750719085765),
        ("Janet’s ducks lay 16 eggs per day. She eats three for breakfast every", 11484006392317508080),
    ],
)
def test_blake2b_64_known_answers(feature, expected):
    assert _bloom_hash(feature) == expected


def test_hashes_round_trip_through_parquet_as_uint64(tmp_path):
    hashes = [11484006392317508080, 3516195607690045812, 2**64 - 1]
    pq.write_table(pa.table({"hash": pa.array(hashes, pa.uint64())}), tmp_path / "h.parquet")
    assert pq.read_table(tmp_path / "h.parquet")["hash"].to_pylist() == hashes


# ---- 2. tokenization -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "tokens"),
    [("a\x1cb", ["a", "b"]), ("a\x1fb", ["a", "b"]), ("a\xa0b", ["a", "b"]), ("a​b", ["a​b"])],
    ids=["file-separator", "unit-separator", "nbsp", "zero-width-space"],
)
def test_tokens_split_on_python_whitespace(text, tokens):
    assert list(_extract_features(text + " c d", MARK_NGRAM)) == [" ".join([*tokens, "c", "d"])]


def test_matching_is_case_and_punctuation_sensitive():
    membership = phrases(E14)
    assert removed(E14 + "\n\n" + FILL, membership)
    assert not removed(E14.upper() + "\n\n" + FILL, membership)
    assert not removed(E14.replace("ev3 ", "ev3, ") + "\n\n" + FILL, membership)


def test_letterless_ngrams_emit_no_features():
    numbers = " ".join(str(i) for i in range(30))
    assert list(_extract_features(numbers, MARK_NGRAM)) == []
    assert list(_extract_ngrams(numbers + " x", 13, 0)) == [" ".join([*numbers.split()[-12:], "x"])]


def test_crlf_is_not_a_paragraph_break():
    assert list(_iter_paragraphs("a\r\n\r\nb", "\n\n")) == ["a\r\n\r\nb"]
    assert list(_iter_paragraphs("a\n\n\nb\n\n", "\n\n")) == ["a", "\nb"]


# ---- 3. eval-side features -------------------------------------------------------------------------


def test_short_record_is_one_exact_feature():
    assert list(_extract_features("Lana Del Rey sings", MARK_NGRAM)) == ["Lana Del Rey sings"]


def test_short_paragraph_inside_long_record_is_not_a_feature():
    long_match = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
    assert list(_extract_features(f"{long_match}\n\nSee comment above.", MARK_NGRAM)) == [
        "one two three four five six seven eight nine ten eleven twelve thirteen",
        "two three four five six seven eight nine ten eleven twelve thirteen fourteen",
    ]


def test_all_short_paragraph_record_gives_cross_paragraph_ngrams():
    assert list(_extract_features(MCQ, MARK_NGRAM)) == [
        "Which planet is largest? A. Mars B. Jupiter C. Venus D. Mercury Answer:",
        "planet is largest? A. Mars B. Jupiter C. Venus D. Mercury Answer: B",
    ]


def test_gsm8k_prefix_features():
    text = "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins"
    features = list(_extract_features(text, MARK_NGRAM))
    assert len(features) == 5
    assert _bloom_hash(features[0]) == 11484006392317508080


@pytest.mark.parametrize(
    ("text", "features"),
    [
        ("Distinctive alpha phrase", ["Distinctive alpha phrase"]),
        ("Alpha beta\n\ngamma", ["Alpha beta gamma"]),
        (
            "In which fiscal quarter of 2024 did Atlassian record its second highest revenue?\n\nQ4",
            ["In which fiscal quarter of 2024 did Atlassian record its second highest revenue?"],
        ),
        (
            "one two three four\n\nfive six seven eight\n\nnine ten eleven twelve\n\nthirteen fourteen",
            [
                "one two three four five six seven eight nine ten eleven twelve thirteen",
                "two three four five six seven eight nine ten eleven twelve thirteen fourteen",
            ],
        ),
    ],
    ids=["short-exact", "short-record-fallback", "one-ngram-with-answer", "ngram-record-fallback"],
)
def test_complete_record_matches_itself_with_its_features(text, features):
    """Upstream test_decon_complete_record_retains_match_evidence."""
    score, matched = _document_overlap_and_matches(text, phrases(text), FILTER_NGRAM)
    assert score == 1.0
    assert set(matched) == {_bloom_hash(f) for f in features}


# ---- 4. rule boundaries ----------------------------------------------------------------------------


def test_overlap_threshold_is_inclusive():
    membership = phrases(E14)
    half = E14 + " q0 q1"  # 4 n-grams, 2 hits
    below = E14 + " q0 q1 q2"  # 5 n-grams, 2 hits
    assert _document_overlap_and_matches(half + "\n\n" + FILL, membership, FILTER_NGRAM)[0] == 0.5
    assert removed(half + "\n\n" + FILL, membership)
    assert not removed(below + "\n\n" + FILL, membership)


def test_single_feature_special_cases():
    membership = phrases(E13)
    assert removed(E13, membership)  # single-paragraph doc equal to one eval 13-gram
    assert removed(E13 + "\n\nshort line here", membership)  # the only usable n-gram in the doc
    assert removed(E13 + "\n\n \n\n", membership)
    assert not removed(E13 + "\n\n" + FILL, membership)  # one distinct hit among other n-gram text


def test_two_distinct_hits_in_a_paragraph_remove_a_long_document():
    assert removed(E14 + "\n\n" + FILL, phrases(E14))


def test_dropping_one_of_two_hits_keeps_the_document():
    membership = phrases(E14)
    drop = frozenset(sorted(membership)[:1])
    assert _document_overlap_and_matches(E14 + "\n\n" + FILL, membership, FILTER_NGRAM, drop) == (1.0, [])


def test_short_eval_record_only_matches_a_whole_short_document():
    membership = phrases("Lana Del Rey sings")
    assert removed("Lana Del Rey sings", membership)
    assert not removed("Lana Del Rey sings\n\n" + FILL, membership)


@pytest.mark.parametrize("short_match", ["September 29, 2011", "2, 3 and 4"])
def test_one_short_match_does_not_mark_a_long_document(short_match):
    """Upstream test_decon_does_not_mark_long_document_from_one_short_match."""
    doc = (
        "This document has unrelated material before the short value.\n\n"
        f"{short_match}\n\n"
        "It also has unrelated material after the short value."
    )
    assert _document_overlap_and_matches(doc, phrases(short_match), FILTER_NGRAM) == (1.0, [])


def test_mcq_in_short_lines_matches_alone_but_not_after_a_long_paragraph():
    membership = phrases(MCQ)
    assert removed(MCQ, membership)
    assert not removed(FILL + "\n\n" + MCQ, membership)


def test_repeated_ngrams_count_with_multiplicity():
    doc = ((E13 + " ") * 3).strip() + "\n\n" + FILL
    assert _document_overlap_and_matches(doc, phrases(E13), FILTER_NGRAM)[0] == pytest.approx(1 / 9)


def test_lone_short_line_does_not_remove_but_a_leaked_passage_does():
    """From the as-run test_decon_8m.py."""
    membership = phrases(LEAK, "2 years ago")
    assert removed(f"Homework help\n\n{LEAK}\n\nPosted 2 years ago", membership)
    assert not removed(f"{FILLER}\n\n2 years ago", membership)


def test_uses_configured_minimum_distinct_features():
    text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
    membership = phrases(text)
    three = NGramConfig(ngram_length=13, overlap_threshold=0.5, min_matched_features=3)
    assert _document_overlap_and_matches(text, membership, FILTER_NGRAM)[1]
    assert not _document_overlap_and_matches(text, membership, three)[1]


def test_only_triggering_hashes_are_reported():
    strong, weak = "alpha beta gamma delta epsilon zeta", "September 29, 2011"
    ngram = NGramConfig(min_matched_features=2, ngram_length=3, overlap_threshold=0.5)
    membership = phrases(strong, weak, ngram=ngram)
    _, matched = _document_overlap_and_matches(f"{strong}\n\n{weak}", membership, ngram)
    assert set(matched) == {_bloom_hash(g) for g in _extract_ngrams(strong, 3, 0)}


# ---- 5. drop semantics -----------------------------------------------------------------------------


def test_drop_set_stops_a_match():
    membership = phrases(LEAK)
    assert not removed(f"Homework help\n\n{LEAK}", membership, frozenset(membership))


def test_dropped_hashes_leave_numerator_and_denominator_and_never_appear_in_matches():
    membership = phrases(LEAK)
    grams = [_bloom_hash(f) for f in _extract_features(LEAK, MARK_NGRAM)]
    drop = frozenset(grams[:-2])
    score, matched = _document_overlap_and_matches(f"{FILLER}\n\n{LEAK}", membership, FILTER_NGRAM, drop)
    assert score == 1.0  # 2 remaining phrases, both hits
    assert set(matched) == set(grams[-2:])
    assert not set(matched) & drop


# ---- 8. properties ---------------------------------------------------------------------------------


def _random_docs(rng: random.Random, lines: list[str], n: int) -> list[str]:
    return ["\n\n".join(rng.sample(lines, rng.randint(1, min(4, len(lines))))) for _ in range(n)]


def test_every_filtered_doc_is_a_mark_candidate():
    rng = random.Random(0)
    full = phrases(LEAK, MCQ, "2 years ago", "Lana Del Rey")
    hero = phrases(LEAK, "2 years ago")
    lines = [LEAK, FILLER, "2 years ago", "Lana Del Rey", "A. Mars", "B. Jupiter", MCQ, " ".join(LEAK.split()[:13])]
    for membership in (full, hero):
        for doc in _random_docs(rng, lines, 160):
            for drop in (frozenset(), frozenset(rng.sample(sorted(membership), len(membership) // 3))):
                if removal_matches(doc, membership, drop):
                    assert mark_matches(doc, full)[1], doc


def test_replay_with_mark_matched_hashes_equals_the_full_eval_set():
    """S' (the union of mark matches) can replace the bloom filter on the same documents."""
    rng = random.Random(1)
    evals = [LEAK, MCQ, E14, "Lana Del Rey sings", FILLER.replace("garden", "orchard")]
    full = phrases(*evals)
    lines = [LEAK, FILLER, MCQ, E13, E14, FILL, "Lana Del Rey sings", " ".join(LEAK.split()[:14]), "a b"]
    docs = _random_docs(rng, lines, 300)
    eval_hits = {h for doc in docs for h in mark_matches(doc, full)[1]}
    for drop in (frozenset(), frozenset(rng.sample(sorted(full), len(full) // 4))):
        for doc in docs:
            assert removal_matches(doc, eval_hits, drop) == removal_matches(doc, full, drop), doc


@pytest.mark.parametrize("paragraph_break", ["\n\n", "\n"], ids=["many-paragraphs", "one-huge-paragraph"])
def test_large_text_path_equals_in_memory_path(monkeypatch, paragraph_break):
    """Records >= 1 MiB use streaming tokenization and paragraph splitting; results must not change.

    The streaming threshold is lowered so small texts take the large-record code path.
    """
    rng = random.Random(2)
    vocab = [f"w{i}" for i in range(50)] + ["12", "--", "x", "a\x1cb", "c\xa0d", "e\u200bf"]
    paragraph = " ".join(rng.choice(vocab) for _ in range(40))
    body = paragraph_break.join([paragraph, LEAK, E14, "a b c", "", paragraph, "  "])
    text = (body + paragraph_break) * 6
    membership = phrases(LEAK, E14, "a b c")
    drop = frozenset(sorted(phrases(E14))[:1])

    def run():
        return (
            list(_iter_paragraphs(text, "\n\n")),
            list(_extract_features(text, MARK_NGRAM)),
            list(_extract_ngrams(text, 13, 0)),
            _document_overlap_and_matches(text, membership, FILTER_NGRAM, drop),
            _document_overlap_and_matches(text, membership, MARK_NGRAM),
            _document_overlap_and_matches("a b c", membership, FILTER_NGRAM),
        )

    monkeypatch.setattr(matching, "LARGE_TEXT_STREAMING_THRESHOLD", 4)
    streamed = run()
    monkeypatch.setattr(matching, "LARGE_TEXT_STREAMING_THRESHOLD", 1 << 62)
    in_memory = run()
    assert streamed == in_memory
    assert streamed[4][0] > 0  # eval phrases were found in both paths


# ---- 9. degenerate inputs --------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["", " ", "\n\n", "\n\n \n\n", "a", "a b", "1 2 3"])
def test_empty_and_tiny_inputs_are_kept(text):
    membership = phrases(E14, "a b c")
    assert _document_overlap_and_matches(text, membership, FILTER_NGRAM) == (0.0, [])
