# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decontamination matching, rule version 4 (the rule that produced ``decon_matches``).

Verbatim port of ``experiments/fast_curation/decon_v4.py`` at Marin tag ``lpv11-8m-as-run``
(itself copied from upstream ``lib/marin/src/marin/datakit/decon.py`` at ``cb3b7d0b02``). Only the
imports changed and the node-local bloom loader moved to :mod:`atlantic_cc.decon.bloom`. Keep every
function byte-for-byte equivalent: tokenization (``str.split``), the alphabetic guard, the blake2b-64
hash, paragraph splitting and the document rule decide which documents count as contaminated.
"""

import re
from collections import deque
from collections.abc import Container, Iterator
from dataclasses import dataclass
from hashlib import blake2b as _blake2b
from itertools import chain, islice
from typing import NamedTuple

FEATURE_FILTER_VERSION = 4
MIN_SHORT_EXACT_TOKENS = 3
DEFAULT_PARAGRAPH_DELIMITER = "\n\n"
LARGE_TEXT_STREAMING_THRESHOLD = 1024 * 1024
_TOKEN_PATTERN = re.compile(r"\S+")


@dataclass(frozen=True, kw_only=True)
class NGramConfig:
    """Word-ngram matching parameters.

    Attributes:
        ngram_length: Size of each ngram in whitespace-split word tokens.
        stride: Step between successive ngrams. 0 = contiguous (every position).
        overlap_threshold: Minimum fraction of paragraph ngrams that must hit
            the filter for the paragraph to count as contaminated.
        min_matched_features: Minimum number of distinct matched features that
            a paragraph must contain. A complete document with exactly one
            feature can still match.
        paragraph_delimiter: String the text is split on to form paragraphs (the
            unit the overlap fraction is computed over). Defaults to ``"\\n\\n"``, a
            blank-line-delimited block, so ngrams span single line breaks.
    """

    ngram_length: int = 13
    stride: int = 0
    overlap_threshold: float = 0.5
    min_matched_features: int
    paragraph_delimiter: str = DEFAULT_PARAGRAPH_DELIMITER

    def __post_init__(self) -> None:
        if self.min_matched_features < 1:
            raise ValueError("min_matched_features must be at least 1")


def _bloom_hash(x: str) -> int:
    return int.from_bytes(_blake2b(x.encode(), digest_size=8).digest(), "big")


def _has_alpha(ngram: str) -> bool:
    """True if *ngram* contains any alphabetic character.

    A 13-gram with no letters (numeric sequences, punctuation runs, form-field boilerplate) carries
    no contamination signal but collides with number-list eval items. Skipping these on both the
    eval side and the document side keeps the overlap denominator consistent.
    """
    return any(map(str.isalpha, ngram))


def _extract_token_ngrams(tokens: list[str], n: int, stride: int) -> Iterator[str]:
    token_has_alpha = bytearray(_has_alpha(token) for token in tokens)
    for i in range(0, len(tokens) - n + 1, stride + 1):
        if any(token_has_alpha[i : i + n]):
            yield " ".join(tokens[i : i + n])


def _extract_streaming_ngrams(text: str, n: int, stride: int) -> Iterator[str]:
    """Yield n-grams without retaining every token from a large record."""
    window: deque[tuple[int, int, bool]] = deque()
    alpha_tokens = 0
    step = stride + 1
    for token_index, match in enumerate(_TOKEN_PATTERN.finditer(text)):
        if len(window) == n:
            _, _, had_alpha = window.popleft()
            alpha_tokens -= had_alpha

        start, end = match.span()
        has_alpha = any(map(str.isalpha, text[start:end]))
        window.append((start, end, has_alpha))
        alpha_tokens += has_alpha

        ngram_start = token_index - n + 1
        if len(window) == n and ngram_start % step == 0 and alpha_tokens:
            yield " ".join(text[start:end] for start, end, _ in window)


def _extract_ngrams(text: str, n: int, stride: int) -> Iterator[str]:
    if len(text) < LARGE_TEXT_STREAMING_THRESHOLD:
        yield from _extract_token_ngrams(text.split(), n, stride)
        return
    yield from _extract_streaming_ngrams(text, n, stride)


def _short_exact_feature_from_tokens(tokens: list[str], n: int) -> str | None:
    if MIN_SHORT_EXACT_TOKENS <= len(tokens) < n and any(_has_alpha(token) for token in tokens):
        return " ".join(tokens)
    return None


def _short_exact_feature(text: str, n: int) -> str | None:
    """Return one normalized feature for guarded short exact matching."""
    if len(text) < LARGE_TEXT_STREAMING_THRESHOLD:
        return _short_exact_feature_from_tokens(text.split(), n)

    matches = list(islice(_TOKEN_PATTERN.finditer(text), n))
    if not MIN_SHORT_EXACT_TOKENS <= len(matches) < n:
        return None
    spans = [match.span() for match in matches]
    if not any(any(map(str.isalpha, text[start:end])) for start, end in spans):
        return None
    return " ".join(text[start:end] for start, end in spans)


def _iter_nonempty_segments(text: str, delimiter: str) -> Iterator[str]:
    """Yield split segments one at a time to bound memory for large records."""
    if not delimiter:
        raise ValueError("empty separator")
    start = 0
    while True:
        end = text.find(delimiter, start)
        if end < 0:
            if start < len(text):
                yield text[start:]
            return
        if end > start:
            yield text[start:end]
        start = end + len(delimiter)


def _iter_paragraphs(text: str, delimiter: str) -> Iterator[str]:
    if len(text) < LARGE_TEXT_STREAMING_THRESHOLD:
        yield from (paragraph for paragraph in text.split(delimiter) if paragraph)
        return
    yield from _iter_nonempty_segments(text, delimiter)


def _extract_features(text: str, ngram: NGramConfig | None) -> Iterator[str]:
    """Yield matchable features from a complete record.

    N-gram mode emits n-grams from each paragraph. It does not emit exact
    features for short paragraphs in a longer record. With the default
    blank-line policy, the complete record emits n-grams when all paragraphs
    are short. A complete short record emits one guarded exact feature.
    """
    delimiter = ngram.paragraph_delimiter if ngram is not None else "\n"
    paragraphs = _iter_paragraphs(text, delimiter)
    if ngram is None:
        for para in paragraphs:
            yield para
        return

    short_exact = _short_exact_feature(text, ngram.ngram_length)
    if short_exact is not None:
        yield short_exact
        return

    yielded = False
    for para in paragraphs:
        for feature in _extract_ngrams(para, ngram.ngram_length, ngram.stride):
            yielded = True
            yield feature
    if not yielded and ngram.paragraph_delimiter == DEFAULT_PARAGRAPH_DELIMITER:
        yield from _extract_ngrams(text, ngram.ngram_length, ngram.stride)


class _ParagraphOverlap(NamedTuple):
    score: float
    matched_hashes: list[int]
    feature_count: int
    has_ngrams: bool


def _paragraph_overlap_matches_and_presence(
    paragraph: str,
    bf: Container[int],
    ngram: NGramConfig | None,
    drop_hashes: frozenset[int] = frozenset(),
) -> _ParagraphOverlap:
    """Return overlap after drop filtering and n-gram presence before filtering."""
    if ngram is None:
        h = _bloom_hash(paragraph)
        if h in drop_hashes:
            return _ParagraphOverlap(0.0, [], 0, False)
        return _ParagraphOverlap(1.0, [h], 1, False) if h in bf else _ParagraphOverlap(0.0, [], 1, False)

    has_ngram_features = False
    feature_count = 0
    matched: list[int] = []
    features: Iterator[str]
    if len(paragraph) < LARGE_TEXT_STREAMING_THRESHOLD:
        tokens = paragraph.split()
        short_exact = _short_exact_feature_from_tokens(tokens, ngram.ngram_length)
        features = _extract_token_ngrams(tokens, ngram.ngram_length, ngram.stride)
    else:
        short_exact = _short_exact_feature(paragraph, ngram.ngram_length)
        features = _extract_ngrams(paragraph, ngram.ngram_length, ngram.stride)
    if short_exact is not None:
        features = iter((short_exact,))
    for feature in features:
        has_ngram_features = short_exact is None
        hash_value = _bloom_hash(feature)
        if hash_value in drop_hashes:
            continue
        feature_count += 1
        if hash_value in bf:
            matched.append(hash_value)
    if feature_count == 0:
        return _ParagraphOverlap(0.0, [], feature_count, has_ngram_features)
    return _ParagraphOverlap(len(matched) / feature_count, matched, feature_count, has_ngram_features)


def _document_overlap_and_matches(
    text: str,
    bf: Container[int],
    ngram: NGramConfig | None,
    drop_hashes: frozenset[int] = frozenset(),
) -> tuple[float, list[int]]:
    """Return the highest overlap and the hashes that cause a document mark."""
    minimum = ngram.min_matched_features if ngram is not None else 1
    threshold = ngram.overlap_threshold if ngram is not None else 0.0
    delimiter = ngram.paragraph_delimiter if ngram is not None else "\n"
    paragraph_iter = iter(_iter_paragraphs(text, delimiter))
    first_paragraph = next(paragraph_iter, None)
    second_paragraph = next(paragraph_iter, None)
    single_paragraph = first_paragraph is not None and second_paragraph is None
    if first_paragraph is None:
        paragraphs: Iterator[str] = iter(())
    elif second_paragraph is None:
        paragraphs = iter((first_paragraph,))
    else:
        paragraphs = chain((first_paragraph, second_paragraph), paragraph_iter)
    max_score = 0.0
    matched: set[int] = set()
    has_paragraph_ngrams = False
    document_ngram_feature_count = 0
    document_ngram_hits: set[int] = set()

    for paragraph in paragraphs:
        score, hits, feature_count, paragraph_has_ngrams = _paragraph_overlap_matches_and_presence(
            paragraph, bf, ngram, drop_hashes
        )
        if ngram is not None and paragraph_has_ngrams:
            has_paragraph_ngrams = True
            document_ngram_feature_count += feature_count
            document_ngram_hits.update(hits)
        max_score = max(max_score, score)
        if not hits:
            continue
        if ngram is None:
            matched.update(hits)
            continue

        complete_single_feature_document = single_paragraph and feature_count == 1 and score == 1.0
        distinct_hits = len(set(hits))
        if score >= threshold and (distinct_hits >= minimum or complete_single_feature_document):
            matched.update(hits)

    # Retain a match when drop filtering leaves one usable n-gram in the document.
    if ngram is not None and document_ngram_feature_count == 1 and document_ngram_hits:
        matched.update(document_ngram_hits)

    if ngram is not None and not has_paragraph_ngrams:
        use_record_fallback = (
            ngram.paragraph_delimiter == DEFAULT_PARAGRAPH_DELIMITER
            or _short_exact_feature(text, ngram.ngram_length) is not None
        )
        if use_record_fallback:
            score, hits, feature_count, _has_ngrams = _paragraph_overlap_matches_and_presence(
                text, bf, ngram, drop_hashes
            )
            max_score = max(max_score, score)
            distinct_hits = len(set(hits))
            complete_single_feature_document = feature_count == 1 and score == 1.0
            if score >= threshold and (distinct_hits >= minimum or complete_single_feature_document):
                matched.update(hits)

    return max_score, sorted(matched)
