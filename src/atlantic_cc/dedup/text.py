# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-document normalization, content ids and document keys (as-run ``rdedup/docs.py``).

A kept document is dropped if its text is missing or blank after ``str.strip``; every whitespace run longer than
128 characters is cut to its first 128 characters; ``xxh3_128`` of the compacted UTF-8 text is its content id
(the exact-dedup key, printed as 32 hex digits). Python's ``str`` methods and ``re`` decide what counts as
whitespace, so CPython 3.12 (Unicode 15.0) is part of the contract.
"""

import re

import dupekit_native as dupekit
import numpy as np

from atlantic_cc.dedup.params import CC_SOURCE_TAG, MAX_ROWS_PER_FILE, MAX_WHITESPACE_RUN_CHARS, ROW_BITS

_LONG_WHITESPACE = re.compile(r"\s{" + str(MAX_WHITESPACE_RUN_CHARS + 1) + r",}")


def compact_whitespace(text: str) -> str:
    """Cut whitespace runs longer than 128 characters to their first 128 characters."""
    if _LONG_WHITESPACE.search(text) is None:
        return text
    return _LONG_WHITESPACE.sub(lambda m: m.group(0)[:MAX_WHITESPACE_RUN_CHARS], text)


def is_kept_text(text: str | None) -> bool:
    """Falsy text is skipped, then text that is blank after ``str.strip``."""
    return bool(text) and bool(text.strip())


def content_id(text: str) -> int:
    """The 128-bit content id: ``xxh3_128`` of the UTF-8 text."""
    return dupekit.hash_xxh3_128(text.encode("utf-8"))


def id_hex(hi: int, lo: int) -> str:
    return format((int(hi) << 64) | int(lo), "032x")


def doc_keys(file_index: int, rows: np.ndarray) -> np.ndarray:
    """Globally unique, file-ordered document keys: ``file_index << 24 | row``."""
    return (np.uint64(file_index) << np.uint64(ROW_BITS)) | rows.astype(np.uint64)


def key_file(keys: np.ndarray) -> np.ndarray:
    return keys >> np.uint64(ROW_BITS)


def key_row(keys: np.ndarray) -> np.ndarray:
    return keys & np.uint64(MAX_ROWS_PER_FILE - 1)


def normalize_file(texts: list[str | None]) -> tuple[np.ndarray, list[str], int]:
    """Kept row indices, their compacted texts, and how many texts compaction changed."""
    rows = []
    compacted_texts = []
    compacted = 0
    for i, text in enumerate(texts):
        if not is_kept_text(text):
            continue
        normalized = compact_whitespace(text)
        if len(normalized) != len(text):
            compacted += 1
        rows.append(i)
        compacted_texts.append(normalized)
    return np.asarray(rows, dtype=np.int64), compacted_texts, compacted


# The canonical member of a component (``attrs.canon``) minimizes the decimal string of
# ``xxh3_128("source_000|" + id_hex)`` in lexicographic order. Padding a string on the right with a character below
# '0' and reading it in base 11 preserves that order; 39 digits split into three 13-digit words keep each word below
# 11**13 < 2**64.
_ORDER_DIGITS = 39
_ORDER_WORD_DIGITS = 13
_ORDER_POWERS = np.array([11**p for p in range(_ORDER_WORD_DIGITS - 1, -1, -1)], dtype=np.uint64)


def order_words(digit_strings: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three uint64 words per decimal string whose tuple order is the strings' lexicographic order."""
    n = len(digit_strings)
    if n == 0:
        empty = np.zeros(0, dtype=np.uint64)
        return empty, empty, empty
    # '/' is the byte just below '0': it maps to digit 0 and the digits '0'-'9' map to 1-10.
    padded = "".join(s.ljust(_ORDER_DIGITS, "/") for s in digit_strings).encode()
    digits = (np.frombuffer(padded, dtype=np.uint8).reshape(n, _ORDER_DIGITS) - ord("/")).astype(np.uint64)
    w = _ORDER_WORD_DIGITS
    return (
        digits[:, 0:w] @ _ORDER_POWERS,
        digits[:, w : 2 * w] @ _ORDER_POWERS,
        digits[:, 2 * w : 3 * w] @ _ORDER_POWERS,
    )


def id_norm(hi: int, lo: int) -> str:
    """The ordering string of this document's single-source node in Marin's ``connected_components``."""
    return str(dupekit.hash_xxh3_128(f"{CC_SOURCE_TAG}|{id_hex(hi, lo)}".encode()))


def canonical_order_keys(hi: np.ndarray, lo: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Order words for many docs: the canonical of a component has the smallest."""
    return order_words([id_norm(high, low) for high, low in zip(hi.tolist(), lo.tolist(), strict=True)])
