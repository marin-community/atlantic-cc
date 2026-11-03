# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""U-TOK: gigatoken == HF tokenizers, the 65,536-code-point cap, [CLS]+8190+[SEP], truncate_ids.

Needs ``answerdotai/ModernBERT-base`` at the pinned revision in the local Hugging Face cache
(``load_tokenizer()`` with network access fills it once); skipped otherwise.
"""

import numpy as np
import pytest

from atlantic_cc import spec
from atlantic_cc.classify.text import normalize_text
from atlantic_cc.classify.tokenize import (
    ModernBertTokenizer,
    assert_parity,
    load_tokenizer,
    tokenize_batch,
    tokenize_batch_reference,
    truncate_ids,
)

# Adversarial cases beyond the startup gate (research probe ``gt_parity_probe.py``): added-token
# literals, NFC, ZWJ emoji, BOM/ZWSP, control characters, giant unspaced runs, RTL marks, math
# alphanumerics, Turkish I, combining stacks, Unicode 15 additions, mixed digits, NBSP variants.
PROBE_CASES = {
    "endoftext_literal": "before <|endoftext|> after",
    "padding_literal": "x <|padding|> y",
    "unused0_literal": "see [unused0] and [unused82] here",
    "ip_upper_lowered": "|||IP_ADDRESS||| becomes lower",
    "cls_upper_lowered": "[CLS] [SEP] [MASK]",
    "nfc_decomposed_e": "cafe\u0301 re\u0301sume\u0301",
    "nfc_angstrom_sign": "\u212b and \u2126 ohm",
    "nfc_cjk_compat": "\uf900\uf901\uf902 \ufa00",
    "hangul_jamo": "\u1100\u1161\u11a8 \u1112\u1161\u11ab",
    "zwj_emoji": "family \U0001f468\u200d\U0001f469\u200d\U0001f467 flag \U0001f1fa\U0001f1f8",
    "bom_mid": "a\ufeffb \u200bzero width",
    "nul_and_ctrl": "a\x00b\x01c\x7f d",
    "long_no_space": "x" * 70000,
    "long_url": "https://example.com/" + "a1b2c3/" * 12000,
    "rtl_marks": "\u202eabc\u202c \u200f\u05e9\u05dc\u05d5\u05dd",
    "math_alnum": "\U0001d400\U0001d401 \U0001d7ce",
    "turkish_i_upper": "\u0130stanbul I",
    "combining_stack": "a\u0300\u0301\u0302\u0303\u0304 o\u0308\u0304",
    "new_unicode_15": "\U0001fae8 \U0001fa77 \U0001e030",
    "digits_mixed": "\u0661\u0662\u0663 \u0967\u0968 123,456.78",
    "tab_newline_cr": "a\tb\nc\rd\x0be\x0cf",
    "nbsp_variants": "a\xa0b\u2007c\u202fd\u3000e",
    "surrogate_free_big": ("word " * 9000) + "<|endoftext|>",
}


@pytest.fixture(scope="module")
def tok() -> ModernBertTokenizer:
    try:
        return load_tokenizer(local_files_only=True)
    except OSError as e:
        pytest.skip(f"ModernBERT-base tokenizer not in the local Hugging Face cache: {e}")


def _ids(tok, text: str, max_length: int = spec.POOLED_MAX_TOKENS) -> list[int]:
    return tokenize_batch(tok, [text], max_length=max_length)[0].tolist()


@pytest.mark.parametrize("max_length", [spec.POOLED_MAX_TOKENS, 128])
def test_gigatoken_matches_hf_on_adversarial_cases(tok, max_length):
    texts = [normalize_text(t) for t in PROBE_CASES.values()]
    fast = tokenize_batch(tok, texts, max_length=max_length)
    reference = tokenize_batch_reference(tok, texts, max_length=max_length)
    mismatched = [name for name, a, b in zip(PROBE_CASES, fast, reference, strict=True) if not np.array_equal(a, b)]
    assert mismatched == []
    assert all(ids.dtype == np.int32 for ids in fast)


@pytest.mark.parametrize(
    ("text", "ids"),
    [
        ("before <|endoftext|> after", [50281, 9131, 209, 50279, 846, 50282]),
        ("x <|padding|> y", [50281, 89, 209, 1, 340, 50282]),
        ("see [unused0] and [unused82] here", [50281, 2887, 209, 50285, 285, 209, 50367, 1060, 50282]),
        ("hello world", [50281, 25521, 1533, 50282]),
        ("", [spec.CLS_TOKEN_ID, spec.SEP_TOKEN_ID]),
    ],
)
def test_exact_ids_including_added_token_literals(tok, text, ids):
    assert _ids(tok, text) == ids


def test_long_documents_get_8190_body_tokens(tok):
    ids = _ids(tok, normalize_text("word " * 20000))
    assert len(ids) == spec.POOLED_MAX_TOKENS
    assert (ids[0], ids[-1]) == (spec.CLS_TOKEN_ID, spec.SEP_TOKEN_ID)


def test_only_the_first_65536_code_points_are_tokenized(tok):
    """Above 8 characters per token the cap, not the token budget, ends the document."""
    text = normalize_text(" understanding" * 10000)  # 139,999 code points, ~14 per token
    ids = _ids(tok, text)
    assert len(ids) == 4685
    assert ids == _ids(tok, text[: spec.TOKENIZE_CHAR_CAP])
    assert len(tok.fast.encode(text)) > spec.POOLED_MAX_TOKENS - 2  # uncapped, it would fill the budget


@pytest.mark.parametrize("n_words", [0, 10, 1000, 2046, 2047, 2048, 5000])
def test_truncate_ids_equals_tokenizing_at_2048(tok, n_words):
    text = " ".join(f"tok{i}word{i * 7 % 13}" for i in range(n_words))
    long_ids = tokenize_batch(tok, [text])[0]
    derived = truncate_ids(long_ids, spec.TERMINAL_MAX_TOKENS)
    assert np.array_equal(derived, tokenize_batch_reference(tok, [text], max_length=spec.TERMINAL_MAX_TOKENS)[0])
    assert np.array_equal(derived, tokenize_batch(tok, [text], max_length=spec.TERMINAL_MAX_TOKENS)[0])


def test_truncate_ids_boundaries():
    ids = np.arange(10, dtype=np.int32)
    assert truncate_ids(ids, 10) is ids
    assert truncate_ids(ids, 5).tolist() == [0, 1, 2, 3, spec.SEP_TOKEN_ID]
    assert truncate_ids(ids, 5).dtype == np.int32


def test_parity_gate_refuses_a_divergent_tokenizer(tok):
    class _OffByOne:
        def encode_batch(self, texts):
            return tok.fast.encode_batch(texts) + 1

    assert_parity(tok)
    with pytest.raises(RuntimeError, match="diverges from HF on parity text"):
        assert_parity(ModernBertTokenizer(hf=tok.hf, fast=_OffByOne()))


def test_empty_batch(tok):
    assert tokenize_batch(tok, []) == []
    assert tokenize_batch_reference(tok, []) == []
