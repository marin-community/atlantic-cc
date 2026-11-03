# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ModernBERT-base tokenization of the normalized text (as-run ``tokenize_trunc_batch_gigatoken``).

Each document becomes ``[CLS] + ids[:8190] + [SEP]`` as int32, where ``ids`` tokenizes only the
first 65,536 code points (8 per token of the 8,192-token budget). Production used gigatoken 0.10.0
over the Hugging Face tokenizer and refused to start unless gigatoken reproduced the HF
``tokenizers`` reference exactly on a fixed set of adversarial texts; both paths and that gate are
kept here. Added-token literals that survive lowercasing (``<|endoftext|>``, ``<|padding|>``,
``[unused0]``..``[unused82]``) are tokenized as their added-token ids by both paths.

Production loaded the tokenizer with ``transformers.AutoTokenizer``. This module reads the pinned
``tokenizer.json`` with ``tokenizers`` instead: it serializes to the same JSON as the transformers
5.12.1 backend did (1,483,360 characters, identical), so gigatoken is built from the same JSON and
both paths are unchanged. The native libraries load on first use, which keeps this module cheap to
import (phase B and every worker process import it).

The terminal model reads the first 2,048 of these ids: :func:`truncate_ids` derives them, and they
equal a fresh tokenization at ``max_length=2048``.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from atlantic_cc import spec

logger = logging.getLogger(__name__)

TOKENIZER_FILE = "tokenizer.json"
# Only the first max_length * 8 code points are tokenized. Below 8 characters per token the first
# max_length - 2 tokens lie inside this prefix and the cap changes nothing; above that the cap ends
# the document before the token budget does. Production applied it, so it is part of the contract.
CHARS_PER_TOKEN_CAP = 8
# Sub-batch of the HF reference path (bounds the char-capped copies alive at once).
_REFERENCE_BATCH = 1024

# as-run: experiments/fast_curation/preprocess.py:291-299 (the worker-startup parity texts).
PARITY_TEXTS = (
    "",
    " ",
    "short doc",
    "word " * 20000,
    " ".join(f"tok{i}word{i * 7 % 13}" for i in range(60000)),
    "some utf-8: caf\xe9 π 漢字 emoji \U0001f680 mixed in. " * 500,
    "a" * (8192 * 8 + 50),
)


@dataclass(frozen=True)
class ModernBertTokenizer:
    hf: Any  # reference: ``tokenizers.Tokenizer`` loaded from the pinned tokenizer.json
    fast: Any  # production: ``gigatoken.Tokenizer`` built from ``hf``


def load_tokenizer(*, local_files_only: bool = False) -> ModernBertTokenizer:
    """Load ``answerdotai/ModernBERT-base``'s tokenizer at the pinned revision and run the startup parity gate."""
    import gigatoken  # noqa: PLC0415 - native; loaded on use to keep this module cheap to import
    import tokenizers  # noqa: PLC0415 - idem
    from huggingface_hub import hf_hub_download  # noqa: PLC0415 - pulls in httpx; idem

    path = hf_hub_download(
        spec.TOKENIZER_REPO, TOKENIZER_FILE, revision=spec.TOKENIZER_REVISION, local_files_only=local_files_only
    )
    hf = tokenizers.Tokenizer.from_file(path)
    # gigatoken serializes ``hf`` (``to_str()``) here, before the reference path sets truncation.
    tok = ModernBertTokenizer(hf=hf, fast=gigatoken.Tokenizer(hf))
    assert_parity(tok)
    return tok


def tokenize_batch(
    tok: ModernBertTokenizer, normalized_texts: list[str], *, max_length: int = spec.POOLED_MAX_TOKENS
) -> list[np.ndarray]:
    """gigatoken: int32 ``[CLS] + ids[: max_length - 2] + [SEP]`` per text, ids of ``t[: 8 * max_length]``."""
    import awkward as ak  # noqa: PLC0415 - native; see load_tokenizer

    if not normalized_texts:
        return []
    capped = [t[: max_length * CHARS_PER_TOKEN_CAP] for t in normalized_texts]
    body = ak.values_astype(tok.fast.encode_batch(capped)[:, : max_length - 2], np.int32)
    n = len(body)
    one = np.ones(n, np.int64)
    cls_col = ak.unflatten(np.full(n, spec.CLS_TOKEN_ID, np.int32), one)
    sep_col = ak.unflatten(np.full(n, spec.SEP_TOKEN_ID, np.int32), one)
    full = ak.concatenate([cls_col, body, sep_col], axis=1)
    flat = np.asarray(ak.flatten(full), dtype=np.int32)
    lengths = np.asarray(ak.num(full), dtype=np.int64)
    return np.split(flat, np.cumsum(lengths)[:-1])


def tokenize_batch_reference(
    tok: ModernBertTokenizer, normalized_texts: list[str], *, max_length: int = spec.POOLED_MAX_TOKENS
) -> list[np.ndarray]:
    """HF ``tokenizers``: right truncation to ``max_length`` with the default special tokens.

    Same output as :func:`tokenize_batch`. Sets the tokenizer's truncation, so calls at different
    ``max_length`` on one tokenizer must not interleave across threads.
    """
    if not normalized_texts:
        return []
    tok.hf.enable_truncation(max_length, stride=0, strategy="longest_first", direction="right")
    out: list[np.ndarray] = []
    for i in range(0, len(normalized_texts), _REFERENCE_BATCH):
        chunk = [t[: max_length * CHARS_PER_TOKEN_CAP] for t in normalized_texts[i : i + _REFERENCE_BATCH]]
        out.extend(np.asarray(enc.ids, dtype=np.int32) for enc in tok.hf.encode_batch_fast(chunk))
    return out


def assert_parity(tok: ModernBertTokenizer, *, max_length: int = spec.POOLED_MAX_TOKENS) -> None:
    """Raise unless gigatoken reproduces the HF reference exactly on :data:`PARITY_TEXTS`."""
    texts = list(PARITY_TEXTS)
    reference = tokenize_batch_reference(tok, texts, max_length=max_length)
    candidate = tokenize_batch(tok, texts, max_length=max_length)
    for i, (a, b) in enumerate(zip(reference, candidate, strict=True)):
        if not np.array_equal(a, b):
            raise RuntimeError(
                f"gigatoken tokenization diverges from HF on parity text {i} (len {len(a)} vs {len(b)}); "
                "refusing to tokenize the corpus with it."
            )
    logger.info("gigatoken parity check passed (%d texts, max_length=%d)", len(texts), max_length)


def truncate_ids(ids: np.ndarray, max_len: int) -> np.ndarray:
    """Ids of a longer tokenization -> the ids a tokenization at ``max_len`` yields: ``ids[:max_len-1] + [SEP]``."""
    if len(ids) <= max_len:
        return ids
    return np.concatenate([ids[: max_len - 1], np.array([spec.SEP_TOKEN_ID], dtype=ids.dtype)])
