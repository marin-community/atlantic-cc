# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Length-sorted, bucketed, fixed-size batches for the neural classifiers (as-run ``score_survivors``).

The scores of the bf16 models depend on the padded shape of the batch a document is scored in, so
this batching is part of the reproducibility contract, not a tuning knob. Per call (one WARC's
documents for one model):

* documents are ordered by token count with a stable sort and cut into batches of 32;
* each batch is padded to the smallest length bucket (128, 256, ..., 8192) that holds its longest
  document, capped at the model's maximum length;
* a short final batch is filled with empty rows up to 32, so every batch has a static shape;
* ids are padded with ``[PAD]`` (50283) and the attention mask is True exactly at real tokens.
"""

import numpy as np

from atlantic_cc.models import Scorer
from atlantic_cc.spec import LENGTH_BUCKETS, PAD_TOKEN_ID, SCORE_BATCH_SIZE

_EMPTY_ROW = np.zeros(0, dtype=np.int32)


def bucket_for(n_tokens: int, max_len: int) -> int:
    """The smallest length bucket >= ``n_tokens`` that is <= ``max_len``; ``max_len`` if none fits."""
    for bucket in LENGTH_BUCKETS:
        if n_tokens <= bucket <= max_len:
            return bucket
    return max_len


def pad_batch(rows: list[np.ndarray], length: int) -> tuple[np.ndarray, np.ndarray]:
    """Pad id rows to int32 ``(len(rows), length)`` with ``[PAD]``; the mask is True at real tokens.

    A row longer than ``length`` is cut to ``length`` (as-run behavior; the callers never pass one).
    """
    input_ids = np.full((len(rows), length), PAD_TOKEN_ID, dtype=np.int32)
    attention_mask = np.zeros((len(rows), length), dtype=bool)
    for i, ids in enumerate(rows):
        n = min(len(ids), length)
        input_ids[i, :n] = ids[:n]
        attention_mask[i, :n] = True
    return input_ids, attention_mask


def score_ids(scorer: Scorer, ids: list[np.ndarray], max_len: int) -> np.ndarray:
    """P(useful) per document, float32 in input order, batched exactly as production batched it.

    Args:
        scorer: The model; called once per batch of ``SCORE_BATCH_SIZE`` rows.
        ids: One int32 token-id array per document, each at most ``max_len`` long.
        max_len: The model's maximum sequence length (8192 pooled, 2048 terminal); caps the buckets.
    """
    probs = np.zeros(len(ids), dtype=np.float32)
    if not ids:
        return probs
    order = np.argsort([len(x) for x in ids], kind="stable")
    for start in range(0, len(ids), SCORE_BATCH_SIZE):
        idx = order[start : start + SCORE_BATCH_SIZE]
        rows = [ids[i] for i in idx]
        length = bucket_for(max(len(r) for r in rows), max_len)
        input_ids, attention_mask = pad_batch(rows + [_EMPTY_ROW] * (SCORE_BATCH_SIZE - len(rows)), length)
        batch_probs = np.asarray(scorer.score(input_ids, attention_mask))
        if batch_probs.dtype != np.float32 or batch_probs.shape != (SCORE_BATCH_SIZE,):
            raise ValueError(
                f"scorer returned {batch_probs.dtype}{batch_probs.shape}, expected float32({SCORE_BATCH_SIZE},)"
            )
        probs[idx] = batch_probs[: len(rows)]
    return probs
