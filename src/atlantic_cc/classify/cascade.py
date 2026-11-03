# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The pooled -> terminal decision for fastText survivors (as-run ``tpu_phase.process_warc_v3``).

On the pooled probability ``p`` of a document:

* ``p >= POOLED_ACCEPT_AT``: kept without the terminal model (its ``modernbert_prob`` stays NaN);
* ``p < POOLED_DROP_BELOW``: dropped;
* otherwise (the band): the terminal model scores it and it is kept iff that score is
  ``>= TERMINAL_KEEP_AT``.

Every comparison is between float32 values, as production's NumPy 2 (NEP 50) comparisons of
float32 arrays against Python floats were, so the effective thresholds are the float32 roundings of
the decimal constants: a terminal score of exactly ``float32(0.4378) = 0.43779999`` is kept. NaN
compares False everywhere, so a NaN pooled score is dropped and a NaN terminal score is rejected.
"""

import math

import numpy as np

from atlantic_cc.schemas import TOMBSTONE_PROB_DECIMALS
from atlantic_cc.spec import POOLED_ACCEPT_AT, POOLED_DROP_BELOW, TERMINAL_KEEP_AT


def _require_float32(probs: np.ndarray) -> None:
    if probs.dtype != np.float32:
        raise TypeError(f"cascade probabilities must be float32 (production's dtype), got {probs.dtype}")


def accept_mask(pooled_probs: np.ndarray) -> np.ndarray:
    """Documents the pooled model keeps outright: ``p >= POOLED_ACCEPT_AT``."""
    _require_float32(pooled_probs)
    return pooled_probs >= POOLED_ACCEPT_AT


def band_mask(pooled_probs: np.ndarray) -> np.ndarray:
    """Documents the terminal model scores: ``POOLED_DROP_BELOW <= p < POOLED_ACCEPT_AT``."""
    _require_float32(pooled_probs)
    return (POOLED_DROP_BELOW <= pooled_probs) & (pooled_probs < POOLED_ACCEPT_AT)


def keep_mask(pooled_probs: np.ndarray, terminal_probs: np.ndarray) -> np.ndarray:
    """``accept | (terminal >= TERMINAL_KEEP_AT)``; ``terminal_probs`` is NaN outside the band."""
    _require_float32(terminal_probs)
    return accept_mask(pooled_probs) | (terminal_probs >= TERMINAL_KEEP_AT)


def tombstone_rows(
    doc_ids: list[str], pooled_probs: np.ndarray, terminal_probs: np.ndarray, keep: np.ndarray
) -> list[dict]:
    """One row per dropped document, in input order, as written to ``tombstones/``.

    ``{"doc_id", "pooled_prob": round(p, 6), "modernbert_prob": round(t, 6) or None}``; None marks a
    document the terminal model never scored (or scored NaN). Kept documents get no row.
    """
    rows = []
    for i in np.flatnonzero(~keep):
        terminal = float(terminal_probs[i])
        rows.append(
            {
                "doc_id": doc_ids[i],
                "pooled_prob": round(float(pooled_probs[i]), TOMBSTONE_PROB_DECIMALS),
                "modernbert_prob": None if math.isnan(terminal) else round(terminal, TOMBSTONE_PROB_DECIMALS),
            }
        )
    return rows
