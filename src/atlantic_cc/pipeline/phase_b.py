# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase B: the neural cascade over one WARC's fastText survivors (as-run ``tpu_phase.process_warc_v3``).

1. Tokenize ``normalize_text(text)`` of every survivor: ``[CLS] + <= 8,190 ids + [SEP]``.
2. Score all of them with the pooled model on those ids.
3. Score the band (``POOLED_DROP_BELOW <= p < POOLED_ACCEPT_AT``) with the terminal model on the
   first 2,048 ids (``truncate_ids``).
4. Keep pooled accepts and band documents the terminal model accepts. The kept table is the
   survivors in WARC order plus ``n_tokens``, ``pooled_prob`` and ``modernbert_prob`` (NaN for
   pooled accepts); every other survivor becomes a tombstone row.
"""

import time
from collections.abc import Callable

import numpy as np
import pyarrow as pa

from atlantic_cc.classify.batching import score_ids
from atlantic_cc.classify.cascade import accept_mask, band_mask, keep_mask, tombstone_rows
from atlantic_cc.classify.text import normalize_text
from atlantic_cc.classify.tokenize import truncate_ids
from atlantic_cc.models import Scorer
from atlantic_cc.pipeline.outputs import PhaseBCounters
from atlantic_cc.schemas import KEPT_SCHEMA, PRESURVIVOR_SCHEMA
from atlantic_cc.spec import POOLED_MAX_TOKENS, TERMINAL_MAX_TOKENS

# Normalized texts -> one int32 id array per text (``functools.partial(tokenize.tokenize_batch, tok)``).
Tokenize = Callable[[list[str]], list[np.ndarray]]


def _require_max_tokens(scorer: Scorer, expected: int, role: str) -> None:
    if scorer.max_tokens != expected:
        raise ValueError(f"the {role} scorer reads {scorer.max_tokens} tokens; the cascade feeds it {expected}")


def run_phase_b(
    presurvivors: pa.Table, tokenize: Tokenize, pooled: Scorer, terminal: Scorer
) -> tuple[pa.Table, list[dict], PhaseBCounters]:
    """Classify one WARC's fastText survivors.

    Args:
        presurvivors: ``schemas.PRESURVIVOR_SCHEMA`` rows in WARC record order.
        tokenize: The ModernBERT tokenization at 8,192 tokens.
        pooled: The pooled band model (8,192 tokens).
        terminal: The Ettin terminal model (2,048 tokens).

    Returns:
        The kept table (``schemas.KEPT_SCHEMA``, WARC order), the tombstone rows (WARC order) and the counters.
    """
    if not presurvivors.schema.equals(PRESURVIVOR_SCHEMA):
        raise ValueError(f"presurvivor schema differs from schemas.PRESURVIVOR_SCHEMA:\n{presurvivors.schema}")
    _require_max_tokens(pooled, POOLED_MAX_TOKENS, "pooled")
    _require_max_tokens(terminal, TERMINAL_MAX_TOKENS, "terminal")

    start = time.monotonic()
    ids = tokenize([normalize_text(text) for text in presurvivors.column("text").to_pylist()])
    if len(ids) != presurvivors.num_rows:
        raise ValueError(f"tokenize returned {len(ids)} id arrays for {presurvivors.num_rows} texts")
    tokenized = time.monotonic()

    pooled_probs = score_ids(pooled, ids, POOLED_MAX_TOKENS)
    pooled_done = time.monotonic()

    band = np.flatnonzero(band_mask(pooled_probs))
    terminal_probs = np.full(len(ids), np.nan, dtype=np.float32)
    terminal_probs[band] = score_ids(
        terminal, [truncate_ids(ids[i], TERMINAL_MAX_TOKENS) for i in band], TERMINAL_MAX_TOKENS
    )
    terminal_done = time.monotonic()

    keep = keep_mask(pooled_probs, terminal_probs)
    n_tokens = np.array([len(x) for x in ids], dtype=np.int32)
    kept = presurvivors.filter(pa.array(keep))
    kept = kept.append_column("n_tokens", pa.array(n_tokens[keep], type=pa.int32()))
    kept = kept.append_column("pooled_prob", pa.array(pooled_probs[keep], type=pa.float32()))
    kept = kept.append_column("modernbert_prob", pa.array(terminal_probs[keep], type=pa.float32()))
    tombstones = tombstone_rows(presurvivors.column("doc_id").to_pylist(), pooled_probs, terminal_probs, keep)
    counters = PhaseBCounters(
        n_presurvivors=len(ids),
        n_hi_accept=int(accept_mask(pooled_probs).sum()),
        n_band=len(band),
        n_kept=int(keep.sum()),
        tokenize_seconds=tokenized - start,
        pooled_seconds=pooled_done - tokenized,
        terminal_seconds=terminal_done - pooled_done,
    )
    return kept.cast(KEPT_SCHEMA), tombstones, counters
