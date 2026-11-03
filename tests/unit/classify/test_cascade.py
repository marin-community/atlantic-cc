# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""U-CASCADE: band routing at the float32 thresholds, NaN handling and tombstone rows."""

import json
import math

import numpy as np

from atlantic_cc.classify.cascade import accept_mask, band_mask, keep_mask, tombstone_rows

NAN = np.float32("nan")


def below(x: float) -> np.float32:
    """The largest float32 strictly below float32(x)."""
    return np.nextafter(np.float32(x), np.float32(0))


def test_pooled_routing_at_the_float32_thresholds():
    pooled = np.array(
        [below(0.079), np.float32(0.079), below(0.8883), np.float32(0.8883), 1.0, 0.0, NAN], dtype=np.float32
    )
    np.testing.assert_array_equal(band_mask(pooled), [False, True, True, False, False, False, False])
    np.testing.assert_array_equal(accept_mask(pooled), [False, False, False, True, True, False, False])


def test_terminal_threshold_is_float32_0_4378():
    pooled = np.full(5, 0.5, dtype=np.float32)
    terminal = np.array(
        [
            below(0.4378),
            # 0.43779999 < 0.4378: a float64 comparison would drop it, production's float32 one kept it.
            np.float32(0.4378),
            np.float32(1 / (1 + math.exp(0.25))),  # sigmoid(-0.25) = 0.4378235, the lowest bf16-logit keep
            np.float32(0.9),
            NAN,
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(keep_mask(pooled, terminal), [False, True, True, True, False])


def test_pooled_accepts_are_kept_without_a_terminal_score():
    pooled = np.array([np.float32(0.8883), 0.99], dtype=np.float32)
    np.testing.assert_array_equal(keep_mask(pooled, np.array([NAN, NAN])), [True, True])


def test_tombstones_cover_exactly_the_dropped_documents_in_order():
    doc_ids = ["accepted", "pooled_drop", "band_reject", "band_keep", "nan_pooled", "band_reject_at_threshold"]
    pooled = np.array([0.95, 0.013863, 0.116129, 0.5, NAN, 0.5], dtype=np.float32)
    terminal = np.array([NAN, NAN, 0.055823, 0.9, NAN, below(0.4378)], dtype=np.float32)
    keep = keep_mask(pooled, terminal)

    lines = [json.dumps(row) for row in tombstone_rows(doc_ids, pooled, terminal, keep)]

    assert lines == [
        '{"doc_id": "pooled_drop", "pooled_prob": 0.013863, "modernbert_prob": null}',
        '{"doc_id": "band_reject", "pooled_prob": 0.116129, "modernbert_prob": 0.055823}',
        '{"doc_id": "nan_pooled", "pooled_prob": NaN, "modernbert_prob": null}',
        # Rejected at 0.43779996; six decimals round it to the threshold's decimal value.
        '{"doc_id": "band_reject_at_threshold", "pooled_prob": 0.5, "modernbert_prob": 0.4378}',
    ]
