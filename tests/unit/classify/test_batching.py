# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""U-BATCH: the batches the models see (length sort, batch size, buckets, empty rows, padding)."""

import numpy as np
import pytest

from atlantic_cc.classify.batching import bucket_for, score_ids
from atlantic_cc.spec import CLS_TOKEN_ID, PAD_TOKEN_ID, SEP_TOKEN_ID

MARKER_BASE = 1000  # document i carries token MARKER_BASE + i, so a batch row identifies its document


class RecordingScorer:
    """Returns i / 100 for the row of document i (0 for empty rows) and keeps every batch it saw."""

    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens
        self.batches: list[tuple[np.ndarray, np.ndarray]] = []

    def score(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        self.batches.append((input_ids.copy(), attention_mask.copy()))
        doc = np.where(attention_mask[:, 0], input_ids[:, 1] - MARKER_BASE, 0)
        return (doc / 100).astype(np.float32)


def doc_ids(index: int, length: int) -> np.ndarray:
    return np.array([CLS_TOKEN_ID] + [MARKER_BASE + index] * (length - 2) + [SEP_TOKEN_ID], dtype=np.int32)


@pytest.mark.parametrize(
    ("n_tokens", "max_len", "bucket"),
    [
        (3, 8192, 128),
        (128, 8192, 128),
        (129, 8192, 256),
        (4097, 8192, 8192),
        (8192, 8192, 8192),
        (2048, 2048, 2048),
        (1025, 2048, 2048),
        (3000, 3000, 3000),  # a cap between buckets is used as is (as-run bucket_for)
    ],
)
def test_bucket_is_smallest_fitting_length_up_to_the_cap(n_tokens, max_len, bucket):
    assert bucket_for(n_tokens, max_len) == bucket


def test_batches_are_stable_length_sorted_bucketed_and_filled_to_32():
    lengths = [300] + [5] * 32
    lengths[7] = 129
    ids = [doc_ids(i, n) for i, n in enumerate(lengths)]
    scorer = RecordingScorer(max_tokens=8192)

    probs = score_ids(scorer, ids, 8192)

    np.testing.assert_array_equal(probs, (np.arange(33) / 100).astype(np.float32))
    (first_ids, first_mask), (second_ids, second_mask) = scorer.batches
    # Ties keep input order; the 129-token document closes the first batch, which pads to 256.
    assert first_ids.shape == (32, 256)
    np.testing.assert_array_equal(first_ids[:, 1] - MARKER_BASE, [*range(1, 7), *range(8, 33), 7])
    # The 300-token document is alone in a 512 batch with 31 empty rows.
    assert second_ids.shape == (32, 512)
    assert second_ids[0, 1] == MARKER_BASE
    assert (second_ids[1:] == PAD_TOKEN_ID).all() and not second_mask[1:].any()
    for batch_ids, mask in scorer.batches:
        real = mask.sum(axis=1)
        for row in range(32):
            assert mask[row, : real[row]].all() and not mask[row, real[row] :].any()
            assert (batch_ids[row, real[row] :] == PAD_TOKEN_ID).all()
        assert batch_ids.dtype == np.int32 and mask.dtype == bool
    np.testing.assert_array_equal(first_mask.sum(axis=1), [5] * 31 + [129])


def test_no_documents_means_no_model_call():
    scorer = RecordingScorer(max_tokens=2048)
    probs = score_ids(scorer, [], 2048)
    assert probs.dtype == np.float32 and probs.shape == (0,)
    assert scorer.batches == []
