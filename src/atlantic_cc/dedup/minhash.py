# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""LSH buckets of a document: dupekit ``CleanText -> MinHash(286, 5, 42) -> MinHashLSH(26)`` (as-run ``docs.py``).

``marin-dupekit-native==0.1.6.dev31131847556`` (PyPI) is the as-run build; its Rust sources are byte-identical
to the tag's ``lib/dupekit/rust/src``. :mod:`atlantic_cc.dedup.oracle` is a pure-Python port of the same
pipeline that the tests hold the native wheel to.
"""

import dupekit_native as dupekit
import numpy as np
import pyarrow as pa

from atlantic_cc.dedup.params import NGRAM_SIZE, NUM_BANDS, NUM_PERMS, SEED

_PIPELINE = [
    dupekit.Transformation.CleanText(input_col="text", output_col="clean_text"),
    dupekit.Transformation.MinHash(
        input_col="clean_text", output_col="signature", num_perms=NUM_PERMS, ngram_size=NGRAM_SIZE, seed=SEED
    ),
    dupekit.Transformation.MinHashLSH(input_col="signature", output_col="buckets", num_bands=NUM_BANDS),
    dupekit.Transformation.SelectColumns(columns=["buckets"]),
]


def minhash_buckets(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """``(has_signature, buckets)``: a bool per text and an ``(n, 26)`` uint64 array (zeros where unsigned)."""
    n = len(texts)
    buckets = np.zeros((n, NUM_BANDS), dtype=np.uint64)
    if n == 0:
        return np.zeros(0, dtype=bool), buckets
    result = dupekit.transform(pa.record_batch({"text": pa.array(texts, type=pa.large_string())}), _PIPELINE)
    column = result.column("buckets")
    valid = np.asarray(column.is_valid().to_numpy(zero_copy_only=False), dtype=bool)
    if valid.any():
        flat = column.filter(pa.array(valid)).flatten().to_numpy(zero_copy_only=False)
        if len(flat) != int(valid.sum()) * NUM_BANDS:
            raise ValueError(f"dupekit returned {len(flat)} buckets for {int(valid.sum())} signatures")
        buckets[valid] = flat.astype(np.uint64).reshape(-1, NUM_BANDS)
    return valid, buckets
