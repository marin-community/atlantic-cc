# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The fastText gate: P(``__label__useful``) on the normalized text, keep iff >= 0.0048.

Production scored with the low-level ``model.f.predict(text, 2, 0.0, "strict")`` and **no trailing
newline**. fastText's high-level ``model.predict`` appends ``"\\n"``, which adds an end-of-sentence
token (and a bigram with it) and changes the probability; the threshold was calibrated without it.
fastText returns ``exp(log(p + 1e-5))`` as float32, so a score is P(useful) plus about 1e-5.
"""

import logging
import os
import shutil
import tempfile

import fsspec
import numpy as np

from atlantic_cc import spec

logger = logging.getLogger(__name__)

USEFUL_LABEL = "__label__useful"
_PREDICT_K = 2  # both labels, so the useful one is always present
_COPY_BUFFER_BYTES = 64 << 20


class FastTextGate:
    """Scores normalized text with the Atlantic-CC fastText model exactly as production did."""

    def __init__(self, model) -> None:
        self._model = model

    @classmethod
    def load(cls, path: str) -> "FastTextGate":
        """Load ``model.bin`` from a local path or an fsspec URL (copied to a temporary file first)."""
        import fasttext  # noqa: PLC0415 - native; loaded on use because stage CLIs import this module

        if "://" not in path:
            return cls(fasttext.load_model(path))
        with tempfile.TemporaryDirectory(prefix="atlantic_cc_fasttext_") as tmp:
            local = os.path.join(tmp, "model.bin")
            logger.info("copying fastText model %s -> %s", path, local)
            with fsspec.open(path, "rb") as src, open(local, "wb") as dst:
                shutil.copyfileobj(src, dst, _COPY_BUFFER_BYTES)
            return cls(fasttext.load_model(local))

    def p_useful(self, normalized_text: str) -> float:
        """P(useful) for one normalized document; 0.0 for "" and for a document fastText cannot score."""
        if not normalized_text:
            return 0.0
        try:
            predictions = self._model.f.predict(normalized_text, _PREDICT_K, 0.0, "strict")
        except Exception as e:  # as-run: an unscoreable document counts as not useful
            logger.warning("fastText predict failed on a %d-char doc (%s); scoring it 0.0", len(normalized_text), e)
            return 0.0
        for a, b in predictions:
            prob, label = (a, b) if isinstance(b, str) else (b, a)
            if label == USEFUL_LABEL:
                return float(prob)
        return 0.0


def passes_gate(prob: float) -> bool:
    """Keep unless ``float32(prob) < float32(0.0048)`` (production: ``if prob < threshold: continue``).

    Production compared the Python float against the float64 literal 0.0048; for the float32 values
    fastText returns the two comparisons agree, since no float32 lies in [0.0048, float32(0.0048)).
    """
    return not np.float32(prob) < spec.FASTTEXT_KEEP_AT
