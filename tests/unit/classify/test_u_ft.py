# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""U-FT: the fastText gate's call, label lookup, failure handling and float32 threshold.

Uses a tiny supervised model trained here (deterministic: one thread, fixed seed); the released
model's exact scores are checked against stored production values by the R1 parity harness.
"""

from pathlib import Path

import fasttext
import numpy as np
import pytest

from atlantic_cc import spec
from atlantic_cc.classify.fasttext_gate import FastTextGate, passes_gate

TEXT = "the history of the city"


@pytest.fixture(scope="module")
def model_path(tmp_path_factory) -> str:
    root = tmp_path_factory.mktemp("fasttext")
    train = root / "train.txt"
    lines = []
    for i in range(40):
        lines.append(f"__label__useful the history of the city and its people chapter {i % 7} explains the economy")
        lines.append(f"__label__no_useful click here login cart buy now cookie policy page {i % 5} menu")
    train.write_text("\n".join(lines) + "\n")
    model = fasttext.train_supervised(
        input=str(train), dim=8, epoch=5, minCount=1, wordNgrams=2, bucket=1000, thread=1, seed=7, verbose=0
    )
    path = root / "model.bin"
    model.save_model(str(path))
    return str(path)


def _useful(model, text: str) -> float:
    return dict((label, prob) for prob, label in model.f.predict(text, 2, 0.0, "strict"))["__label__useful"]


def test_scores_without_a_trailing_newline(model_path):
    """The newline variant (what ``model.predict`` sends) adds the EOS token and scores differently."""
    gate = FastTextGate.load(model_path)
    raw = fasttext.load_model(model_path)
    assert gate.p_useful(TEXT) == _useful(raw, TEXT)
    assert gate.p_useful(TEXT) != _useful(raw, TEXT + "\n")


def test_score_is_exactly_a_float32(model_path):
    score = FastTextGate.load(model_path).p_useful(TEXT)
    assert float(np.float32(score)) == score


def test_load_from_a_url_matches_a_local_load(model_path):
    assert FastTextGate.load(Path(model_path).as_uri()).p_useful(TEXT) == FastTextGate.load(model_path).p_useful(TEXT)


def test_unscoreable_text_scores_zero(model_path):
    """A lone surrogate cannot cross into C++; production scored such a document 0.0 (dropped)."""
    assert FastTextGate.load(model_path).p_useful("bad \ud800 text") == 0.0


class _Model:
    def __init__(self, predictions):
        self.calls = []
        outer = self

        class _F:
            @staticmethod
            def predict(text, k, threshold, on_unicode_error):
                outer.calls.append((text, k, threshold, on_unicode_error))
                return predictions

        self.f = _F()


@pytest.mark.parametrize(
    "predictions",
    [
        [(0.91, "__label__useful"), (0.09, "__label__no_useful")],
        [("__label__no_useful", 0.09), ("__label__useful", 0.91)],
        [(0.09, "__label__no_useful"), (0.91, "__label__useful")],
    ],
)
def test_useful_label_is_found_by_name_in_either_tuple_order(predictions):
    model = _Model(predictions)
    assert FastTextGate(model).p_useful("hi") == 0.91
    assert model.calls == [("hi", 2, 0.0, "strict")]


def test_missing_label_and_empty_text_score_zero():
    model = _Model([(1.0, "__label__no_useful")])
    assert FastTextGate(model).p_useful("hi") == 0.0
    assert FastTextGate(model).p_useful("") == 0.0
    assert model.calls == [("hi", 2, 0.0, "strict")]  # "" never reaches fastText


def test_threshold_is_float32_0_0048():
    t = spec.FASTTEXT_KEEP_AT
    below = np.nextafter(t, np.float32(0))
    assert passes_gate(float(t)) and not passes_gate(float(below))
    # Production compared the float32 score as a Python float with the float64 literal 0.0048;
    # the two comparisons agree on every float32 near the threshold.
    neighborhood = t + np.arange(-2000, 2001, dtype=np.float32) * np.float32(np.spacing(t))
    assert all(passes_gate(float(x)) == (not float(x) < 0.0048) for x in neighborhood)
    assert passes_gate(float("nan"))  # as-run: ``nan < 0.0048`` is False, so the doc was not dropped
