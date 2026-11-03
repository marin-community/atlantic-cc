# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests of the ModernBERT (Ettin-68M) terminal model ports, on a tiny random-weight ModernBERT.

The references are the as-run Levanter outputs in ``tests/golden/models/synthetic.json`` for the same
weights and token batches, computed on an x86_64 CPU. The tiny model has 4 layers (global attention on
layers 0 and 3), a ±8-token local window and different global/local RoPE thetas, so both attention
variants matter. In float32 the JAX port and Hugging Face's ModernBERT (eager and SDPA) must match
Levanter's logits structurally; bf16 is checked loosely here (see ``test_u_model_pooled.py``) and
exactly on the cluster. Empty filler rows score NaN in production (the mean pool divides by zero
tokens) and in the JAX port.
"""

from pathlib import Path

import numpy as np
import pytest

from atlantic_cc.models import Backend, Precision, Scorer, load_terminal
from tests.unit.models import goldens
from tests.unit.models import random_models as rm

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
torch = pytest.importorskip("torch")

from atlantic_cc.models import ettin_jax, ettin_torch  # noqa: E402

CASES = goldens.cases("ettin", "random")
ATTENTION = ("eager", "sdpa")
FP32_LOGIT_ATOL = 1e-4  # float32 ports vs Levanter float32: structural parity
PROB_ATOL = {Precision.FP32: 1e-5, Precision.BF16: 1e-2}  # JAX port vs Levanter
TORCH_BF16_PROB_ATOL = 0.03  # independent bf16 implementations; real-model gates are in tests/parity
INVARIANCE_ATOL = {Precision.FP32: 1e-6, Precision.BF16: 1e-2}


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("ettin_tiny")
    rm.write_ettin_hf(directory, rm.ETTIN_SEED)
    return directory


@pytest.fixture(scope="module")
def scorers(model_dir) -> dict[tuple[Backend, Precision], Scorer]:
    return {(b, p): load_terminal(str(model_dir), b, precision=p) for b in Backend for p in Precision}


@pytest.fixture(scope="module")
def jax_logits(model_dir):
    """Levanter's logits function: the forward with its float32 cast inside the jit."""
    fns = {}
    for precision in Precision:
        dtype = jnp.bfloat16 if precision is Precision.BF16 else jnp.float32
        params, spec, _ = ettin_jax.load_model(str(model_dir), dtype=dtype)
        fn = jax.jit(lambda p, i, m, s=spec: ettin_jax.logits_fn(p, i, m, s).astype(jnp.float32))
        fns[precision] = (fn, jax.device_put(params))
    return lambda precision, ids, mask: np.asarray(fns[precision][0](fns[precision][1], ids, mask))


@pytest.fixture(scope="module")
def torch_models(model_dir) -> dict:
    return {
        (attn, precision): ettin_torch.load_model(
            str(model_dir),
            dtype=torch.bfloat16 if precision is Precision.BF16 else torch.float32,
            attn_implementation=attn,
        )
        for attn in ATTENTION
        for precision in Precision
    }


def _real_rows(lengths: list[int]) -> list[int]:
    return [i for i, n in enumerate(lengths) if n]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_jax_port_matches_levanter(case, scorers, jax_logits):
    ids, mask, lengths = goldens.batch(case.batch)
    precision, rows = Precision(case.precision), _real_rows(lengths)
    if precision is Precision.FP32:
        logits = jax_logits(precision, ids, mask)
        np.testing.assert_allclose(logits[rows], case.logits[rows], rtol=0, atol=FP32_LOGIT_ATOL)
    probs = scorers[(Backend.JAX, precision)].score(ids, mask)
    np.testing.assert_allclose(probs, case.probs, rtol=0, atol=PROB_ATOL[precision], equal_nan=True)


@pytest.mark.parametrize("attn", ATTENTION)
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_torch_matches_levanter(case, attn, torch_models):
    ids, mask, lengths = goldens.batch(case.batch)
    precision, rows = Precision(case.precision), _real_rows(lengths)
    with torch.no_grad():
        logits = torch_models[(attn, precision)](
            input_ids=torch.from_numpy(ids[rows].astype(np.int64)), attention_mask=torch.from_numpy(mask[rows])
        ).logits.float()
    if precision is Precision.FP32:
        np.testing.assert_allclose(logits.numpy(), case.logits[rows], rtol=0, atol=FP32_LOGIT_ATOL)
    probs = torch.softmax(logits, dim=-1)[:, 1].numpy()
    np.testing.assert_allclose(probs, case.probs[rows], rtol=0, atol=TORCH_BF16_PROB_ATOL)


@pytest.mark.parametrize("key", [(b, p) for b in Backend for p in Precision], ids=lambda k: f"{k[0]}-{k[1]}")
def test_padding_and_batch_composition_do_not_change_scores(key, scorers):
    scorer, atol = scorers[key], INVARIANCE_ATOL[key[1]]
    ids, mask, lengths = goldens.batch("ettin_tiny_64")
    rows = _real_rows(lengths)
    reference = scorer.score(ids, mask)[rows]
    wide_ids = np.full((len(lengths), 128), rm.PAD, dtype=np.int32)
    wide_ids[:, :64], wide_mask = ids, np.pad(mask, ((0, 0), (0, 64)))
    np.testing.assert_allclose(scorer.score(wide_ids, wide_mask)[rows], reference, rtol=0, atol=atol)
    alone = np.concatenate([scorer.score(ids[i : i + 1], mask[i : i + 1]) for i in rows])
    np.testing.assert_allclose(alone, reference, rtol=0, atol=atol)


def test_released_layout_loads_without_new_weights(model_dir, caplog):
    model = ettin_torch.load_model(str(model_dir), dtype=torch.float32, attn_implementation="eager")
    assert model.config.architectures == ["ModernBertForSequenceClassification"]
    assert model.config.id2label == {0: "not_useful", 1: "useful"}
    assert not any("newly initialized" in r.getMessage() or "MISSING" in r.getMessage() for r in caplog.records)
    spec = ettin_jax.spec_from_config(model.config, jnp.float32)
    assert (spec.global_every, spec.local_radius) == (3, rm.ETTIN_TINY["local_attention"] // 2)
    assert (spec.global_rope_theta, spec.local_rope_theta) == (160000.0, 10000.0)


def test_nonzero_classifier_bias_is_rejected():
    state = rm.ettin_state_dict(rm.ETTIN_SEED)
    state["classifier.bias"] = np.ones(2, dtype=np.float32)
    with pytest.raises(ValueError, match="classifier.bias"):
        ettin_jax.params_from_state_dict(state, num_layers=4, num_heads=2, dtype=np.float32)
