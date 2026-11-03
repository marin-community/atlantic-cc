# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests of the pooled-transformer ports, on a tiny random-weight model.

The references are the as-run Levanter outputs in ``tests/golden/models/synthetic.json`` for the same
weights and token batches, computed on an x86_64 CPU. In float32 both ports must match Levanter's
logits structurally. In bf16 the comparison is looser: the JAX port issues Levanter's operations, but
XLA:CPU's bf16 rounding points depend on the platform's compiled program, and the PyTorch port
emulates bf16 inputs with float32 storage. The exact same-machine and TPU comparisons run on the
cluster (``tests/parity/models``).
"""

from pathlib import Path

import numpy as np
import pytest

from atlantic_cc.models import Backend, Precision, Scorer, load_pooled, pooled_eqx
from tests.unit.models import goldens
from tests.unit.models import random_models as rm

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
torch = pytest.importorskip("torch")

from atlantic_cc.models import pooled_jax  # noqa: E402
from atlantic_cc.models.torch_scorer import load_pooled_scorer  # noqa: E402

CASES = goldens.cases("pooled", "random")
FP32_LOGIT_ATOL = 1e-4  # float32 ports vs Levanter float32: structural parity
BF16_PROB_ATOL = 2e-3  # bf16 ports vs Levanter bf16 on a CPU: max |dP| of rounding-boundary flips
PROB_ATOL = {Precision.FP32: 1e-5, Precision.BF16: BF16_PROB_ATOL}
INVARIANCE_ATOL = {Precision.FP32: 1e-6, Precision.BF16: 2e-3}


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("pooled_tiny")
    rm.write_pooled_hf(directory, rm.POOLED_SEED)
    return directory


@pytest.fixture(scope="module")
def scorers(model_dir) -> dict[tuple[Backend, Precision], Scorer]:
    return {(b, p): load_pooled(str(model_dir), b, precision=p) for b in Backend for p in Precision}


@pytest.fixture(scope="module")
def jax_logits(model_dir):
    """Levanter's logits function: the forward with its float32 cast inside the jit."""
    fns = {}
    for precision in Precision:
        dtype = jnp.bfloat16 if precision is Precision.BF16 else jnp.float32
        params, spec, _ = pooled_jax.load_model(str(model_dir), matmul_dtype=dtype)
        fn = jax.jit(lambda p, i, m, s=spec: pooled_jax.forward(p, i, m.astype(jnp.float32), s))
        fns[precision] = (fn, jax.device_put(params))
    return lambda precision, ids, mask: np.asarray(fns[precision][0](fns[precision][1], ids, mask))


@pytest.fixture(scope="module")
def levanter_cpu_torch(model_dir):
    """The PyTorch port configured like Levanter on a CPU: bf16 matmul inputs, float32 attention."""
    return load_pooled_scorer(str(model_dir), device="cpu", bf16_attention_inputs=False)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_jax_port_matches_levanter(case, scorers, jax_logits):
    ids, mask, _ = goldens.batch(case.batch)
    precision = Precision(case.precision)
    if precision is Precision.FP32:
        np.testing.assert_allclose(jax_logits(precision, ids, mask), case.logits, rtol=0, atol=FP32_LOGIT_ATOL)
    probs = scorers[(Backend.JAX, precision)].score(ids, mask)
    np.testing.assert_allclose(probs, case.probs, rtol=0, atol=PROB_ATOL[precision])


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_torch_port_matches_levanter(case, scorers, levanter_cpu_torch):
    ids, mask, _ = goldens.batch(case.batch)
    if case.precision == Precision.FP32:
        logits = scorers[(Backend.TORCH, Precision.FP32)].logits(ids, mask).numpy()
        np.testing.assert_allclose(logits, case.logits, rtol=0, atol=FP32_LOGIT_ATOL)
        return
    np.testing.assert_allclose(levanter_cpu_torch.score(ids, mask), case.probs, rtol=0, atol=BF16_PROB_ATOL)


def test_default_numerics_also_round_attention_operands(scorers, levanter_cpu_torch):
    """The released default rounds the attention operands to bf16 like a TPU; the CPU emulation does not."""
    ids, mask, _ = goldens.batch("pooled_tiny_256")
    tpu_like = scorers[(Backend.TORCH, Precision.BF16)].score(ids, mask)
    cpu_like = levanter_cpu_torch.score(ids, mask)
    assert np.abs(tpu_like - cpu_like).max() > 0
    np.testing.assert_allclose(tpu_like, cpu_like, rtol=0, atol=BF16_PROB_ATOL)


@pytest.mark.parametrize("key", [(b, p) for b in Backend for p in Precision], ids=lambda k: f"{k[0]}-{k[1]}")
def test_padding_and_batch_composition_do_not_change_scores(key, scorers):
    scorer, atol = scorers[key], INVARIANCE_ATOL[key[1]]
    ids, mask, lengths = goldens.batch("pooled_tiny_128")
    reference = scorer.score(ids, mask)
    # the same rows padded to 512 tokens (8 windows instead of 2) ...
    wide_ids = np.full((len(lengths), 512), rm.PAD, dtype=np.int32)
    wide_ids[:, :128], wide_mask = ids, np.pad(mask, ((0, 0), (0, 384)))
    np.testing.assert_allclose(scorer.score(wide_ids, wide_mask), reference, rtol=0, atol=atol)
    # ... each row scored alone, and a length that is not a multiple of the 64-token window
    alone = np.concatenate([scorer.score(ids[i : i + 1], mask[i : i + 1]) for i in range(len(lengths))])
    np.testing.assert_allclose(alone, reference, rtol=0, atol=atol)
    short = [i for i, n in enumerate(lengths) if n <= 100]
    np.testing.assert_allclose(scorer.score(ids[:, :100], mask[:, :100])[short], reference[short], rtol=0, atol=atol)


@pytest.mark.parametrize("backend", list(Backend))
def test_scorer_contract(backend, scorers):
    scorer = scorers[(backend, Precision.BF16)]
    ids, mask, _ = goldens.batch("pooled_tiny_128")
    assert scorer.max_tokens == rm.POOLED_TINY["max_seq_len"]
    probs = scorer.score(ids, mask)
    assert probs.dtype == np.float32 and probs.shape == (len(ids),)
    assert scorer.score(ids[:0], mask[:0]).shape == (0,)
    too_long = np.full((1, scorer.max_tokens + 64), rm.PAD, dtype=np.int32)
    with pytest.raises(ValueError, match="exceeds"):
        scorer.score(too_long, too_long != rm.PAD)


def test_eqx_reader_and_conversion(tmp_path):
    rm.write_pooled_levanter(tmp_path, rm.POOLED_SEED)
    num_layers = rm.POOLED_TINY["num_layers"]
    leaves = pooled_eqx.read_eqx_leaves(str(tmp_path / "model.eqx"), num_layers)
    expected = rm.pooled_leaves(rm.POOLED_SEED)
    assert list(leaves) == list(expected)
    for name, value in expected.items():
        np.testing.assert_array_equal(leaves[name], value)
    state = pooled_eqx.state_dict_from_eqx(leaves, num_layers)
    np.testing.assert_array_equal(state["layers.1.w1.weight"], expected["layers.1.w1"].T)
    with pytest.raises(ValueError, match="bytes left"):
        pooled_eqx.read_eqx_leaves(str(tmp_path / "model.eqx"), num_layers - 1)
