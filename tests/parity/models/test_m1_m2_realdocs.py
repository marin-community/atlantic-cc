# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""M1 (ModernBERT terminal) and M2 (pooled transformer) gates on real documents (``docs/reproducibility.md``).

Scores the golden's ~2,000 released documents with every port available here (see ``realdocs.py``) and
writes ``.cache/models/parity/m1_m2_realdocs.json``. PyTorch runs on the CPU. Two terminal forwards are slow
on CPUs: JAX on XLA:CPU computes full attention matrices on every layer, and PyTorch bf16 is slow without
bfloat16 instructions (on 16 AMD EPYC 7B12 cores, 3.4 s per band document against 0.43 s in float32).
Those runs score the terminal model only on the golden's ``terminal_bf16`` sample (64 band documents, half
near the threshold); PyTorch float32, and JAX on an accelerator, score every band document.
"""

import pytest

from atlantic_cc.models import Backend, Precision
from tests.parity.models import common, realdocs

pytestmark = [pytest.mark.slow, pytest.mark.network]
jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")

# Gates: structural parity in float32; bf16 within rounding-boundary flips; decisions vs production
FP32_PROB_ATOL = 1e-5  # either port in float32 vs Levanter float32 (CPU)
BF16_VS_LEVANTER_CPU = {"p99": 5e-4, "max": 2e-3}  # bf16 ports vs Levanter bf16 (CPU), pooled
POOLED_VS_TPU_P99 = 5e-3
TERMINAL_VS_TPU_P99 = 0.02
MIN_AGREEMENT = 0.995  # decisions on the natural strata
FLIP_MARGIN = {"pooled": 0.02, "terminal": 0.05}  # every flipped document lies this close to its threshold
RUNS = (
    realdocs.Run(Backend.TORCH, Precision.FP32),
    realdocs.Run(Backend.TORCH, Precision.BF16),
    realdocs.Run(Backend.TORCH, Precision.BF16, levanter_cpu=True),
    realdocs.Run(Backend.JAX, Precision.FP32),
    realdocs.Run(Backend.JAX, Precision.BF16),
)


def terminal_selection(run: realdocs.Run, jax_on_cpu: bool) -> str:
    """The band documents whose terminal score ``run`` computes (see ``realdocs.terminal_rows``)."""
    if run.backend is Backend.JAX:
        return "terminal_bf16" if jax_on_cpu else "all"
    return "all" if run.precision is Precision.FP32 else "terminal_bf16"


@pytest.fixture(scope="module")
def results() -> dict:
    golden = common.realdocs_golden()
    docs = realdocs.load_documents(golden, common.fetch_texts(golden))
    jax_on_cpu = realdocs.accelerator() == "cpu"
    out, terminal = {}, {}
    for run in RUNS:
        terminal[run.name] = terminal_selection(run, jax_on_cpu)
        out |= realdocs.run_all([run], docs, staging=common.STAGING_DIR, terminal=terminal[run.name], device="cpu")
    summaries = {name: {k: v for k, v in r.items() if k != "scores"} for name, r in out.items()}
    common.write_report(
        "m1_m2_realdocs", {"environment": realdocs.environment(), "terminal": terminal, "runs": summaries}
    )
    return out


@pytest.mark.parametrize("model", ["pooled", "terminal"])
@pytest.mark.parametrize("backend", list(Backend))
def test_float32_ports_match_levanter(results, model, backend):
    levanter = results[f"{backend}-fp32"][model]["levanter"]["fp32"]
    assert levanter["n"] > 0 and levanter["max"] <= FP32_PROB_ATOL


@pytest.mark.parametrize("run", ["torch-bf16-levanter-cpu", "jax-bf16"])
def test_bf16_pooled_ports_match_levanter_cpu(results, run):
    levanter = results[run]["pooled"]["levanter"]["bf16"]
    assert levanter["p99"] <= BF16_VS_LEVANTER_CPU["p99"] and levanter["max"] <= BF16_VS_LEVANTER_CPU["max"]


@pytest.mark.parametrize("model", ["pooled", "terminal"])
@pytest.mark.parametrize("run", ["torch-bf16", "jax-bf16"])
def test_production_numerics_match_stored_tpu_scores(results, run, model):
    stored = results[run][model]["stored_tpu"]
    assert stored["all"]["abs_diff"]["p99"] <= (POOLED_VS_TPU_P99 if model == "pooled" else TERMINAL_VS_TPU_P99)
    natural = stored["natural_strata"]["decisions"]
    if natural["n"]:
        assert natural["agreement"] >= MIN_AGREEMENT
    assert stored["all"]["decisions"]["max_flip_distance_to_threshold"] < FLIP_MARGIN[model]
