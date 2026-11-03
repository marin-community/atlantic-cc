# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The c075 solver's SIMD tie-order dependence, pinned per platform.

``np.argpartition`` keeps an implementation-defined subset of tied values. numpy 2.3.5 dispatches
``ArgQSelect_AVX512_SKX`` on AVX-512 x86_64 CPUs, ``ArgQSelect_AVX2`` on AVX2-only ones and a scalar introselect on
ARM, and they keep different subsets. The solver calls it twice for batches of more than 256 members (the 32 rarest
probe n-grams; the 32 best-sharing candidates), so those batches can remove different documents on different CPUs.
The release was built on TPU v4 hosts (AMD EPYC 7B12: AVX2, no AVX-512). The dedup CLI therefore masks AVX-512 by
default (``--simd mask-avx512``), and exact parity is claimed for x86_64 with AVX2 only: on ARM, and on x86_64
CPUs without AVX2 (Intel before Haswell), these tests document the scalar kernel's choice and are otherwise
skipped. Masking AVX2 as well leaves x86_64 with the scalar selection, which keeps the same tied items as ARM.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy._core import _multiarray_umath

from atlantic_cc.dedup.runtime import NUMPY_DISABLE_ENV, enabled_avx512_features, tie_order_platform

GOLDENS = json.loads((Path(__file__).parents[2] / "golden" / "dedup" / "asrun_goldens.json").read_text())["tie_probe"]
RELEASE_PLATFORM = "x86_64-avx2"
_PROBE = (
    "import json, numpy as np\n"
    "from tests.unit.dedup.test_u_dedup_tie_order import tie_probe\n"
    "print(json.dumps(tie_probe()))\n"
)


def tie_probe() -> dict[str, list[int]]:
    """Must equal ``tests/golden/dedup/generate_asrun.py::tie_probe``."""
    rng = np.random.default_rng(2026)
    counts = rng.integers(1, 6, 500).astype(np.int64)
    shared = rng.integers(1, 5, 300).astype(np.int64)
    return {
        "rarest_32": sorted(np.argpartition(counts, 32)[:32].tolist()),
        "best_32": sorted(np.argpartition(-shared, 31)[:32].tolist()),
    }


def test_argpartition_ties_match_the_recorded_choice_for_this_platform():
    tag = tie_order_platform()
    if tag not in GOLDENS:
        pytest.skip(f"no recorded tie order for {tag}")
    assert tie_probe() == GOLDENS[tag]


def _probe_with_mask(features: list[str]) -> dict[str, list[int]]:
    """The tie probe in a fresh interpreter with these NumPy CPU features disabled."""
    env = {**os.environ, NUMPY_DISABLE_ENV: " ".join(features)}
    root = Path(__file__).parents[3]
    masked = subprocess.run([sys.executable, "-c", _PROBE], env=env, cwd=root, capture_output=True, text=True)
    assert masked.returncode == 0, masked.stderr
    return json.loads(masked.stdout)


_X86_ONLY = pytest.mark.skipif(not tie_order_platform().startswith("x86_64"), reason="ARM has no SIMD selection")
# The release's order comes from NumPy's AVX2 selection kernel. Without AVX2 (ARM, or x86_64 before Haswell)
# NumPy has only the scalar selection, so no mask can restore that order.
_AVX2_ONLY = pytest.mark.skipif(
    not _multiarray_umath.__cpu_features__.get("AVX2"), reason="NumPy reports no AVX2 on this CPU"
)


@_AVX2_ONLY
def test_masking_avx512_restores_the_release_tie_order():
    enabled = enabled_avx512_features()
    probe = _probe_with_mask(enabled) if enabled else tie_probe()
    assert probe == GOLDENS[RELEASE_PLATFORM]


@_X86_ONLY
def test_masking_avx2_too_gives_the_tie_order_of_arm_hosts():
    """Without a SIMD selection kernel, x86_64 keeps the same tied items as ARM's scalar code."""
    assert _probe_with_mask([*enabled_avx512_features(), "AVX2"]) == GOLDENS["arm64-scalar"]
