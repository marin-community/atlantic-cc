# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""M3 (``docs/reproducibility.md``): the released fastText model reproduces the stored ``fasttext_score``.

- The production call (low-level predict, no trailing newline) on the normalized release text gives
  the stored score, on every document of the release row groups the model golden draws from
  (83,262 documents of four crawls); the keep decisions at float32(0.0048) are identical.
- Every installable fastText package scores identically.
- The high-level ``predict`` (which appends a newline, hence an end-of-sentence token) deviates;
  the size of that deviation is recorded for the model card.

Scores are float32 dot products, so a different build can move a score by about one float32 ulp
(observed on a macOS arm64 source build of fasttext-wheel 0.9.2: 2.7% of scores, max 2.7e-7). The
exact contract is the ``fasttext-wheel`` 0.9.2 manylinux wheel on Linux x86_64.
"""

import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pytest

from atlantic_cc.classify.fasttext_gate import FastTextGate, passes_gate
from atlantic_cc.classify.text import normalize_text
from atlantic_cc.models import resolve_model_dir
from tests.parity.models import common

pytestmark = [pytest.mark.slow, pytest.mark.network]

VARIANT_SCRIPT = Path(__file__).with_name("fasttext_variant.py")
# package specs installed into isolated environments; one may fail to build on a given platform
VARIANTS = {
    "fasttext-wheel==0.9.2": ["fasttext-wheel==0.9.2", "numpy==2.3.5"],
    "fasttext-wheel==0.9.2 numpy<2": ["fasttext-wheel==0.9.2", "numpy<2"],
    "fasttext==0.9.3": ["fasttext==0.9.3", "numpy==2.3.5"],
    "fasttext-numpy2-wheel==0.9.2": ["fasttext-numpy2-wheel==0.9.2", "numpy==2.3.5"],
    "fasttext-numpy2==0.10.4": ["fasttext-numpy2==0.10.4", "numpy==2.3.5"],
}
MAX_ABS_DIFF = 1e-6  # from the stored score, and across packages
N_VARIANT_TEXTS = 500


@pytest.fixture(scope="module")
def model_path() -> str:
    return str(Path(resolve_model_dir(common.model_source("fasttext"))) / "model.bin")


@pytest.fixture(scope="module")
def texts() -> tuple[list[str], np.ndarray]:
    rows = common.release_rows(common.realdocs_golden(), ["text", "fasttext_score"])
    normalized = [normalize_text(t) for t in rows.column("text").to_pylist()]
    return normalized, rows.column("fasttext_score").to_numpy()


def test_m3_matches_stored_scores(model_path, texts):
    normalized, stored = texts
    gate = FastTextGate.load(model_path)
    mine = np.array([gate.p_useful(t) for t in normalized], dtype=np.float32)
    with_newline = np.array([gate.p_useful(t + "\n") for t in normalized], dtype=np.float32)
    keep_newline = np.array([passes_gate(p) for p in with_newline])
    report = {
        "n_documents": len(normalized),
        "low_level_vs_stored": common.abs_diff_summary(mine, stored.astype(np.float32)),
        "with_newline_vs_low_level": common.abs_diff_summary(with_newline, mine),
        "with_newline_decision_flips": int(np.sum(~keep_newline)),
        "platform": f"{platform.system()} {platform.machine()}",
        "fasttext": importlib.metadata.version("fasttext-wheel"),
    }
    common.write_report("m3_fasttext_vs_stored", report)
    assert all(passes_gate(p) for p in mine), "every released document passed the gate in production"
    assert report["low_level_vs_stored"]["max"] <= MAX_ABS_DIFF
    assert report["with_newline_vs_low_level"]["max"] > 1e-3  # the end-of-sentence token matters


def test_m3_identical_across_packages(model_path, texts, tmp_path):
    normalized, _ = texts
    texts_file = tmp_path / "texts.json"
    texts_file.write_text(json.dumps(normalized[:N_VARIANT_TEXTS]))
    results, failures = {}, {}
    for name, packages in VARIANTS.items():
        out = tmp_path / f"{len(results) + len(failures)}.json"
        cmd = ["uv", "run", "--isolated", "--no-project", "--python", "3.12"]
        cmd += [arg for p in packages for arg in ("--with", p)]
        cmd += ["python", "-I", str(VARIANT_SCRIPT), model_path, str(texts_file), str(out)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode:
            failures[name] = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "failed"
            continue
        results[name] = json.loads(out.read_text())
    reference = results["fasttext-wheel==0.9.2"]["low_level"]
    report = {
        "installed": {n: r["packages"] | {"numpy": r["numpy"]} for n, r in results.items()},
        "failed_to_install": failures,
        "max_abs_diff_vs_fasttext_wheel": {
            n: float(np.max(np.abs(np.array(r["low_level"]) - reference))) for n, r in results.items()
        },
    }
    common.write_report("m3_fasttext_packages", report)
    assert len(results) >= 2
    assert all(d == 0.0 for d in report["max_abs_diff_vs_fasttext_wheel"].values())
