# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""M4 and M5 (``docs/reproducibility.md``) on the locally staged repos (``uv run python hf/models/stage.py`` first).

M4 creates fresh virtual environments (transformers 5.12.1 and 4.48.3) with only torch,
transformers, safetensors, huggingface_hub and fastText, serves the staged folders from an offline
Hugging Face cache, loads every model with zero initialization warnings and runs the card snippets
verbatim (``hf/models/verify_staged.py``); creating the environments is heavy, so run it in an Iris
job. M5 applies the ``upload-hf-model`` skill's pre-publish checks and verifies every file against
``manifest.json`` (``hf/models/check_repo.py``).
"""

import importlib.util
import subprocess
import sys

import pytest

from atlantic_cc import spec
from tests.parity.models import common

pytestmark = pytest.mark.slow
HF_DIR = common.REPO_ROOT / "hf" / "models"
REPOS = tuple(r.split("/")[1] for r in (spec.FASTTEXT_HF_REPO, spec.POOLED_HF_REPO, spec.TERMINAL_HF_REPO))


def _load(name: str):
    module_spec = importlib.util.spec_from_file_location(name, HF_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module", autouse=True)
def staged():
    missing = [r for r in REPOS if not (common.STAGING_DIR / r).is_dir()]
    if missing:
        pytest.skip(f"not staged: {missing}; run hf/models/stage.py")


@pytest.mark.parametrize("repo", REPOS)
def test_m5_prepublish_checks(repo):
    assert _load("check_repo").check(str(common.STAGING_DIR / repo)) == []


@pytest.mark.network
def test_m4_round_trip_in_fresh_environments():
    report = common.REPORT_DIR / "m4_roundtrip.json"
    cmd = [
        sys.executable,
        str(HF_DIR / "verify_staged.py"),
        "--staging",
        str(common.STAGING_DIR),
        "--report",
        str(report),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
