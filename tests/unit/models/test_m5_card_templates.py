# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test M5 (``docs/reproducibility.md``) on the model-card templates: the required card content.

The staged cards (numbers filled in) are checked by ``tests/parity/models/test_m4_m5_staged.py``.
"""

import hashlib
import importlib.util
from pathlib import Path

import pytest

from atlantic_cc import spec

HF_DIR = Path(__file__).resolve().parents[3] / "hf" / "models"
REPOS = tuple(r.split("/")[1] for r in (spec.FASTTEXT_HF_REPO, spec.POOLED_HF_REPO, spec.TERMINAL_HF_REPO))


def _check_repo():
    spec = importlib.util.spec_from_file_location("check_repo", HF_DIR / "check_repo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_license_file_is_the_official_openmdw_text():
    digest = hashlib.sha256((HF_DIR / "LICENSE.OpenMDW-1.1").read_bytes()).hexdigest()
    assert digest == _check_repo().OPENMDW_SHA256


@pytest.mark.parametrize("repo", REPOS)
def test_card_template_has_required_content(repo):
    check_repo = _check_repo()
    readme = (HF_DIR / repo / "README.md").read_text()
    problems = check_repo.problems(readme, (HF_DIR / "LICENSE.OpenMDW-1.1").read_bytes())
    # templates still hold {{placeholders}} that stage.py fills
    assert [p for p in problems if "'{{'" not in p] == []
    assert "](cascade.png)" in readme and (HF_DIR / repo / "cascade.png").is_file()
    # the banner comes first; stage.py copies it from docs/assets/
    body = readme.split("\n---\n", 1)[1].lstrip("\n")
    assert body.startswith("![") and "](atlantic-cc-header.png)" in body.splitlines()[0]
    assert (HF_DIR.parents[1] / "docs" / "assets" / "atlantic-cc-header.png").is_file()
