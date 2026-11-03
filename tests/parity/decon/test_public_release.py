# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""R5 (public): replaying decontamination on published release files reproduces ``decon_matches`` exactly.

Needs network access (data.commoncrawl.org) and the S' release assets in the directory named by
``ATLANTIC_CC_DECON_ARTIFACTS`` (``eval_hits.npy``, ``eval_hit_names.parquet``). Release files are cached
in ``ATLANTIC_CC_RELEASE_CACHE`` when set, else in a pytest temporary directory::

    ATLANTIC_CC_DECON_ARTIFACTS=<dir> uv run pytest -m network tests/parity/decon
"""

import os
from pathlib import Path

import pytest

from atlantic_cc.decon.artifacts import (
    EVAL_HIT_NAMES_FILE,
    EVAL_HITS_FILE,
    load_drop_hashes,
    load_eval_hit_names,
    load_eval_hits,
)
from tests.parity.decon.public_release import CASES, run_case

pytestmark = [pytest.mark.network, pytest.mark.slow]


@pytest.fixture(scope="module")
def replay_inputs():
    artifacts = os.environ.get("ATLANTIC_CC_DECON_ARTIFACTS")
    if not artifacts:
        pytest.skip("set ATLANTIC_CC_DECON_ARTIFACTS to the directory with the S' release assets")
    root = Path(artifacts)
    return (
        load_eval_hits(str(root / EVAL_HITS_FILE)),
        load_drop_hashes(),
        load_eval_hit_names(str(root / EVAL_HIT_NAMES_FILE)),
    )


@pytest.fixture(scope="module")
def release_cache(tmp_path_factory) -> Path:
    cache = os.environ.get("ATLANTIC_CC_RELEASE_CACHE")
    return Path(cache) if cache else tmp_path_factory.mktemp("release")


@pytest.mark.parametrize("case", CASES, ids=[c.name.split()[0] for c in CASES])
def test_replay_reproduces_published_decon_matches(case, replay_inputs, release_cache, tmp_path):
    membership, drop, names = replay_inputs
    result = run_case(case, release_cache, tmp_path, membership, drop, names, workers=2)
    assert result["mismatched"] == 0, result["mismatched_ids"]
    assert result["matched_hashes_without_names"] == 0
    assert result.get("flagged_equal_expected", True)
