# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The committed goldens give R2/R3 exactly production's outputs, as read from Marin's GCS (maintainers)."""

import math

import pytest

from tests.parity.r2_asrun import FIXTURES, asrun_from_golden, load_asrun
from tests.parity.r2_compare import survivors_from_outputs
from tests.parity.r2_phase_b import PHASE_A_COUNTERS, PHASE_B_COUNTERS

pytestmark = pytest.mark.gcs

CATALOG_FIELDS = (
    *PHASE_A_COUNTERS.values(),
    *PHASE_B_COUNTERS.values(),
    "device",
    "n_devices",
    "tokenize_s",
    "pooled_s",
    "score_s",
    "wall_s",
)


def survivor_rows(outputs) -> list[tuple]:
    """Every survivor in comparison order, with NaN made comparable."""
    survivors = survivors_from_outputs(outputs.kept, outputs.tombstones)
    return [
        (doc_id, s.route, s.pooled, None if math.isnan(s.terminal) else s.terminal, s.exact)
        for doc_id, s in survivors.items()
    ]


@pytest.mark.parametrize("name", ["F1", "F2", "F3", "F4", "F5"])
def test_golden_equals_production_outputs(name):
    golden, production = asrun_from_golden(FIXTURES[name]), load_asrun(FIXTURES[name])
    assert survivor_rows(golden) == survivor_rows(production)
    assert {f: golden.catalog_row[f] for f in CATALOG_FIELDS} == {f: production.catalog_row[f] for f in CATALOG_FIELDS}
