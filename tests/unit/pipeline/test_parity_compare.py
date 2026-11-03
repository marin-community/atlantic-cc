# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The R2/R3 comparison: routes recovered from output files, deltas, and flip attribution."""

import gzip
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from tests.parity.r2_compare import Route, Survivor, compare_survivors, survivors_from_outputs

DATA = Path(__file__).parent / "data"
NAN = math.nan


def test_f3_outputs_give_every_survivor_its_route():
    kept = pq.read_table(DATA / "f3_4ff1f9108b5a_kept_scores.parquet")
    lines = gzip.decompress((DATA / "f3_4ff1f9108b5a_tombstones.jsonl.gz").read_bytes()).decode().splitlines()
    survivors = survivors_from_outputs(kept, [json.loads(line) for line in lines])

    comparison = compare_survivors(survivors, survivors)

    # The as-run marker: 9,513 survivors, 1,396 pooled accepts, 3,304 band, 2,830 kept.
    routes = comparison.summary["routes_asrun_to_ours"]
    assert {r: routes[r][r] for r in Route} == {"accept": 1396, "band_keep": 1434, "band_reject": 1870, "drop": 4813}
    assert comparison.summary["pooled_abs_delta"]["n_zero"] == 9513
    assert (comparison.summary["decision_agreement"], comparison.flips) == (1.0, [])


def test_flips_are_attributed_to_the_threshold_that_decided_them():
    terminal_threshold, lo, hi = (float(np.float32(x)) for x in (0.4378, 0.079, 0.8883))
    asrun = {
        "terminal_flip": Survivor(Route.BAND_KEEP, 0.5, 0.4378235, exact=True),
        "near_accept": Survivor(Route.ACCEPT, 0.889, NAN, exact=True),
        "near_drop": Survivor(Route.DROP, 0.0785, NAN, exact=False),
        "far_flip": Survivor(Route.BAND_REJECT, 0.3, 0.2, exact=False),
        "asrun_only": Survivor(Route.ACCEPT, 0.95, NAN, exact=True),
    }
    ours = {
        "terminal_flip": Survivor(Route.BAND_REJECT, 0.5001, 0.437583, exact=False),
        "near_accept": Survivor(Route.BAND_REJECT, 0.888, 0.3, exact=False),
        "near_drop": Survivor(Route.BAND_KEEP, 0.0795, 0.6, exact=True),
        "far_flip": Survivor(Route.BAND_KEEP, 0.3, 0.9, exact=True),
        "ours_only": Survivor(Route.DROP, 0.01, NAN, exact=False),
    }

    comparison = compare_survivors(ours, asrun)

    flips = {f["doc_id"]: f for f in comparison.flips}
    assert {d: (f["stage"], f["threshold"]) for d, f in flips.items()} == {
        "terminal_flip": ("terminal", terminal_threshold),
        "near_accept": ("pooled", hi),
        "near_drop": ("pooled", lo),
        "far_flip": ("terminal", terminal_threshold),
    }
    assert flips["terminal_flip"]["asrun_distance"] == pytest.approx(0.4378235 - terminal_threshold)
    assert [d for d, f in flips.items() if not f["within_epsilon"]] == ["far_flip"]
    summary = comparison.summary
    assert summary["survivors"]["only_asrun_ids"] == ["asrun_only"]
    assert summary["survivors"]["only_ours_ids"] == ["ours_only"]
    assert summary["decision_agreement"] == 0.0
    assert summary["terminal_abs_delta"]["n"] == 2  # documents both sides scored with the terminal model
    assert summary["pass"] == {"decision_agreement": False, "flips_within_epsilon": False}
