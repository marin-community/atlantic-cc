# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare one WARC's classification outputs with production's (the R2/R3 per-WARC report).

Both sides are read from output files only. The kept table carries exact float32 scores and the
tombstones six-decimal ones, and together they give every fastText survivor's route:

* kept, ``modernbert_prob`` NaN: accepted by the pooled model;
* kept, with a terminal score: band document the terminal model kept;
* tombstone, ``modernbert_prob`` null: dropped by the pooled model;
* tombstone, with a terminal score: band document the terminal model rejected.

Gates: decisions agree on at least 99.9% of the survivors both sides
have, and every flip lies within 0.02 of the pooled threshold or 0.05 of the terminal threshold that
decided it.
"""

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pyarrow as pa

from atlantic_cc.spec import POOLED_ACCEPT_AT, POOLED_DROP_BELOW, TERMINAL_KEEP_AT

MIN_DECISION_AGREEMENT = 0.999
POOLED_FLIP_EPSILON = 0.02
TERMINAL_FLIP_EPSILON = 0.05
MAX_LISTED_IDS = 20
PERCENTILES = {"p50": 50, "p90": 90, "p99": 99, "p99.9": 99.9}


class Route(StrEnum):
    ACCEPT = "accept"
    BAND_KEEP = "band_keep"
    BAND_REJECT = "band_reject"
    DROP = "drop"


KEPT_ROUTES = (Route.ACCEPT, Route.BAND_KEEP)
BAND_ROUTES = (Route.BAND_KEEP, Route.BAND_REJECT)


@dataclass(frozen=True)
class Survivor:
    route: Route
    pooled: float
    terminal: float  # NaN unless the terminal model scored the document
    exact: bool  # scores are float32 (kept row), not six-decimal (tombstone)


def survivors_from_outputs(kept: pa.Table, tombstones: list[dict]) -> dict[str, Survivor]:
    """Every fastText survivor of a WARC by doc_id, in WARC order within kept and within tombstones."""
    out: dict[str, Survivor] = {}
    columns = (kept.column(c).to_pylist() for c in ("doc_id", "pooled_prob", "modernbert_prob"))
    for doc_id, pooled, terminal in zip(*columns, strict=True):
        route = Route.ACCEPT if math.isnan(terminal) else Route.BAND_KEEP
        out[doc_id] = Survivor(route, pooled, terminal, exact=True)
    for row in tombstones:
        terminal = row["modernbert_prob"]
        route = Route.DROP if terminal is None else Route.BAND_REJECT
        out[row["doc_id"]] = Survivor(route, row["pooled_prob"], math.nan if terminal is None else terminal, False)
    if len(out) != kept.num_rows + len(tombstones):
        raise ValueError("duplicate doc_ids across kept and tombstones")
    return out


def distribution(values: list[float]) -> dict:
    """Summary of absolute differences: count, how many are exactly 0, mean, percentiles, max."""
    if not values:
        return {"n": 0}
    x = np.asarray(values, dtype=np.float64)
    summary = {"n": int(x.size), "n_zero": int((x == 0).sum()), "mean": float(x.mean())}
    summary.update({name: float(np.percentile(x, q)) for name, q in PERCENTILES.items()})
    summary["max"] = float(x.max())
    return summary


def _deciding_threshold(asrun: Survivor, ours: Survivor) -> tuple[str, float, float]:
    """``(stage, threshold, epsilon)`` of the comparison that split the two decisions."""
    if asrun.route in BAND_ROUTES and ours.route in BAND_ROUTES:
        return "terminal", float(TERMINAL_KEEP_AT), TERMINAL_FLIP_EPSILON
    lo, hi = float(POOLED_DROP_BELOW), float(POOLED_ACCEPT_AT)
    threshold = lo if abs(asrun.pooled - lo) <= abs(asrun.pooled - hi) else hi
    return "pooled", threshold, POOLED_FLIP_EPSILON


@dataclass(frozen=True)
class WarcComparison:
    pooled_deltas: list[float]
    terminal_deltas: list[float]
    n_common: int
    flips: list[dict]
    summary: dict


def compare_survivors(ours: dict[str, Survivor], asrun: dict[str, Survivor]) -> WarcComparison:
    common = [d for d in asrun if d in ours]
    pooled_deltas = [abs(ours[d].pooled - asrun[d].pooled) for d in common]
    terminal_deltas = [
        abs(ours[d].terminal - asrun[d].terminal)
        for d in common
        if not (math.isnan(ours[d].terminal) or math.isnan(asrun[d].terminal))
    ]
    routes = {a.value: {o.value: 0 for o in Route} for a in Route}
    flips = []
    for d in common:
        a, o = asrun[d], ours[d]
        routes[a.route][o.route] += 1
        if (a.route in KEPT_ROUTES) == (o.route in KEPT_ROUTES):
            continue
        stage, threshold, epsilon = _deciding_threshold(a, o)
        score_a, score_o = (a.terminal, o.terminal) if stage == "terminal" else (a.pooled, o.pooled)
        flips.append(
            {
                "doc_id": d,
                "stage": stage,
                "threshold": threshold,
                "asrun": {"route": a.route.value, "pooled": a.pooled, "terminal": _json_float(a.terminal)},
                "ours": {"route": o.route.value, "pooled": o.pooled, "terminal": _json_float(o.terminal)},
                "asrun_distance": abs(score_a - threshold),
                "ours_distance": abs(score_o - threshold),
                "within_epsilon": abs(score_a - threshold) <= epsilon,
            }
        )
    only_ours = [d for d in ours if d not in asrun]
    only_asrun = [d for d in asrun if d not in ours]
    agreement = (len(common) - len(flips)) / len(common) if common else 1.0
    summary = {
        "survivors": {
            "ours": len(ours),
            "asrun": len(asrun),
            "common": len(common),
            "only_ours": len(only_ours),
            "only_asrun": len(only_asrun),
            "only_ours_ids": only_ours[:MAX_LISTED_IDS],
            "only_asrun_ids": only_asrun[:MAX_LISTED_IDS],
        },
        "kept": {
            "ours": sum(s.route in KEPT_ROUTES for s in ours.values()),
            "asrun": sum(s.route in KEPT_ROUTES for s in asrun.values()),
        },
        "pooled_abs_delta": distribution(pooled_deltas),
        "terminal_abs_delta": distribution(terminal_deltas),
        "routes_asrun_to_ours": routes,
        "decision_agreement": agreement,
        "n_flips": len(flips),
        "flips": flips,
        "pass": {
            "decision_agreement": agreement >= MIN_DECISION_AGREEMENT,
            "flips_within_epsilon": all(f["within_epsilon"] for f in flips),
        },
    }
    return WarcComparison(pooled_deltas, terminal_deltas, len(common), flips, summary)


def aggregate(comparisons: list[WarcComparison]) -> dict:
    """Totals over several WARCs, with the same distributions and gates as one WARC."""
    n_common = sum(c.n_common for c in comparisons)
    n_flips = sum(len(c.flips) for c in comparisons)
    agreement = (n_common - n_flips) / n_common if n_common else 1.0
    return {
        "n_warcs": len(comparisons),
        "n_common_survivors": n_common,
        "pooled_abs_delta": distribution([x for c in comparisons for x in c.pooled_deltas]),
        "terminal_abs_delta": distribution([x for c in comparisons for x in c.terminal_deltas]),
        "decision_agreement": agreement,
        "n_flips": n_flips,
        "pass": {
            "decision_agreement": agreement >= MIN_DECISION_AGREEMENT,
            "flips_within_epsilon": all(f["within_epsilon"] for c in comparisons for f in c.flips),
        },
    }


def _json_float(x: float) -> float | None:
    return None if math.isnan(x) else x
