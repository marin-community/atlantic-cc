# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""U-DEDUP rule 7 (layout): unit packing and partition formulas reproduce the as-run 400m-1x plan."""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from atlantic_cc.dedup.plan import Plan, build_plan, inputs_sha256, pack_units
from atlantic_cc.dedup.text import doc_keys

GOLDEN = Path(__file__).parents[2] / "golden" / "dedup"


def test_build_plan_reproduces_the_as_run_400m_1x_plan():
    inputs = pq.read_table(GOLDEN / "inputs_400m_1x.parquet")
    paths, sizes = inputs.column("path").to_pylist(), inputs.column("size").to_pylist()
    asrun = json.loads((GOLDEN / "plan_400m_1x.json").read_text())
    plan = build_plan(
        run_root=asrun["run_root"],
        output_root=asrun["output_root"],
        spec_id=asrun["spec_id"],
        sizes=sizes,
        unit_bytes=int(8.0 * 1e9),
    )
    assert json.loads(plan.to_json()) == asrun
    assert inputs_sha256(paths) == "f34694f7fbf4ef4ce09c7d48af2edc5581ea45d0cdb1779745a9d783c60ae3e1"
    assert sum(sizes[: plan.unit_starts[1]]) == 7_993_776_741  # unit 0 = the T1 parity tier


def test_units_pack_files_in_order_up_to_the_byte_budget():
    assert pack_units([5, 5, 5, 20, 1, 1], 10) == [0, 2, 3, 4, 6]
    assert pack_units([], 10) == [0, 0]


def test_keys_map_to_units_and_node_partitions():
    plan = Plan(
        run_root="r",
        output_root="o",
        spec_id="s",
        n_files=10,
        unit_starts=[0, 3, 7, 10],
        exact_partitions=1,
        lsh_partitions=1,
        units_per_node_partition=2,
        c075_partitions=1,
        files_per_output=32,
        rules=["c075"],
    )
    keys = np.concatenate([doc_keys(f, np.array([0, 5])) for f in (0, 2, 3, 6, 7, 9)])
    assert plan.unit_of_keys(keys).tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    assert plan.node_partition_of_keys(keys).tolist() == [0] * 8 + [1] * 4
    assert list(plan.node_partition_units(1)) == [2]


def test_plans_for_other_rules_are_refused():
    with pytest.raises(ValueError, match="c075"):
        Plan("r", "o", "s", 1, [0, 1], 1, 1, 1, 1, 32, ["cc_canonical", "c075"])
