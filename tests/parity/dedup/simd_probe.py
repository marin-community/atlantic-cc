# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure the c075 solver's SIMD tie-order dependence on this host (maintainers; meant for x86_64 hosts).

Runs the argpartition tie probe and the index-path solver clusters of ``tests/golden/dedup/asrun_goldens.json``
in child processes: with NumPy's native dispatch, with AVX-512 masked through ``NPY_DISABLE_CPU_FEATURES`` (what
``atlantic-cc dedup --simd mask-avx512`` does), and with further features masked (``--extra-mask AVX2`` leaves
NumPy's generic, non-SIMD selection). Each result is compared with the goldens of every recorded platform;
``x86_64-avx2`` is the tie order the release was built with. Reads no data::

    python -m tests.parity.dedup.simd_probe [--extra-mask AVX2] [--report probe.json]
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from atlantic_cc.dedup import solver
from atlantic_cc.dedup.runtime import NUMPY_DISABLE_ENV, enabled_avx512_features, runtime_info, tie_order_platform
from atlantic_cc.io.resolve import store_for
from tests.unit.dedup import synthetic
from tests.unit.dedup.test_u_dedup_tie_order import tie_probe

GOLDENS = json.loads((Path(__file__).parents[2] / "golden" / "dedup" / "asrun_goldens.json").read_text())
RELEASE_PLATFORM = "x86_64-avx2"
_DOCUMENT_CAP = 400  # as in the index-path goldens (generate_asrun.py)
_BIG_CLUSTER = 300
_CHILD_FLAG = "--child"


def measure() -> dict:
    """Tie probe and index-path removals under this process's NumPy dispatch."""
    removed = {}
    for cap in GOLDENS["solver_index_path"][RELEASE_PLATFORM]:
        solver.SOLVE_MAX_CLUSTER_CHARS = int(cap)
        solver.SOLVE_MAX_DOCUMENT_CHARS = _DOCUMENT_CAP
        rows = synthetic.solve_rows(int(cap), big_cluster=_BIG_CLUSTER)
        id_of = {r["d"]: r["id"] for r in rows}
        removed[cap] = sorted(id_of[d] for d in solver.solve_sorted_rows([dict(r) for r in rows], solver.SolveStats()))
    return {
        "tie_order_platform": tie_order_platform(),
        "avx512_enabled": enabled_avx512_features(),
        NUMPY_DISABLE_ENV: os.environ.get(NUMPY_DISABLE_ENV, ""),
        "tie_probe": tie_probe(),
        "removed_ids": removed,
    }


def compare(result: dict) -> dict:
    """How ``result`` differs from the goldens of every recorded platform."""
    out = {}
    for tag, goldens in GOLDENS["solver_index_path"].items():
        out[tag] = {"tie_probe_equal": result["tie_probe"] == GOLDENS["tie_probe"].get(tag)}
        for cap, ids in result["removed_ids"].items():
            golden = set(goldens[cap]["removed_ids"])
            out[tag][f"cap_{cap}"] = {
                "removed": len(ids),
                "golden_removed": len(golden),
                "only_here": len(set(ids) - golden),
                "only_golden": len(golden - set(ids)),
            }
    return out


def _child(env: dict[str, str]) -> dict:
    root = Path(__file__).parents[3]
    done = subprocess.run(
        [sys.executable, "-m", "tests.parity.dedup.simd_probe", _CHILD_FLAG],
        env=env,
        cwd=root,
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        raise RuntimeError(f"probe child failed:\n{done.stderr}")
    return json.loads(done.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extra-mask", action="append", default=[], help="Also mask these NumPy CPU features.")
    parser.add_argument("--report", help="Also write the JSON report here (local path or URL).")
    parser.add_argument(_CHILD_FLAG, action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        print(json.dumps(measure()))
        return
    environment = {k: v for k, v in os.environ.items() if k != NUMPY_DISABLE_ENV}
    native = _child(environment)
    masks = {}
    if native["avx512_enabled"]:
        masks["mask_avx512"] = native["avx512_enabled"]
    if args.extra_mask:
        masks["mask_" + "_".join(f.lower() for f in args.extra_mask)] = native["avx512_enabled"] + args.extra_mask
    results = {"native": native}
    results.update({name: _child({**environment, NUMPY_DISABLE_ENV: " ".join(m)}) for name, m in masks.items()})
    info = runtime_info()
    report = {
        "host": {k: v for k, v in info.items() if k not in ("numpy_show_runtime", "lscpu")},
        "cpu_model": next(
            (line.split(":", 1)[1].strip() for line in info["lscpu"].splitlines() if line.startswith("Model name:")),
            platform.processor(),
        ),
        "variants": {name: {**result, "vs_goldens": compare(result)} for name, result in results.items()},
    }
    text = json.dumps(report, indent=1, sort_keys=True)
    print(text)
    if args.report:
        store_for(args.report).write(args.report, text.encode())


if __name__ == "__main__":
    main()
