# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Count the c075 removals of a finished run that change under another ``np.argpartition`` tie order (maintainers).

Re-solves every c075 partition of a run whose ``c075_text`` was kept (``dedup run|worker --keep-intermediates``)
under this process's NumPy dispatch, with the production solve code, and compares the removed doc keys with the
run's ``c075_marks``. Only solve batches of more than 256 members can differ. On an x86_64 host,
``NPY_DISABLE_CPU_FEATURES`` picks the kernel: masking AVX2 (and any AVX-512 features) leaves NumPy's generic
selection, which keeps the same tied items as ARM hosts. Reads the run root only::

    NPY_DISABLE_CPU_FEATURES=AVX2 python -m tests.parity.dedup.tie_measure --run-root RUN --procs 8 --report r.json
"""

import argparse
import dataclasses
import json
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from atlantic_cc.dedup.plan import load_plan
from atlantic_cc.dedup.runtime import runtime_info
from atlantic_cc.dedup.solver import SolveStats
from atlantic_cc.dedup.stages import Pipeline, c075_marks_of_unit, solve_c075_partition
from atlantic_cc.io.resolve import store_for

logger = logging.getLogger(__name__)

_EXAMPLES = 20
_PIPELINES: dict[str, Pipeline] = {}


def _pipeline(run_root: str) -> Pipeline:
    if run_root not in _PIPELINES:
        store = store_for(run_root)
        _PIPELINES[run_root] = Pipeline(store, load_plan(store, run_root), procs=1, keep_intermediates=True)
    return _PIPELINES[run_root]


def _solve(run_root: str, partition: int) -> tuple[np.ndarray, dict, float]:
    started = time.monotonic()
    removed, stats, _passes = solve_c075_partition(_pipeline(run_root).ctx, partition)
    logger.info("partition %d: %d removals in %.0fs", partition, len(removed), time.monotonic() - started)
    return removed, dataclasses.asdict(stats), time.monotonic() - started


def measure(run_root: str, procs: int) -> dict:
    """Removals under this process's tie order against the run's marks."""
    started = time.time()
    pipeline = _pipeline(run_root)
    plan = pipeline.ctx.plan
    release = np.sort(np.concatenate([c075_marks_of_unit(pipeline.ctx, u) for u in range(plan.n_units)]))
    partitions = range(plan.c075_partitions)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=procs, mp_context=context) as pool:
        results = list(pool.map(_solve, [run_root] * len(partitions), partitions))
    here = np.sort(np.concatenate([removed for removed, _, _ in results]))
    only_here, only_release = np.setdiff1d(here, release), np.setdiff1d(release, here)
    # A run's marks are per unit, not per c075 partition, so only extra removals can be attributed to a partition.
    with_extra = sum(1 for removed, _, _ in results if len(np.setdiff1d(removed, release)))
    stats = {f.name: sum(r[1][f.name] for r in results) for f in dataclasses.fields(SolveStats)}
    info = runtime_info()
    return {
        "run_root": run_root,
        "tie_order_platform": info["tie_order_platform"],
        "NPY_DISABLE_CPU_FEATURES": info["NPY_DISABLE_CPU_FEATURES"],
        "numpy_cpu_enabled": info["numpy_cpu_enabled"],
        "cpu_model": next(
            (line.split(":", 1)[1].strip() for line in info["lscpu"].splitlines() if line.startswith("Model name:")),
            info["machine"],
        ),
        "run_removed": len(release),
        "removed": len(here),
        "only_here": len(only_here),
        "only_run": len(only_release),
        "only_here_examples": [int(k) for k in only_here[:_EXAMPLES]],
        "only_run_examples": [int(k) for k in only_release[:_EXAMPLES]],
        "partitions": plan.c075_partitions,
        "partitions_with_extra_removals": with_extra,
        "solve_stats": stats,
        "solve_cpu_seconds": round(sum(r[2] for r in results), 1),
        "elapsed_seconds": round(time.time() - started, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--procs", type=int, required=True)
    parser.add_argument("--report", help="Also write the JSON report here (local path or URL).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    report = measure(args.run_root, args.procs)
    text = json.dumps(report, indent=1, sort_keys=True)
    print(text)
    if args.report:
        store_for(args.report).write(args.report, text.encode())


if __name__ == "__main__":
    main()
