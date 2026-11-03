# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The single-host runner: every stage's tasks on a local process pool, then the stage's seal.

Results land in the same ``stages/{name}/done/{task}.json`` and ``sealed.json`` marks the claims runner writes, so
an interrupted run resumes where it stopped (finished tasks are skipped) and either runner can continue a run the
other started. Tasks that use a process pool of their own (pass1, c075_text, apply_c075) run one at a time; the
others run ``StageSpec.slots(cpus, memory_gb)`` at a time.
"""

import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor

from atlantic_cc.dedup.plan import load_plan
from atlantic_cc.dedup.stages import Pipeline, StageSpec
from atlantic_cc.dedup.tasks import worker_identity
from atlantic_cc.io.resolve import store_for
from atlantic_cc.io.store import PreconditionFailed, Store, dumps, read_json

logger = logging.getLogger(__name__)

_PIPELINES: dict[tuple[str, int, bool], Pipeline] = {}


def _pipeline(run_root: str, procs: int, keep_intermediates: bool) -> Pipeline:
    """One pipeline per process and run: it caches the plan, the inputs and sealed results, which never change."""
    key = (run_root, procs, keep_intermediates)
    if key not in _PIPELINES:
        store = store_for(run_root)
        plan = load_plan(store, run_root)
        _PIPELINES[key] = Pipeline(store, plan, procs=procs, keep_intermediates=keep_intermediates)
    return _PIPELINES[key]


def _done_path(pipeline: Pipeline, stage: str, task: int) -> str:
    return f"{pipeline.ctx.stage_dir(stage)}/done/{task:06d}.json"


def _complete(store: Store, path: str, payload: dict) -> None:
    """Write a done mark once; a mark already there (from an earlier attempt or another runner) is kept."""
    try:
        store.create(path, dumps(payload))
    except PreconditionFailed:
        logger.warning("%s already exists; keeping it", path)


def _run_task(run_root: str, procs: int, keep_intermediates: bool, stage: str, task: int) -> int:
    """Run one task of the frontier stage ``stage`` and record its done mark (runs in a pool process)."""
    pipeline = _pipeline(run_root, procs, keep_intermediates)
    spec = pipeline.frontier()
    if spec is None or spec.name != stage:
        raise RuntimeError(f"frontier is {spec.name if spec else None}, not {stage}")
    started = time.time()
    result = spec.run(pipeline.ctx, task)
    payload = {
        **result,
        "_started": started,
        "_worker": f"{worker_identity()}/local",
        "_task": f"{task:06d}",
        "_attempt": 1,
        "_finished": time.time(),
    }
    _complete(pipeline.store, _done_path(pipeline, stage, task), payload)
    return task


def _pending(pipeline: Pipeline, spec: StageSpec) -> list[int]:
    done = {
        os.path.basename(info.path).removesuffix(".json")
        for info in pipeline.store.list(f"{pipeline.ctx.stage_dir(spec.name)}/done/")
        if info.path.endswith(".json")
    }
    return [t for t in range(spec.n_tasks) if f"{t:06d}" not in done]


def _run_stage(pipeline: Pipeline, spec: StageSpec, run_root: str, cpus: int, memory_gb: float) -> None:
    pending = _pending(pipeline, spec)
    slots = spec.slots(cpus, memory_gb)
    logger.info("stage %s: %d of %d tasks to run, %d at a time", spec.name, len(pending), spec.n_tasks, slots)
    started = time.time()
    if spec.procs > 1 or slots == 1 or len(pending) <= 1:
        for task in pending:
            _run_task(run_root, cpus, pipeline.ctx.keep_intermediates, spec.name, task)
    elif pending:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(slots, len(pending)), mp_context=context) as pool:
            futures = [
                pool.submit(_run_task, run_root, cpus, pipeline.ctx.keep_intermediates, spec.name, t) for t in pending
            ]
            for future in futures:
                future.result()
    results = [read_json(pipeline.store, _done_path(pipeline, spec.name, t)) for t in range(spec.n_tasks)]
    seal_started = time.time()
    sealed = spec.seal(pipeline.ctx, results)
    payload = {
        **sealed,
        "_started": seal_started,
        "_worker": f"{worker_identity()}/local",
        "_task": "seal",
        "_attempt": 1,
        "_finished": time.time(),
    }
    _complete(pipeline.store, pipeline.sealed_path(spec.name), payload)
    logger.info("stage %s sealed in %.0fs", spec.name, time.time() - started)


def run_local(run_root: str, cpus: int, memory_gb: float, keep_intermediates: bool = False) -> None:
    """Run every remaining stage of the run at ``run_root`` on this host."""
    pipeline = _pipeline(run_root, cpus, keep_intermediates)
    while (spec := pipeline.frontier()) is not None:
        _run_stage(pipeline, spec, run_root, cpus, memory_gb)
    logger.info("every stage of %s is sealed", run_root)
