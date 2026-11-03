# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The multi-host claims runner: a stateless worker that claims a task of the frontier stage, runs it, repeats.

A port of the as-run ``rdedup/worker.py``. The worker process supervises ``cpus`` slot processes. Each slot
independently finds the frontier stage and works on it only if its index is below the number of concurrent tasks
that stage allows on this worker's shape (a pass-1 task uses every core through its own process pool, while a numpy
reduce uses one). A slot that dies -- lost claim, deadline, OOM kill -- is restarted; its claim goes stale and
another worker takes the task over. Workers never talk to each other: start any number of them, on any hosts, against
one run root on a store with conditional writes (a GCS bucket or a shared POSIX filesystem). When every stage is
sealed, slots exit 0 and so does the worker.
"""

import logging
import multiprocessing
import random
import sys
import time
import traceback

from atlantic_cc.dedup.plan import load_plan
from atlantic_cc.dedup.stages import Pipeline, StageSpec
from atlantic_cc.dedup.tasks import SEAL, ClaimConfig, Heartbeat, TaskBoard, worker_identity
from atlantic_cc.io.resolve import store_for
from atlantic_cc.io.store import Store

logger = logging.getLogger(__name__)

IDLE_POLL_SECONDS = 60.0
FAILURE_BACKOFF_SECONDS = (10.0, 30.0)
CLAIM_TRIES = 8
RESTART_DELAY_SECONDS = 20.0
_SUPERVISE_POLL_SECONDS = 5.0


def _record_failure(store: Store, board: TaskBoard, task: str, attempt: int, worker: str) -> None:
    path = f"{board.stage_dir}/failures/{task}-{attempt:02d}-{int(time.time())}.txt"
    try:
        store.write(path, f"worker {worker}\n{traceback.format_exc()}".encode())
    except Exception:
        # The failure itself is already logged; a lost note must not mask it.
        logger.exception("could not record failure of %s", task)


def _run_claimed(store: Store, pipeline: Pipeline, spec: StageSpec, board: TaskBoard, task, worker: str) -> bool:
    """Claim ``task`` and run it; returns whether this slot got the claim."""
    claim = board.try_claim(task, worker)
    if claim is None:
        return False
    started = time.time()
    logger.info("%s: running %s/%s (attempt %d)", worker, spec.name, claim.task, claim.attempt)
    try:
        with Heartbeat(board, claim, worker, spec.deadline_hours * 3600):
            if task == SEAL:
                result = spec.seal(pipeline.ctx, board.done_results())
            else:
                result = spec.run(pipeline.ctx, task)
    except Exception:
        # A failed task leaves its claim to go stale; another attempt (here or elsewhere) retries it.
        logger.exception("%s: %s/%s failed; its claim will go stale and be retried", worker, spec.name, claim.task)
        _record_failure(store, board, claim.task, claim.attempt, worker)
        time.sleep(random.uniform(*FAILURE_BACKOFF_SECONDS))
        return True
    result = {**result, "_started": started, "_worker": worker}
    board.complete(claim, result)
    logger.info("%s: finished %s/%s in %.0fs", worker, spec.name, claim.task, time.time() - started)
    return True


def slot_loop(
    slot: int, run_root: str, cpus: int, memory_gb: float, claims: ClaimConfig, keep_intermediates: bool = False
) -> int:
    """Work on the run until every stage is sealed; returns the exit code."""
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s [slot{slot}] %(levelname)s %(name)s: %(message)s")
    store = store_for(run_root)
    plan = load_plan(store, run_root)
    pipeline = Pipeline(store, plan, procs=cpus, keep_intermediates=keep_intermediates)
    worker = f"{worker_identity()}/s{slot}"
    boards: dict[str, TaskBoard] = {}
    while True:
        spec = pipeline.frontier()
        if spec is None:
            logger.info("%s: every stage is sealed; exiting", worker)
            return 0
        if slot >= spec.slots(cpus, memory_gb):
            time.sleep(IDLE_POLL_SECONDS * random.uniform(0.5, 1.0))
            continue
        board = boards.setdefault(spec.name, TaskBoard(store, pipeline.ctx.stage_dir(spec.name), spec.n_tasks, claims))
        state = board.state()
        if not board.pending(state):
            if not _run_claimed(store, pipeline, spec, board, SEAL, worker):
                time.sleep(IDLE_POLL_SECONDS * random.uniform(0.5, 1.5))
            continue
        candidates = board.claimable(state)
        random.shuffle(candidates)
        if not any(_run_claimed(store, pipeline, spec, board, t, worker) for t in candidates[:CLAIM_TRIES]):
            time.sleep(IDLE_POLL_SECONDS * random.uniform(0.5, 1.5))


def _slot_entry(
    slot: int, run_root: str, cpus: int, memory_gb: float, claims: ClaimConfig, keep_intermediates: bool
) -> None:
    sys.exit(slot_loop(slot, run_root, cpus, memory_gb, claims, keep_intermediates))


def supervise(
    run_root: str,
    cpus: int,
    memory_gb: float,
    claims: ClaimConfig,
    keep_intermediates: bool = False,
    restart_delay: float = RESTART_DELAY_SECONDS,
) -> int:
    """Run ``cpus`` slots until every one reports the pipeline complete."""
    context = multiprocessing.get_context("spawn")

    def start(slot: int) -> multiprocessing.process.BaseProcess:
        process = context.Process(
            target=_slot_entry,
            args=(slot, run_root, cpus, memory_gb, claims, keep_intermediates),
            name=f"slot{slot}",
        )
        process.start()
        return process

    slots = {slot: start(slot) for slot in range(cpus)}
    finished: set[int] = set()
    while len(finished) < cpus:
        time.sleep(_SUPERVISE_POLL_SECONDS)
        for slot, process in list(slots.items()):
            if slot in finished or process.is_alive():
                continue
            if process.exitcode == 0:
                finished.add(slot)
                continue
            logger.warning("slot %d exited with %s; restarting in %.0fs", slot, process.exitcode, restart_delay)
            time.sleep(restart_delay)
            slots[slot] = start(slot)
    logger.info("all slots finished: pipeline complete")
    return 0
