# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The per-WARC worker: phase A on CPU subprocesses feeding phase B on the accelerator through RAM.

A port of production's fused worker (``experiments/fast_curation/fused_phase.py``) without its
cluster machinery. Each of ``phase_a_procs`` spawned subprocesses loads the fastText model and its
own crash-isolating extractor pool once, then runs whole WARCs. The main process keeps at most
``queue_depth`` WARCs in flight and runs phase B on each finished WARC in completion order, so the
CPUs extract while the accelerator scores. Outputs are per WARC, so the processing order never
changes them, and rows within a WARC stay in WARC record order.

Resumption and failure handling follow production:

* a WARC is done when ``markers/data-{warc_hash}.json`` exists; reruns skip it;
* every start of a WARC is counted in ``attempts/`` before its work begins, and a WARC started more
  than ``MAX_WARC_ATTEMPTS`` times is recorded as abandoned with no documents (production
  abandoned 101 WARCs that hung or killed their workers);
* a WARC that cannot be downloaded is recorded as failed with no documents;
* any other phase-A error leaves the WARC unmarked; the run continues and reports it, so a rerun
  retries it until the attempt cap;
* if no WARC finishes phase A for ``stall_seconds`` (production: 25 minutes), the worker raises
  :class:`WorkerStalledError`; the CLI then exits hard so a supervisor can restart it.

Production ran in one fixed environment; a user's setup can be broken, and that must never cost a
WARC its starts. So before counting any start, one task proves that a phase-A subprocess loads the
fastText model and passes the extractor's smoke extraction. When phase A cannot start, or phase B
raises (CUDA out of memory, a model error), the worker stops with :class:`WorkerEnvironmentError`.
Whenever the run stops on an error other than a stall or a dead phase-A subprocess (the two ways a
WARC can take its worker down), the starts it counted for WARCs it did not finish are taken back.
"""

import contextlib
import io
import logging
import multiprocessing as mp
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from multiprocessing.util import Finalize
from typing import Protocol

import pyarrow as pa

from atlantic_cc.classify.fasttext_gate import FastTextGate
from atlantic_cc.extract.pool import ExtractorPool, open_extractor_pool
from atlantic_cc.io.store import Store
from atlantic_cc.pipeline.outputs import (
    PhaseBCounters,
    WarcMarker,
    WarcStatus,
    is_done,
    record_attempt,
    release_attempt,
    write_marker,
    write_warc_outputs,
)
from atlantic_cc.pipeline.phase_a import PhaseACounters, run_phase_a
from atlantic_cc.schemas import PRESURVIVOR_SCHEMA
from atlantic_cc.spec import CC_HTTPS_PREFIX, CC_S3_PREFIX
from atlantic_cc.warc.identity import warc_hash
from atlantic_cc.worklist.frozen import read_paths_file
from atlantic_cc.worklist.identity import hash_shard, to_s3_url

logger = logging.getLogger(__name__)

MAX_WARC_ATTEMPTS = 3  # as-run fused_phase.MAX_WARC_ATTEMPTS
STALL_MINUTES = 25.0  # as-run fused_phase.STALL_EXIT_MINUTES
# multiprocessing.queues.Queue stops its feeder thread in an exit finalizer of priority 10. A phase-A
# runner's pools must shut down, waiting for their children, before that: otherwise the children's
# stop sentinels are never sent and the exiting process hangs joining them.
RUNNER_CLOSE_EXIT_PRIORITY = 100

# Presurvivors -> (kept, tombstones, counters): ``functools.partial(phase_b.run_phase_b, tokenize=..., ...)``.
Classify = Callable[[pa.Table], tuple[pa.Table, list[dict], PhaseBCounters]]


class PhaseARunner(Protocol):
    """Phase A inside one worker subprocess."""

    def run(self, warc_path: str) -> tuple[pa.Table, PhaseACounters]: ...

    def close(self) -> None:
        """Release what the runner holds (its extraction processes); called when the subprocess exits."""
        ...


class PhaseAFactory(Protocol):
    """Builds a runner inside each worker subprocess; must be picklable (it is sent to spawned processes)."""

    def __call__(self) -> PhaseARunner: ...


@dataclass(frozen=True)
class CascadeRunner:
    gate: FastTextGate
    extractor_pool: ExtractorPool

    def run(self, warc_path: str) -> tuple[pa.Table, PhaseACounters]:
        return run_phase_a(warc_path, self.gate, self.extractor_pool)

    def close(self) -> None:
        self.extractor_pool.close()


@dataclass(frozen=True)
class CascadePhaseA:
    """The production phase A: fastText gate and resiliparse-rs, loaded once per subprocess.

    Args:
        fasttext_model: Local path or fsspec URL of the fastText ``model.bin``.
        extractor_pkg_dir: Package directory of an installed resiliparse-rs artifact, or None for a
            regular install (see ``atlantic_cc.extract.install``).
        extract_procs: Extraction processes per phase-A subprocess (production: 6).
    """

    fasttext_model: str
    extractor_pkg_dir: str | None
    extract_procs: int

    def __call__(self) -> CascadeRunner:
        with contextlib.redirect_stderr(io.StringIO()):  # fastText prints a deprecation notice on every load
            gate = FastTextGate.load(self.fasttext_model)
        return CascadeRunner(gate, open_extractor_pool(self.extract_procs, self.extractor_pkg_dir))


@dataclass(frozen=True)
class PhaseB:
    """Phase B as the main process runs it, plus the device it runs on (recorded in every marker).

    ``classify`` is injected rather than imported so that the phase-A subprocesses, which import this
    module, never load the tokenizer and model stacks.
    """

    classify: Classify
    backend: str
    device: str
    n_devices: int


@dataclass(frozen=True)
class WorkerReport:
    n_warcs: int  # in this worker's list
    n_already_done: int
    n_done: int  # marked in this run, any status
    n_download_failed: int
    n_abandoned: int
    n_errors: int  # phase A raised; unmarked, retried by the next run

    @property
    def n_failed(self) -> int:
        """WARCs this run abandoned, could not download, or saw raise in phase A."""
        return self.n_abandoned + self.n_download_failed + self.n_errors


class WorkerStalledError(RuntimeError):
    """No WARC finished phase A within the stall window; a phase-A subprocess is probably hung."""


class WorkerEnvironmentError(RuntimeError):
    """The run stopped for a reason outside any WARC: phase A could not start, or phase B raised."""


_RUNNER: PhaseARunner | None = None  # set once per subprocess by _init_phase_a
_STARTUP_ERROR: Exception | None = None  # set instead when the runner could not be built


def _init_phase_a(factory: PhaseAFactory) -> None:
    global _RUNNER, _STARTUP_ERROR
    try:
        _RUNNER = factory()
    except Exception as e:  # raised by this subprocess's tasks: a failing initializer only breaks the pool
        _STARTUP_ERROR = e
        return
    # A multiprocessing child joins its own child processes while it exits, before the executor
    # machinery would stop them, so the runner's extraction pool is closed by an exit finalizer.
    Finalize(_RUNNER, _RUNNER.close, exitpriority=RUNNER_CLOSE_EXIT_PRIORITY)


def _runner() -> PhaseARunner:
    if _STARTUP_ERROR is not None:
        raise WorkerEnvironmentError(
            f"phase A could not start ({type(_STARTUP_ERROR).__name__}: {_STARTUP_ERROR}); check the fastText "
            "model and the resiliparse-rs install (`atlantic-cc extractor-check`)"
        ) from _STARTUP_ERROR
    return _RUNNER


def _check_phase_a() -> None:
    """A run's first task: it returns only if this subprocess built its phase-A runner."""
    _runner()


def _phase_a_task(warc_path: str) -> tuple[pa.Table, PhaseACounters, float]:
    runner = _runner()
    start = time.monotonic()
    table, counters = runner.run(warc_path)
    return table, counters, time.monotonic() - start


def _wait_for_phase_a(executor: ProcessPoolExecutor, timeout: float) -> None:
    """Raise unless a phase-A subprocess starts and builds its runner within ``timeout`` seconds."""
    ready = executor.submit(_check_phase_a)
    if not wait([ready], timeout=timeout).done:
        raise WorkerStalledError(f"phase A did not start in {timeout:.0f}s")
    try:
        ready.result()
    except BrokenProcessPool as e:
        raise WorkerEnvironmentError("phase A could not start: its subprocess died while starting") from e


def as_s3_url(path: str) -> str:
    """A worklist line (bucket-relative, ``s3://commoncrawl/`` or ``https://data.commoncrawl.org/``) as an S3 URL."""
    if path.startswith(CC_S3_PREFIX):
        return path
    return to_s3_url(path.removeprefix(CC_HTTPS_PREFIX))


def read_warc_list(path: str) -> list[str]:
    """The S3 URLs of a paths file (Common Crawl's ``warc.paths`` format, optionally gzipped)."""
    return [as_s3_url(line) for line in read_paths_file(path)]


def select_shard(warc_urls: list[str], shard_index: int, num_shards: int) -> list[str]:
    """Production's static partition, ``int(sha256(url)[:8], 16) % num_shards == shard_index``, sorted.

    With ``num_shards=32000`` a shard holds exactly production's base shard of the same index. A URL
    listed twice is kept once (the worker runs each WARC once).
    """
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard index {shard_index} outside [0, {num_shards})")
    return sorted({url for url in warc_urls if hash_shard(url, num_shards) == shard_index})


def run_worker(
    warc_urls: list[str],
    store: Store,
    root: str,
    phase_a: PhaseAFactory,
    phase_b: PhaseB,
    *,
    phase_a_procs: int,
    queue_depth: int,
    stall_seconds: float = STALL_MINUTES * 60,
) -> WorkerReport:
    """Extract and classify every WARC in ``warc_urls`` that has no marker under ``root``.

    Raises:
        WorkerEnvironmentError: Phase A could not start, or phase B raised; no WARC was charged a start.
        WorkerStalledError: Phase A did not start, or no WARC finished it, within ``stall_seconds``.
    """
    todo = [url for url in warc_urls if not is_done(store, root, warc_hash(url))]
    n_already_done = len(warc_urls) - len(todo)
    logger.info("%d WARCs, %d already done, %d to process", len(warc_urls), n_already_done, len(todo))
    counts = {status: 0 for status in WarcStatus}
    n_errors = 0
    unfinished: dict[str, int] = {}  # url -> the start this run counted, until the WARC is marked or raises

    def report() -> WorkerReport:
        return WorkerReport(
            n_warcs=len(warc_urls),
            n_already_done=n_already_done,
            n_done=sum(counts.values()),
            n_download_failed=counts[WarcStatus.DOWNLOAD_FAILED],
            n_abandoned=counts[WarcStatus.ABANDONED],
            n_errors=n_errors,
        )

    def finish(url: str, attempt: int, table: pa.Table, a: PhaseACounters, a_seconds: float, error: str) -> None:
        status = WarcStatus.ABANDONED if attempt > MAX_WARC_ATTEMPTS else WarcStatus.OK
        if a.download_failed:
            status, error = WarcStatus.DOWNLOAD_FAILED, a.download_error
        h = warc_hash(url)
        try:
            kept, tombstones, b = phase_b.classify(table)
        except Exception as e:
            raise WorkerEnvironmentError(f"phase B failed on {url}: {type(e).__name__}: {e}") from e
        start = time.monotonic()
        write_warc_outputs(store, root, h, kept, tombstones)
        marker = WarcMarker(
            warc_path=url,
            warc_hash=h,
            status=status,
            error=error,
            attempt=attempt,
            n_in=a.n_in,
            n_prefilter_dropped=a.n_prefilter_dropped,
            n_extract_empty=a.n_extract_empty,
            n_extract_crashed=a.n_extract_crashed,
            n_presurvivors=b.n_presurvivors,
            n_hi_accept=b.n_hi_accept,
            n_band=b.n_band,
            n_kept=b.n_kept,
            phase_a_seconds=round(a_seconds, 3),
            tokenize_seconds=round(b.tokenize_seconds, 3),
            pooled_seconds=round(b.pooled_seconds, 3),
            terminal_seconds=round(b.terminal_seconds, 3),
            write_seconds=round(time.monotonic() - start, 3),
            backend=phase_b.backend,
            device=phase_b.device,
            n_devices=phase_b.n_devices,
        )
        write_marker(store, root, marker)
        del unfinished[url]
        counts[status] += 1
        logger.info(
            "%s %s: %d in -> %d survivors -> %d accepted + %d band -> %d kept (A %.1fs, tokenize %.1fs, "
            "pooled %.1fs, terminal %.1fs)",
            h,
            status,
            a.n_in,
            b.n_presurvivors,
            b.n_hi_accept,
            b.n_band,
            b.n_kept,
            a_seconds,
            b.tokenize_seconds,
            b.pooled_seconds,
            b.terminal_seconds,
        )

    if not todo:
        return report()
    executor = ProcessPoolExecutor(
        max_workers=phase_a_procs, mp_context=mp.get_context("spawn"), initializer=_init_phase_a, initargs=(phase_a,)
    )
    pending: dict[Future, tuple[str, int]] = {}
    queue = iter(todo)

    def fill() -> None:
        while len(pending) < queue_depth:
            url = next(queue, None)
            if url is None:
                return
            attempt = record_attempt(store, root, warc_hash(url))
            unfinished[url] = attempt
            if attempt > MAX_WARC_ATTEMPTS:
                reason = f"abandoned: {attempt - 1} starts never completed"
                logger.error("%s %s; recording an explicit failure", url, reason)
                finish(url, attempt, PRESURVIVOR_SCHEMA.empty_table(), PhaseACounters(), 0.0, reason)
                continue
            pending[executor.submit(_phase_a_task, url)] = (url, attempt)

    try:
        _wait_for_phase_a(executor, stall_seconds)
        fill()
        while pending:
            finished, _ = wait(pending, timeout=stall_seconds, return_when=FIRST_COMPLETED)
            if not finished:
                in_flight = sorted(u for u, _ in pending.values())
                raise WorkerStalledError(f"no WARC finished phase A in {stall_seconds:.0f}s; in flight: {in_flight}")
            for future in finished:
                url, attempt = pending.pop(future)
                fill()  # keep the CPUs busy while the accelerator scores this WARC
                try:
                    table, a_counters, a_seconds = future.result()
                except (BrokenProcessPool, WorkerEnvironmentError):
                    raise
                except Exception:  # production retried the WARC; this start counts
                    logger.exception("phase A failed on %s (attempt %d); leaving it for a rerun", url, attempt)
                    n_errors += 1
                    del unfinished[url]
                    continue
                finish(url, attempt, table, a_counters, a_seconds, "")
    except BaseException as e:
        executor.shutdown(wait=False, cancel_futures=True)
        # A stall or a dead phase-A subprocess may be a WARC's doing (a hang, a crash), so as in production
        # those starts count; so do the starts of an interrupted or killed run.
        if isinstance(e, Exception) and not isinstance(e, (WorkerStalledError, BrokenProcessPool)):
            for url, attempt in unfinished.items():
                release_attempt(store, root, warc_hash(url), attempt)
            logger.info("took back the starts of %d unfinished WARCs", len(unfinished))
        raise
    executor.shutdown(wait=True)
    return report()
