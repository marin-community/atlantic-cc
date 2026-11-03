# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The per-WARC worker end to end: fake phase A in real subprocesses, real phase B with fake models.

In the fake WARCs, survivor i is pooled-accepted when i % 3 == 0, in the band when i % 3 == 1 (the
terminal model keeps it when i is even) and pooled-dropped when i % 3 == 2.
"""

import dataclasses
import functools
import gzip
import json
import multiprocessing as mp
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from atlantic_cc.io.local import LocalStore
from atlantic_cc.pipeline.cli import exit_if_warcs_failed
from atlantic_cc.pipeline.outputs import (
    ATTEMPTS_DIR,
    MARKERS_DIR,
    WarcStatus,
    attempts_path,
    kept_path,
    marker_path,
    read_marker,
    tombstones_path,
)
from atlantic_cc.pipeline.phase_b import run_phase_b
from atlantic_cc.pipeline.worker import (
    MAX_WARC_ATTEMPTS,
    PhaseB,
    WorkerEnvironmentError,
    WorkerStalledError,
    run_worker,
    select_shard,
)
from atlantic_cc.spec import CLS_TOKEN_ID, POOLED_MAX_TOKENS, SEP_TOKEN_ID, TERMINAL_MAX_TOKENS
from atlantic_cc.warc.identity import warc_hash
from tests.unit.pipeline.fake_phase_a import BlockingPhaseA, BrokenPhaseA, FakePhaseA, SlowStartPhaseA, url

DATA = Path(__file__).parent / "data"
REPO_ROOT = Path(__file__).parents[3]
# A healthy worker returns within seconds of its last WARC; a hung one never does.
SHUTDOWN_TIMEOUT = 120
# Runs one WARC through run_worker with a phase A that owns a real extraction pool, in its own
# process session so that a hang can be killed with all of its subprocesses.
POOL_WORKER_SCRIPT = """
import sys
from atlantic_cc.io.local import LocalStore
from atlantic_cc.pipeline.worker import run_worker
from tests.unit.pipeline.fake_phase_a import PoolPhaseA, url
from tests.unit.pipeline.test_worker import PHASE_B

root, log = sys.argv[1:3]
report = run_worker([url(3)], LocalStore(), root, PoolPhaseA(log), PHASE_B, phase_a_procs=1, queue_depth=1)
print(report.n_done)
"""
# Live for the whole session: a hung subprocess may only start after its test has returned.
_RELEASE_HUNG_PHASE_A = mp.get_context("spawn").Event()
_RELEASE_HUNG_STARTUP = mp.get_context("spawn").Event()
_DOC_RE = re.compile(r"doc (\d+)")
_TOKEN_BASE = 1000


def fake_tokenize(texts: list[str]) -> list[np.ndarray]:
    ids = [_TOKEN_BASE + int(_DOC_RE.fullmatch(t).group(1)) for t in texts]
    return [np.array([CLS_TOKEN_ID, i, SEP_TOKEN_ID], np.int32) for i in ids]


class ResidueScorer:
    """P(useful) of survivor i is ``probs[i % len(probs)]``."""

    def __init__(self, max_tokens: int, probs: list[float]) -> None:
        self.max_tokens = max_tokens
        self.probs = np.array(probs, np.float32)

    def score(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        doc = input_ids[:, 1] - _TOKEN_BASE
        return np.where(attention_mask[:, 0], self.probs[doc % len(self.probs)], 0).astype(np.float32)


PHASE_B = PhaseB(
    classify=functools.partial(
        run_phase_b,
        tokenize=fake_tokenize,
        pooled=ResidueScorer(POOLED_MAX_TOKENS, [0.95, 0.5, 0.01]),
        terminal=ResidueScorer(TERMINAL_MAX_TOKENS, [0.9, 0.1]),
    ),
    backend="fake",
    device="cpu",
    n_devices=1,
)


def phase_a_log(log: Path) -> tuple[list[str], set[str], set[str]]:
    """(WARCs started, pids that started one, pids that closed their runner) from the fake's log."""
    lines = [line.split() for line in log.read_text().splitlines()] if log.exists() else []
    started = [fields[2] for fields in lines if fields[0] == "start"]
    return started, {f[1] for f in lines if f[0] == "start"}, {f[1] for f in lines if f[0] == "close"}


def test_worker_writes_outputs_records_failures_and_resumes(tmp_path):
    root, store, log = str(tmp_path / "out"), LocalStore(), tmp_path / "phase_a.log"
    unfetchable, broken = url(7), url(8)
    urls = [url(5), url(0), unfetchable, broken]

    report = run_worker(
        urls,
        store,
        root,
        FakePhaseA(str(log), raises=frozenset({broken}), unfetchable=frozenset({unfetchable})),
        PHASE_B,
        phase_a_procs=2,
        queue_depth=3,
    )

    assert (report.n_done, report.n_download_failed, report.n_errors) == (3, 1, 1)
    # Every phase-A process released its runner (in production, its extraction pool) before exiting.
    _, running_pids, closed_pids = phase_a_log(log)
    assert running_pids and running_pids <= closed_pids
    h5 = warc_hash(url(5))
    assert pq.read_table(kept_path(root, h5)).column("doc_id").to_pylist() == ["w5-0", "w5-3", "w5-4"]
    tombstones = [json.loads(line) for line in gzip.open(tombstones_path(root, h5)).read().splitlines()]
    assert tombstones == [
        {"doc_id": "w5-1", "pooled_prob": 0.5, "modernbert_prob": 0.1},
        {"doc_id": "w5-2", "pooled_prob": 0.01, "modernbert_prob": None},
    ]
    marker = read_marker(store, root, h5)
    assert (marker.status, marker.attempt, marker.n_in, marker.n_extract_empty) == (WarcStatus.OK, 1, 7, 2)
    assert (marker.n_presurvivors, marker.n_hi_accept, marker.n_band, marker.n_kept) == (5, 2, 2, 3)
    assert (marker.backend, marker.device, marker.n_devices) == ("fake", "cpu", 1)
    # A WARC without survivors and an unfetchable one still get a zero-row kept file and a marker.
    for warc, status in ((url(0), WarcStatus.OK), (unfetchable, WarcStatus.DOWNLOAD_FAILED)):
        assert pq.read_table(kept_path(root, warc_hash(warc))).num_rows == 0
        assert not os.path.exists(tombstones_path(root, warc_hash(warc)))
        assert read_marker(store, root, warc_hash(warc)).status == status
    assert read_marker(store, root, warc_hash(unfetchable)).error == f"Failed to download {unfetchable}"
    assert store.stat(marker_path(root, warc_hash(broken))) is None

    # A rerun skips the marked WARCs and retries the one that raised.
    log.unlink()
    report = run_worker(urls, store, root, FakePhaseA(str(log)), PHASE_B, phase_a_procs=2, queue_depth=3)

    assert (report.n_already_done, report.n_done, report.n_errors) == (3, 1, 0)
    assert phase_a_log(log)[0] == [broken]
    marker = read_marker(store, root, warc_hash(broken))
    assert (marker.status, marker.attempt, marker.n_kept) == (WarcStatus.OK, 2, 4)


def test_warc_started_three_times_is_abandoned_without_running(tmp_path):
    root, store, log = str(tmp_path / "out"), LocalStore(), tmp_path / "phase_a.log"
    poison = url(3)
    store.write(attempts_path(root, warc_hash(poison)), b'{"attempts": 3}')

    report = run_worker([poison], store, root, FakePhaseA(str(log)), PHASE_B, phase_a_procs=1, queue_depth=1)

    assert report.n_abandoned == 1
    assert phase_a_log(log)[0] == []
    marker = read_marker(store, root, warc_hash(poison))
    assert (marker.status, marker.attempt, marker.n_kept) == (WarcStatus.ABANDONED, 4, 0)
    assert marker.error == "abandoned: 3 starts never completed"
    assert pq.read_table(kept_path(root, warc_hash(poison))).num_rows == 0


def test_hung_phase_a_stalls_the_worker(tmp_path):
    root, store = str(tmp_path / "out"), LocalStore()
    hung = BlockingPhaseA(_RELEASE_HUNG_PHASE_A)
    try:
        with pytest.raises(WorkerStalledError):
            run_worker([url(1)], store, root, hung, PHASE_B, phase_a_procs=1, queue_depth=1, stall_seconds=1)
    finally:
        _RELEASE_HUNG_PHASE_A.set()  # lets the stuck subprocess finish so its pool can shut down
    assert json.loads(store.read(attempts_path(root, warc_hash(url(1))))) == {"attempts": 1}
    assert store.stat(marker_path(root, warc_hash(url(1)))) is None


def test_a_setup_where_phase_a_cannot_start_never_charges_or_abandons_a_warc(tmp_path):
    # Regression: without the extractor every start failed, four runs abandoned the queued WARCs with
    # zero documents, and the fourth exited 0.
    root, store, log = str(tmp_path / "out"), LocalStore(), tmp_path / "phase_a.log"
    urls = [url(2), url(4)]

    for _ in range(MAX_WARC_ATTEMPTS + 1):
        with pytest.raises(WorkerEnvironmentError, match=r"phase A could not start \(RuntimeError: resiliparse-rs"):
            run_worker(urls, store, root, BrokenPhaseA(), PHASE_B, phase_a_procs=1, queue_depth=2)

    assert store.list(f"{root}/{ATTEMPTS_DIR}/") == []
    assert store.list(f"{root}/{MARKERS_DIR}/") == []
    report = run_worker(urls, store, root, FakePhaseA(str(log)), PHASE_B, phase_a_procs=1, queue_depth=2)
    assert (report.n_done, report.n_abandoned) == (2, 0)
    assert [read_marker(store, root, warc_hash(u)).attempt for u in urls] == [1, 1]


def test_phase_b_failure_stops_the_run_and_takes_back_the_unfinished_starts(tmp_path):
    root, store, log = str(tmp_path / "out"), LocalStore(), tmp_path / "phase_a.log"
    done, failing, queued = url(2), url(4), url(6)
    urls = [done, failing, queued]
    store.write(attempts_path(root, warc_hash(failing)), b'{"attempts": 2}')  # two earlier starts never finished

    def classify_out_of_memory_on_failing(table: pa.Table):
        if table.num_rows and table.column("doc_id")[0].as_py().startswith("w4-"):
            raise RuntimeError("CUDA out of memory")
        return PHASE_B.classify(table)

    out_of_memory = dataclasses.replace(PHASE_B, classify=classify_out_of_memory_on_failing)
    with pytest.raises(WorkerEnvironmentError, match=r"phase B failed on \S+/w4\.warc\.gz: RuntimeError: CUDA out"):
        run_worker(urls, store, root, FakePhaseA(str(log)), out_of_memory, phase_a_procs=1, queue_depth=1)

    # The WARC before the failure is done; the one in phase B and the one queued behind it are unmarked,
    # with the start counts they had before the run.
    assert read_marker(store, root, warc_hash(done)).status == WarcStatus.OK
    assert store.stat(marker_path(root, warc_hash(failing))) is None
    assert store.stat(marker_path(root, warc_hash(queued))) is None
    assert json.loads(store.read(attempts_path(root, warc_hash(failing)))) == {"attempts": 2}
    assert store.stat(attempts_path(root, warc_hash(queued))) is None

    report = run_worker(urls, store, root, FakePhaseA(str(log)), PHASE_B, phase_a_procs=1, queue_depth=1)
    assert (report.n_already_done, report.n_done, report.n_abandoned) == (1, 2, 0)
    marker = read_marker(store, root, warc_hash(failing))
    assert (marker.status, marker.attempt, marker.n_kept) == (WarcStatus.OK, 3, 2)
    assert read_marker(store, root, warc_hash(queued)).attempt == 1


def test_phase_a_that_never_starts_stalls_without_charging_a_warc(tmp_path):
    root, store = str(tmp_path / "out"), LocalStore()
    hung = SlowStartPhaseA(_RELEASE_HUNG_STARTUP, str(tmp_path / "phase_a.log"))
    try:
        with pytest.raises(WorkerStalledError, match="phase A did not start"):
            run_worker([url(1)], store, root, hung, PHASE_B, phase_a_procs=1, queue_depth=1, stall_seconds=1)
    finally:
        _RELEASE_HUNG_STARTUP.set()  # lets the stuck subprocess finish starting so its pool can shut down
    assert store.stat(attempts_path(root, warc_hash(url(1)))) is None


def test_a_run_that_abandons_or_cannot_download_a_warc_exits_nonzero(tmp_path, caplog):
    root, store, log = str(tmp_path / "out"), LocalStore(), tmp_path / "phase_a.log"
    poison, unfetchable = url(3), url(7)
    store.write(attempts_path(root, warc_hash(poison)), b'{"attempts": 3}')
    urls = [poison, unfetchable, url(2)]
    phase_a = FakePhaseA(str(log), unfetchable=frozenset({unfetchable}))

    report = run_worker(urls, store, root, phase_a, PHASE_B, phase_a_procs=1, queue_depth=2)

    with pytest.raises(SystemExit) as exited:
        exit_if_warcs_failed(report)
    assert exited.value.code == 1
    assert "2 of 3 WARCs failed in this run: 1 abandoned and 1 download_failed" in caplog.text
    # The rerun finds every WARC marked, so it has nothing to fail.
    exit_if_warcs_failed(run_worker(urls, store, root, phase_a, PHASE_B, phase_a_procs=1, queue_depth=2))


def test_shards_match_production_base_layout():
    # Two of production's base shards (n_shards=32000), each listed sorted as production stored it.
    shards = json.loads(gzip.decompress((DATA / "asrun_base_shards_05000_10000.json.gz").read_bytes()))
    mixed = sorted(shards["5000"] + shards["10000"], reverse=True)
    assert select_shard(mixed, 10000, 32000) == shards["10000"]
    assert select_shard(mixed, 5000, 32000) == shards["5000"]


def test_worker_returns_when_phase_a_processes_own_extraction_pools(tmp_path):
    # Regression (the R2 TPU parity run hung for hours after its last WARC): production's phase-A processes
    # each own a spawned extraction pool. While such a process exits, multiprocessing closes queue
    # feeder threads in an exit finalizer of priority 10, so a pool that is shut down later, or
    # without waiting, strands its children's stop sentinels and every join waits forever.
    proc = subprocess.Popen(
        [sys.executable, "-c", POOL_WORKER_SCRIPT, str(tmp_path / "out"), str(tmp_path / "phase_a.log")],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=SHUTDOWN_TIMEOUT)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        pytest.fail(f"run_worker did not return within {SHUTDOWN_TIMEOUT}s of finishing its only WARC")
    assert proc.returncode == 0, err
    assert out.split()[-1] == "1"
