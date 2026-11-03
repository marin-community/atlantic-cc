# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fake phase-A factories for the worker tests.

They run in the worker's spawned subprocesses, which import this module to unpickle them, so it
stays free of the tokenizer and model stacks. Synthetic WARC ``w{k}`` holds k fastText survivors
with texts "Doc 0" .. "Doc k-1".
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from atlantic_cc.extract.pool import ExtractorPool
from atlantic_cc.pipeline.phase_a import PhaseACounters
from atlantic_cc.schemas import PRESURVIVOR_SCHEMA
from atlantic_cc.spec import MAX_HTML_CHARS
from atlantic_cc.warc.identity import warc_hash

_WARC_RE = re.compile(r"/w(\d+)\.warc\.gz$")
EXTRACT_EMPTY_PER_WARC = 2


def url(k: int) -> str:
    return f"s3://commoncrawl/crawl-data/CC-MAIN-2026-12/segments/1/warc/w{k}.warc.gz"


@dataclass(frozen=True)
class FakePhaseA:
    """Logs ``start <pid> <url>`` per WARC and ``close <pid>`` per process to ``log``.

    WARCs in ``raises`` fail with an error; WARCs in ``unfetchable`` cannot be downloaded.
    """

    log: str
    raises: frozenset[str] = field(default_factory=frozenset)
    unfetchable: frozenset[str] = field(default_factory=frozenset)

    def __call__(self) -> "FakeRunner":
        return FakeRunner(self)


@dataclass(frozen=True)
class FakeRunner:
    config: FakePhaseA

    def _log(self, line: str) -> None:
        with open(self.config.log, "a") as f:
            f.write(f"{line}\n")

    def run(self, warc_path: str) -> tuple[pa.Table, PhaseACounters]:
        self._log(f"start {os.getpid()} {warc_path}")
        if warc_path in self.config.raises:
            raise ValueError(f"corrupt gzip member in {warc_path}")
        if warc_path in self.config.unfetchable:
            counters = PhaseACounters(download_failed=True, download_error=f"Failed to download {warc_path}")
            return PRESURVIVOR_SCHEMA.empty_table(), counters
        k = int(_WARC_RE.search(warc_path).group(1))
        rows = [
            {
                "doc_id": f"w{k}-{i}",
                "url": f"https://example.com/{i}",
                "warc_hash": warc_hash(warc_path),
                "snapshot": "CC-MAIN-2026-12",
                "fasttext_score": 0.5,
                "text": f"Doc {i}",
            }
            for i in range(k)
        ]
        counters = PhaseACounters(n_in=k + EXTRACT_EMPTY_PER_WARC, n_extract_empty=EXTRACT_EMPTY_PER_WARC)
        return pa.Table.from_pylist(rows, schema=PRESURVIVOR_SCHEMA), counters

    def close(self) -> None:
        self._log(f"close {os.getpid()}")


@dataclass(frozen=True)
class BrokenPhaseA:
    """Fails to start in every subprocess, as production's phase A does without the extractor installed."""

    def __call__(self) -> FakeRunner:
        raise RuntimeError("resiliparse-rs smoke extraction failed (crashed=['_smoke'], result=('', False))")


@dataclass(frozen=True)
class SlowStartPhaseA:
    """Does not finish starting until ``release`` (a spawn-context Event) is set: a hung model load."""

    release: Any
    log: str

    def __call__(self) -> FakeRunner:
        self.release.wait()
        return FakeRunner(FakePhaseA(self.log))


@dataclass(frozen=True)
class BlockingPhaseA:
    """Does not finish a WARC until ``release`` (a spawn-context Event) is set: an extractor hang."""

    release: Any

    def __call__(self) -> "BlockingRunner":
        return BlockingRunner(self.release)


@dataclass(frozen=True)
class BlockingRunner:
    release: Any

    def run(self, warc_path: str) -> tuple[pa.Table, PhaseACounters]:
        self.release.wait()
        return PRESURVIVOR_SCHEMA.empty_table(), PhaseACounters()

    def close(self) -> None:
        pass


def echo_pages(task: tuple[list[str], int]) -> list[tuple[str, bool]]:
    """Stand-in for the resiliparse-rs task: every page "extracts" to itself."""
    htmls, _ = task
    return [(html, True) for html in htmls]


@dataclass(frozen=True)
class PoolPhaseA:
    """A ``FakePhaseA`` whose runner also owns a real spawned ``ExtractorPool``, as production's does."""

    log: str

    def __call__(self) -> "PoolRunner":
        return PoolRunner(FakeRunner(FakePhaseA(self.log)), ExtractorPool(1, None, task_fn=echo_pages))


@dataclass(frozen=True)
class PoolRunner:
    fake: FakeRunner
    pool: ExtractorPool

    def run(self, warc_path: str) -> tuple[pa.Table, PhaseACounters]:
        self.pool.run(["<p>page</p>"], ["page"], MAX_HTML_CHARS)
        return self.fake.run(warc_path)

    def close(self) -> None:
        self.pool.close()
        self.fake.close()
