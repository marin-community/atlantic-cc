# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Crash-isolating ``spawn`` process pool for resiliparse-rs (as-run ``cpu_phase_c.ResiliparseRsPool``).

The engine can kill its process (SIGSEGV on some ``<frameset>`` pages and on pages Common Crawl cut
at 1 MiB; about 4.4e-7 of pages). One page must cost one page, never the WARC:

* a chunk of pages is split into tasks of at most 25 pages and 32 MiB of characters (an oversized
  page gets a task of its own), mapped in order over ``n_procs`` children;
* if anything fails (``BrokenProcessPool`` from a dead child, or a raising task), the whole chunk is
  re-run one page per task on a single reused child; a page whose task dies or raises yields
  ``("", False)`` and its doc id is reported as crashed;
* if more than half of a chunk of at least 8 pages crashes, the install is broken and ``run``
  raises instead of producing empty documents.

Children are spawned (not forked) and pin numeric/Rayon thread pools to one thread.

Lifetime: ``close()`` (or leaving the ``with`` block) stops the children and waits for them, so their
shutdown sentinels go out while the executor's call-queue feeder thread is alive. A pool owned by a
``multiprocessing`` child must be closed before that child's queue finalizers run
(``multiprocessing.queues.Queue`` closes its feeder at exit priority 10): close it before the child's
target returns, or from a ``multiprocessing.util.Finalize`` with ``exitpriority`` above 10. Closed any
later, its children wait forever for a sentinel and the owning process hangs at exit joining them.
The as-run pool shut down with ``wait=False``; outputs are the same either way.
"""

import logging
import multiprocessing as mp
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

from atlantic_cc import spec
from atlantic_cc.extract.engine import SMOKE_HTML, screen_batch
from atlantic_cc.extract.install import activate

logger = logging.getLogger(__name__)

TASK_MAX_DOCS = 25
TASK_MAX_CHARS = 32 * 1024 * 1024
MAX_CRASH_FRACTION = 0.5
MIN_CRASH_SAMPLE = 8
CRASHED_RESULT = ("", False)

_SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
}

TaskFn = Callable[[tuple[list[str], int]], list[tuple[str, bool]]]


def _init_child(pkg_dir: str | None) -> None:
    os.environ.update(_SINGLE_THREAD_ENV)
    activate(pkg_dir)


def size_aware_chunks(
    htmls: list[str], max_docs: int = TASK_MAX_DOCS, max_chars: int = TASK_MAX_CHARS
) -> list[list[str]]:
    """Split pages into tasks of <= ``max_docs`` pages and <= ``max_chars`` characters, keeping order."""
    out: list[list[str]] = []
    cur: list[str] = []
    cur_chars = 0
    for html in htmls:
        size = len(html)
        if cur and (len(cur) >= max_docs or cur_chars + size > max_chars):
            out.append(cur)
            cur, cur_chars = [], 0
        cur.append(html)
        cur_chars += size
    if cur:
        out.append(cur)
    return out


class ExtractorPool:
    """Runs a task function over pages in ``n_procs`` spawned children; see the module docstring.

    ``task_fn`` must be a picklable top-level function; tests inject one that really kills its
    process. ``pkg_dir`` is an installed artifact's package directory, or None for a regular install.
    """

    def __init__(self, n_procs: int, pkg_dir: str | None, *, task_fn: TaskFn = screen_batch) -> None:
        self.n_procs = max(1, n_procs)
        self._ctx = mp.get_context("spawn")
        self._pkg_dir = pkg_dir
        self._task_fn = task_fn
        self._pool: ProcessPoolExecutor | None = None

    def __enter__(self) -> "ExtractorPool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _new_executor(self, n_procs: int) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(
            max_workers=n_procs, mp_context=self._ctx, initializer=_init_child, initargs=(self._pkg_dir,)
        )

    def run(
        self, htmls: list[str], doc_ids: list[str], max_html_chars: int
    ) -> tuple[list[tuple[str, bool]], list[str]]:
        """``(results, crashed_doc_ids)`` with one ``(text, in_population)`` per page, in input order."""
        if not htmls:
            return [], []
        if self._pool is None:
            self._pool = self._new_executor(self.n_procs)
        tasks = [(part, max_html_chars) for part in size_aware_chunks(htmls)]
        try:
            return [result for part in self._pool.map(self._task_fn, tasks) for result in part], []
        except Exception as e:  # as-run: a dead child or a raising page; isolate per document
            logger.warning(
                "resiliparse-rs batch of %d failed (%s: %s); isolating per doc", len(htmls), type(e).__name__, e
            )
        self.close()
        return self._isolate(htmls, doc_ids, max_html_chars)

    def _isolate(
        self, htmls: list[str], doc_ids: list[str], max_html_chars: int
    ) -> tuple[list[tuple[str, bool]], list[str]]:
        results: list[tuple[str, bool]] = []
        crashed: list[str] = []
        solo = self._new_executor(1)  # reused across pages; only a dead child forces a rebuild
        try:
            for doc_id, html in zip(doc_ids, htmls, strict=True):
                try:
                    results.append(solo.submit(self._task_fn, ([html], max_html_chars)).result()[0])
                    continue
                except BrokenProcessPool:
                    reason = "hard process death (SIGSEGV / OOM-kill)"
                    solo.shutdown(wait=False)  # its child is already dead
                    solo = self._new_executor(1)
                except Exception as e:
                    reason = f"{type(e).__name__}: {e}"
                results.append(CRASHED_RESULT)
                crashed.append(doc_id)
                logger.warning("resiliparse-rs dropped %s (%d chars): %s", doc_id, len(html), reason)
        finally:
            solo.shutdown(wait=True)
        if len(htmls) >= MIN_CRASH_SAMPLE and len(crashed) > MAX_CRASH_FRACTION * len(htmls):
            raise RuntimeError(
                f"resiliparse-rs crashed on {len(crashed)}/{len(htmls)} docs; the install is broken "
                "(expected ~1e-6). Refusing to produce a corpus of empty extractions."
            )
        return results, crashed

    def close(self) -> None:
        """Stop the children and wait for them (see "Lifetime" in the module docstring).

        Chunks not yet started are cancelled; after a dead child the executor has already terminated
        the others, so this returns at once.
        """
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None


def open_extractor_pool(n_procs: int, pkg_dir: str | None) -> ExtractorPool:
    """An :class:`ExtractorPool` that has passed production's startup smoke extraction."""
    pool = ExtractorPool(n_procs, pkg_dir)
    results, crashed = pool.run([SMOKE_HTML], ["_smoke"], spec.MAX_HTML_CHARS)
    if crashed or not results[0][0].strip():
        pool.close()
        raise RuntimeError(f"resiliparse-rs smoke extraction failed (crashed={crashed}, result={results[0]!r})")
    logger.info("resiliparse-rs ready (pkg_dir=%s, %d procs)", pkg_dir, pool.n_procs)
    return pool
