# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""U-PRE (processes): crash isolation with real process deaths, task sizing, the startup smoke test.

The pool runs the production task (``engine.screen_batch``) in spawned children that import a fake
``resiliparse._extract_rs`` from ``pkg_dir`` (see ``fake_engine``), so no extension is needed.
"""

import multiprocessing as mp
import signal

import pytest

from atlantic_cc import spec
from atlantic_cc.extract.pool import (
    MIN_CRASH_SAMPLE,
    TASK_MAX_CHARS,
    ExtractorPool,
    open_extractor_pool,
    size_aware_chunks,
)
from tests.unit.extract.fake_engine import raise_on_marker, write_fake_engine

CAP = spec.MAX_HTML_CHARS


@pytest.fixture
def fake_pkg_dir(tmp_path) -> str:
    return write_fake_engine(tmp_path / "fake")


def _page(i: int, marker: str = "") -> str:
    return f"<body>doc {i} {marker}</body>"


def _kill_marker_child(pkg_dir: str) -> None:
    import sys  # noqa: PLC0415 - runs in a fresh spawned interpreter

    sys.path.insert(0, pkg_dir)
    from resiliparse._extract_rs import extract_plain_text  # noqa: PLC0415

    extract_plain_text("<body>KILL</body>", main_content=True, preserve_formatting="markdown")


def test_kill_marker_is_a_real_process_death(fake_pkg_dir):
    proc = mp.get_context("spawn").Process(target=_kill_marker_child, args=(fake_pkg_dir,))
    proc.start()
    proc.join(60)
    assert proc.exitcode == -signal.SIGKILL


def test_a_killed_page_costs_only_that_page(fake_pkg_dir):
    pages = [_page(i) for i in range(30)]
    pages[3] = _page(3, "KILL")
    pages[17] = _page(17, "PANIC")
    pages[20] = "<body></body>outside"
    with ExtractorPool(2, fake_pkg_dir) as pool:
        results, crashed = pool.run(pages, [f"d{i}" for i in range(30)], CAP)
        after = pool.run([_page(99)], ["d99"], CAP)  # the poisoned executor was rebuilt

    expected = [("md:" + p, True) for p in pages]
    expected[3] = ("", False)  # crashed: counted as prefilter-dropped and as crashed
    expected[17] = ("", True)  # a panic is caught in the child: an empty extraction, not a crash
    expected[20] = ("", False)  # out of the population, never extracted
    assert results == expected
    assert crashed == ["d3"]
    assert after == ([("md:" + _page(99), True)], [])


def test_pool_preserves_order_across_tasks(fake_pkg_dir):
    pages = [_page(i) for i in range(103)]
    with ExtractorPool(3, fake_pkg_dir) as pool:
        results, crashed = pool.run(pages, [str(i) for i in range(103)], CAP)
    assert results == [("md:" + p, True) for p in pages]
    assert crashed == []


def test_mostly_crashing_chunk_means_a_broken_install(fake_pkg_dir):
    n = MIN_CRASH_SAMPLE
    with ExtractorPool(1, fake_pkg_dir, task_fn=raise_on_marker) as pool:
        # exactly half crashing is tolerated ...
        pages = [_page(i, "RAISE" if i % 2 else "") for i in range(n)]
        results, crashed = pool.run(pages, [str(i) for i in range(n)], CAP)
        assert crashed == [str(i) for i in range(1, n, 2)]
        assert results == [("", False) if i % 2 else ("ok:" + p, True) for i, p in enumerate(pages)]
        # ... more than half is not
        with pytest.raises(RuntimeError, match="install is broken"):
            pool.run([_page(i, "RAISE" if i <= n // 2 else "") for i in range(n)], [str(i) for i in range(n)], CAP)


def test_small_chunks_may_crash_entirely(fake_pkg_dir):
    n = MIN_CRASH_SAMPLE - 1
    with ExtractorPool(1, fake_pkg_dir, task_fn=raise_on_marker) as pool:
        results, crashed = pool.run([_page(i, "RAISE") for i in range(n)], [str(i) for i in range(n)], CAP)
    assert results == [("", False)] * n
    assert crashed == [str(i) for i in range(n)]


def test_missing_engine_fails_the_smoke_test(tmp_path):
    no_engine = write_fake_engine(tmp_path / "broken", source="raise ImportError('no extension here')\n")
    with pytest.raises(RuntimeError, match="smoke extraction failed"):
        open_extractor_pool(1, no_engine)


def test_empty_engine_fails_the_smoke_test(tmp_path):
    silent = write_fake_engine(tmp_path / "silent", source="def extract_plain_text(html, **kw):\n    return ' '\n")
    with pytest.raises(RuntimeError, match="smoke extraction failed"):
        open_extractor_pool(1, silent)


def test_working_engine_passes_the_smoke_test(fake_pkg_dir):
    with open_extractor_pool(2, fake_pkg_dir) as pool:
        assert pool.run([_page(1)], ["d1"], CAP) == ([("md:" + _page(1), True)], [])


def test_size_aware_chunks_bound_docs_and_characters():
    small = ["x" * 1000] * 100
    assert [len(c) for c in size_aware_chunks(small)] == [25, 25, 25, 25]

    big = ["y" * (10 * 1024 * 1024)] * 8
    chunks = size_aware_chunks(big)
    assert [len(c) for c in chunks] == [3, 3, 2]
    assert all(sum(map(len, c)) <= TASK_MAX_CHARS for c in chunks)

    huge = ["z" * (TASK_MAX_CHARS + 1)]
    assert size_aware_chunks(huge) == [huge]  # an oversized page gets its own task

    mixed = ["a" * 100, "b" * (TASK_MAX_CHARS // 2), "c" * (TASK_MAX_CHARS // 2), "d" * 100]
    chunks = size_aware_chunks(mixed)
    assert [h for c in chunks for h in c] == mixed
    assert [len(c) for c in chunks] == [2, 2]
