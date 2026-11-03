# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A stand-in ``resiliparse._extract_rs`` package for tests that must not need the real extension.

:func:`write_fake_engine` writes it under a directory that is then used as the pool's ``pkg_dir``,
so spawned children import it exactly as they would import an installed artifact. Its
``extract_plain_text`` returns ``"md:" + html`` and reacts to markers in the page:

* ``KILL``: SIGKILL its own process (a hard death the pool cannot catch, like the engine's SIGSEGV;
  SIGKILL avoids macOS crash reports and is what an OOM kill delivers);
* ``PANIC``: raise a ``BaseException`` (pyo3's ``PanicException`` is one);
* ``NOTHING``: return "";
* ``SPACES``: return whitespace only.
"""

from pathlib import Path

FAKE_ENGINE_SOURCE = """
import os
import resource
import signal


class PanicException(BaseException):
    pass


def extract_plain_text(html, main_content=False, preserve_formatting=False):
    assert main_content is True and preserve_formatting == "markdown"
    if "KILL" in html:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        os.kill(os.getpid(), signal.SIGKILL)
    if "PANIC" in html:
        raise PanicException("engine panicked")
    if "NOTHING" in html:
        return ""
    if "SPACES" in html:
        return " \\n\\t "
    return "md:" + html
"""


def raise_on_marker(args: tuple[list[str], int]) -> list[tuple[str, bool]]:
    """Pool task that raises (the child survives) when a page contains ``RAISE``; production counts it as a crash."""
    htmls, _cap = args
    if any("RAISE" in h for h in htmls):
        raise ValueError("extractor is misconfigured")
    return [("ok:" + h, True) for h in htmls]


def write_fake_engine(root: Path, source: str = FAKE_ENGINE_SOURCE) -> str:
    """Write ``resiliparse/_extract_rs.py`` under ``root``; returns ``root`` as a pool ``pkg_dir``."""
    pkg = root / "resiliparse"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "_extract_rs.py").write_text(source)
    return str(root)
