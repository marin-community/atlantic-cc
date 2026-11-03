# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-page extraction with resiliparse-rs, run inside the pool's child processes.

``resiliparse._extract_rs.extract_plain_text(html, main_content=True, preserve_formatting="markdown")``
from the resiliparse-rs fork ``marin-community/atlantic-cc-resiliparse`` (tag ``atlantic-cc-v1.0``) at
850891b, behind production's screens:

* pages longer than 50,000,000 code points give "";
* pages containing ``<frameset`` or ``<FRAMESET`` (case-sensitive) give "", because frameset pages
  hard-crash the engine; a mixed-case ``<FrameSet`` still reaches it and is caught by the pool;
* any exception from the engine except ``KeyboardInterrupt``/``SystemExit`` gives "" (pyo3's
  ``PanicException`` derives from ``BaseException``; lone surrogates raise ``UnicodeEncodeError``).

The extension is imported per call, after the screens and outside the ``try``: a missing or broken
install raises on every page, which the pool turns into a loud failure instead of empty documents.
"""

import logging

from atlantic_cc.extract.prefilter import in_population

logger = logging.getLogger(__name__)

OUT_OF_POPULATION = ("", False)

# A page with real prose, so a working extractor returns non-empty main content.
SMOKE_HTML = (
    "<html><body><article><h1>Smoke</h1><p>This paragraph exists so the startup smoke test can "
    "assert that the Rust extractor returns real main content rather than an empty string.</p>"
    "</article></body></html>"
)


def extract_text(html: str, max_html_chars: int) -> str:
    """Markdown main content of one page, or "" when screened out or when the engine fails."""
    if len(html) > max_html_chars:
        logger.warning("skipping resiliparse-rs on %d-char page (> %d cap)", len(html), max_html_chars)
        return ""
    if "<frameset" in html or "<FRAMESET" in html:
        return ""
    from resiliparse._extract_rs import extract_plain_text  # noqa: PLC0415 - extension installed at runtime

    try:
        return extract_plain_text(html, main_content=True, preserve_formatting="markdown")
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:  # as-run: pyo3 PanicException is a BaseException
        return ""


def screen_batch(args: tuple[list[str], int]) -> list[tuple[str, bool]]:
    """Pool task: ``(text, in_population)`` per page; out-of-population pages are not extracted."""
    htmls, max_html_chars = args
    out: list[tuple[str, bool]] = []
    for html in htmls:
        if not in_population(html):
            out.append(OUT_OF_POPULATION)
            continue
        out.append((extract_text(html, max_html_chars), True))
    return out
