# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Population prefilter: a page is extracted only if its ``<body>`` holds non-whitespace text.

``body_strip`` removes ``<script>`` elements, then concatenates the contents of every
``<body>...</body>`` (non-greedy); with no body match it keeps the whole cleaned page. Tags are not
removed, so any markup counts as text. A page with ``<body></body>`` and its content outside the
body is out of the population. Production applied this inside the extraction pool and counted the
pages it rejects in ``n_prefilter_dropped``; the classifier thresholds were calibrated on this
population.
"""

import re

from atlantic_cc.classify.text import normalize_text

_SCRIPT_TAG_RE = re.compile(r"<script\b[^>]*>.*?</script\s*>", re.IGNORECASE | re.DOTALL)
_BODY_TAG_RE = re.compile(r"<body\b[^>]*>(.*?)</body\s*>", re.IGNORECASE | re.DOTALL)


def body_strip(html: str) -> str:
    cleaned = _SCRIPT_TAG_RE.sub("", html)
    bodies = [m.group(1) for m in _BODY_TAG_RE.finditer(cleaned)]
    return "".join(bodies) if bodies else cleaned


def in_population(html: str) -> bool:
    """as-run ``fasttext_text(body_strip(html))`` is non-empty."""
    return bool(normalize_text(body_strip(html)))
