# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The classifier input: extracted text with whitespace collapsed, stripped and lowercased.

All three classifiers see exactly this string (fastText directly, the transformers after
tokenization). ``\\s`` and ``str.lower`` follow the running Python's Unicode tables; the release was
built with Python 3.12 (Unicode 15.0).
"""

import re

_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """``re.sub(r"\\s+", " ", text).strip().lower()``: collapse first, then lowercase."""
    return _WS_RE.sub(" ", text).strip().lower()
