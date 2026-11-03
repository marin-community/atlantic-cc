# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Eval names of matched phrases: the strings stored in the release's ``decon_matches`` column.

``eval_name`` is verbatim from Marin ``experiments/fast_curation/decon_8m.py`` (tag ``lpv11-8m-as-run``).
Its quirks are part of the release and must be kept: ids whose last ``-`` segment is neither all digits
nor >= 16 hex digits keep their full id (``scicode-77.1``, ``terminal_bench_2_1-crack-7z-hash``), and the
lm-eval split ``training`` is not stripped (``x-training-3`` -> ``x-training``).
"""

import re
from collections.abc import Iterable, Mapping, Sequence

_EVAL_ID_RE = re.compile(r"^(?P<name>.+?)(?:-(?:test|validation|train|dev))?-(?:\d+|[0-9a-f]{16,})$")


def eval_name(eval_id: str) -> str:
    """Eval name of a corpus record id: ``aa_lcr-1`` -> ``aa_lcr``, ``mmlu-test-3`` -> ``mmlu``,
    ``hle-66b2c7c9795022550abb176b`` -> ``hle``."""
    m = _EVAL_ID_RE.match(eval_id)
    return m.group("name") if m else eval_id


def decon_matches(matched_hashes: Iterable[int], names_of: Mapping[int, Sequence[str]]) -> list[str]:
    """Sorted union of the eval names of every matched hash (as-run ``manifest``)."""
    return sorted({name for h in matched_hashes for name in names_of.get(h, ())})
