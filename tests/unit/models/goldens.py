# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Access to the synthetic model goldens (``tests/golden/models/synthetic.json`` + ``synthetic_ids.npz``).

The goldens hold logits and probabilities that the as-run Levanter code (tag ``lpv11-8m-as-run``)
produced on an x86_64 CPU for fixed synthetic token batches (see ``tests/golden/models/make_goldens.py``).
"""

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

from tests.unit.models import random_models as rm

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "golden" / "models"


@dataclass(frozen=True)
class Case:
    model: str  # pooled | ettin
    weights: str  # random:seed=N | asrun
    batch: str
    precision: str  # bf16 | fp32
    logits: np.ndarray  # float32 [B, 2]
    probs: np.ndarray  # float32 [B]

    @property
    def id(self) -> str:
        return f"{self.batch}-{self.precision}"


@cache
def _load() -> tuple[dict, dict[str, np.ndarray]]:
    doc = json.loads((GOLDEN_DIR / "synthetic.json").read_text())
    with np.load(GOLDEN_DIR / "synthetic_ids.npz") as npz:
        ids = {k: npz[k] for k in npz.files}
    return doc, ids


def cases(model: str, weights: str) -> list[Case]:
    doc, _ = _load()
    return [
        Case(
            model=c["model"],
            weights=c["weights"],
            batch=c["batch"],
            precision=c["precision"],
            logits=np.asarray(c["logits"], dtype=np.float32),
            probs=np.asarray(c["probs"], dtype=np.float32),
        )
        for c in doc["cases"]
        if c["model"] == model and c["weights"].split(":")[0] == weights
    ]


def batch(name: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """``(input_ids, real_mask, row_lengths)`` of a golden batch; 0-length rows are empty filler rows."""
    doc, ids = _load()
    spec = doc["batches"][name]
    input_ids, mask = rm.padded_batch(ids[name], spec["lengths"], spec["length"])
    return input_ids, mask, spec["lengths"]
