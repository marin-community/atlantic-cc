# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare a phase-A run with the as-run outputs of the same WARC (R1, I-WARC).

The as-run reference of one WARC is its ``kept/`` rows (doc_id, text, fasttext_score, n_tokens)
plus its tombstone doc_ids, in WARC order, and the phase-A counters of its marker. Because every
fastText survivor was either kept or tombstoned, ``kept ∪ tombstones`` is exactly the survivor set
phase A must reproduce, without any accelerator.

A reference comes from the committed goldens (:func:`load_golden`, text as sha256) or from the
as-run buckets (``r1_phase_a``).
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from atlantic_cc.pipeline.phase_a import PhaseACounters

GOLDEN_ROOT = Path(__file__).resolve().parents[1] / "golden" / "per_warc"
SURVIVORS_FILE = "survivors.parquet"
COUNTERS_FILE = "counters.json"
META_FILE = "meta.json"
COUNTER_NAMES = ("a_n_in", "a_n_prefilter_dropped", "a_n_extract_empty", "a_n_extract_crashed")
MAX_LISTED_IDS = 20
# Goldens store the first 64 bits of the text sha256: plenty to detect a changed text.
TEXT_HASH_HEX_DIGITS = 16


@dataclass(frozen=True)
class KeptDoc:
    text_sha256: str  # full hex digest, or its first TEXT_HASH_HEX_DIGITS for a golden
    fasttext_score: np.float32
    n_tokens: int


@dataclass
class Reference:
    kept: dict[str, KeptDoc]  # in kept-file (= WARC) order
    tombstones: list[str]  # in tombstone-file (= WARC) order
    counters: dict[str, int]  # COUNTER_NAMES plus n_presurvivors
    texts: dict[str, str] = field(default_factory=dict)  # full kept text when read from the buckets


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def golden_dirs() -> list[Path]:
    return sorted(p for p in GOLDEN_ROOT.iterdir() if (p / META_FILE).exists())


def load_golden(golden_dir: Path) -> tuple[dict, Reference]:
    """``(meta, reference)`` of one committed per-WARC golden."""
    meta = json.loads((golden_dir / META_FILE).read_text())
    counters = json.loads((golden_dir / COUNTERS_FILE).read_text())
    rows = pq.read_table(golden_dir / SURVIVORS_FILE).to_pylist()
    kept = {
        r["doc_id"]: KeptDoc(r["text_sha256_prefix"], np.float32(r["fasttext_score"]), r["n_tokens"])
        for r in rows
        if r["kept"]
    }
    tombstones = [r["doc_id"] for r in rows if not r["kept"]]
    return meta, Reference(
        kept=kept, tombstones=tombstones, counters={k: counters[k] for k in (*COUNTER_NAMES, "n_presurvivors")}
    )


def ours_counters(counters: PhaseACounters, n_presurvivors: int) -> dict[str, int]:
    return {
        "a_n_in": counters.n_in,
        "a_n_prefilter_dropped": counters.n_prefilter_dropped,
        "a_n_extract_empty": counters.n_extract_empty,
        "a_n_extract_crashed": counters.n_extract_crashed,
        "n_presurvivors": n_presurvivors,
    }


def compare(presurvivors: pa.Table, counters: PhaseACounters, ref: Reference, n_tokens: list[int] | None) -> dict:
    """A JSON-able report; ``report["exact"]`` is True iff phase A reproduced the reference exactly.

    ``n_tokens`` is our token count per presurvivor row (None to skip that check).
    """
    ids = presurvivors.column("doc_id").to_pylist()
    texts = presurvivors.column("text").to_pylist()
    scores = presurvivors.column("fasttext_score").to_numpy()
    ours = set(ids)
    expected = set(ref.kept) | set(ref.tombstones)
    missing = sorted(expected - ours)
    extra = sorted(ours - expected)

    kept_rows = [i for i, d in enumerate(ids) if d in ref.kept]
    text_mismatch = [ids[i] for i in kept_rows if not text_sha256(texts[i]).startswith(ref.kept[ids[i]].text_sha256)]
    score_diffs = np.array([abs(float(scores[i]) - float(ref.kept[ids[i]].fasttext_score)) for i in kept_rows])
    score_mismatch = [ids[i] for i in kept_rows if scores[i] != ref.kept[ids[i]].fasttext_score]
    tok_mismatch = [] if n_tokens is None else [ids[i] for i in kept_rows if n_tokens[i] != ref.kept[ids[i]].n_tokens]
    ours_c = ours_counters(counters, len(ids))
    counter_report = {k: {"ours": ours_c[k], "asrun": ref.counters[k]} for k in ref.counters}

    checks = {
        "survivor_set": not missing and not extra,
        "kept_order": [d for d in ids if d in ref.kept] == list(ref.kept),
        "tombstone_order": [d for d in ids if d not in ref.kept and d in expected] == ref.tombstones,
        "kept_text": not text_mismatch,
        "fasttext_score": not score_mismatch,
        "n_tokens": None if n_tokens is None else not tok_mismatch,
        "counters": all(v["ours"] == v["asrun"] for v in counter_report.values()),
    }
    return {
        "exact": all(v is not False for v in checks.values()) and not counters.download_failed,
        "checks": checks,
        "n_survivors": {"ours": len(ids), "asrun": len(expected)},
        "n_kept_compared": len(kept_rows),
        "missing": {"count": len(missing), "ids": missing[:MAX_LISTED_IDS]},
        "extra": {"count": len(extra), "ids": extra[:MAX_LISTED_IDS]},
        "kept_text_mismatch": {"count": len(text_mismatch), "ids": text_mismatch[:MAX_LISTED_IDS]},
        "fasttext_score_mismatch": {
            "count": len(score_mismatch),
            "ids": score_mismatch[:MAX_LISTED_IDS],
            "max_abs_diff": float(score_diffs.max()) if len(score_diffs) else 0.0,
        },
        "n_tokens_mismatch": None
        if n_tokens is None
        else {"count": len(tok_mismatch), "ids": tok_mismatch[:MAX_LISTED_IDS]},
        "counters": counter_report,
        "download_failed": counters.download_failed,
        "records": {
            "n_errors": counters.records.n_errors,
            "n_fffd": counters.records.n_fffd,
            "n_brotli": counters.records.n_brotli,
        },
    }
