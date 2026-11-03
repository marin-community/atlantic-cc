# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase B decisions and outputs, replayed against the as-run outputs of fixture F3 (4ff1f9108b5a).

F3's 9,513 fastText survivors are its kept rows plus its tombstone rows. Fake models return the
as-run probability of each document (kept: exact float32; tombstones: their six-decimal values,
none of which sits within rounding distance of a threshold), so the cascade must reproduce the
as-run kept columns, the tombstone bytes and the as-run marker's counters exactly.
"""

import gzip
import json
import random
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from atlantic_cc.pipeline.outputs import tombstones_jsonl_gz_bytes
from atlantic_cc.pipeline.phase_b import run_phase_b
from atlantic_cc.schemas import KEPT_SCHEMA, PRESURVIVOR_SCHEMA
from atlantic_cc.spec import CLS_TOKEN_ID, POOLED_MAX_TOKENS, SEP_TOKEN_ID, TERMINAL_MAX_TOKENS

DATA = Path(__file__).parent / "data"
F3_HASH = "4ff1f9108b5a"
# as-run marker _b_marks/shard-10000/data-4ff1f9108b5a
F3_COUNTS = {"n_presurvivors": 9513, "n_hi_accept": 1396, "n_band": 3304, "n_kept": 2830}
MARKER_BASE = 100_000  # document j is tokenized as [CLS] + [MARKER_BASE + j] * k + [SEP]
_DOC_RE = re.compile(r"doc (\d+)")


class FakeTokenizer:
    """Tokenizes the normalized text "doc {j}" into ``n_tokens[j]`` ids that name document j."""

    def __init__(self, n_tokens: list[int]) -> None:
        self.n_tokens = n_tokens

    def __call__(self, texts: list[str]) -> list[np.ndarray]:
        out = []
        for text in texts:
            j = int(_DOC_RE.fullmatch(text).group(1))
            out.append(np.array([CLS_TOKEN_ID, *[MARKER_BASE + j] * (self.n_tokens[j] - 2), SEP_TOKEN_ID], np.int32))
        return out


class FakeScorer:
    """Returns ``probs[j]`` for the row of document j and 0 for empty rows."""

    def __init__(self, probs: np.ndarray, max_tokens: int) -> None:
        self.probs = probs
        self.max_tokens = max_tokens
        self.longest_input = 0

    def score(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        real = attention_mask.sum(axis=1)
        self.longest_input = max(self.longest_input, int(real.max()))
        rows = np.flatnonzero(real)
        # A document longer than the model's context reaches it cut to max_tokens, still ending in [SEP].
        assert (input_ids[rows, real[rows] - 1] == SEP_TOKEN_ID).all()
        out = np.zeros(len(input_ids), dtype=np.float32)
        out[rows] = self.probs[input_ids[rows, 1] - MARKER_BASE]
        return out


def merged_f3_survivors() -> tuple[pa.Table, list[int], np.ndarray, np.ndarray, pa.Table, bytes]:
    """F3's survivors (kept and dropped interleaved, each in WARC order) with per-document model outputs."""
    kept = pq.read_table(DATA / f"f3_{F3_HASH}_kept_scores.parquet")
    tombstone_bytes = gzip.decompress((DATA / f"f3_{F3_HASH}_tombstones.jsonl.gz").read_bytes())
    tombstones = [json.loads(line) for line in tombstone_bytes.decode().splitlines()]

    # Any interleaving that keeps each list's order is a valid WARC order for these outputs.
    sources = ["kept"] * kept.num_rows + ["tombstone"] * len(tombstones)
    random.Random(0).shuffle(sources)
    next_kept = next_tombstone = 0
    rows, n_tokens, pooled, terminal = [], [], [], []
    for j, source in enumerate(sources):
        if source == "kept":
            k = next_kept
            next_kept += 1
            doc_id, ft, nt = (kept.column(c)[k].as_py() for c in ("doc_id", "fasttext_score", "n_tokens"))
            pooled.append(kept.column("pooled_prob")[k].as_py())
            terminal.append(kept.column("modernbert_prob")[k].as_py())
        else:
            row = tombstones[next_tombstone]
            next_tombstone += 1
            doc_id, ft, nt = row["doc_id"], 0.5, 3 + (j * 7919) % 600
            pooled.append(row["pooled_prob"])
            terminal.append(np.nan if row["modernbert_prob"] is None else row["modernbert_prob"])
        rows.append(
            {
                "doc_id": doc_id,
                "url": f"https://example.com/{j}",
                "warc_hash": F3_HASH,
                "snapshot": "CC-MAIN-2026-12",
                "fasttext_score": ft,
                "text": f"Doc\n  {j} ",
            }
        )
        n_tokens.append(nt)
    table = pa.Table.from_pylist(rows, schema=PRESURVIVOR_SCHEMA)
    return table, n_tokens, np.array(pooled, np.float32), np.array(terminal, np.float32), kept, tombstone_bytes


def test_f3_replay_reproduces_asrun_kept_tombstones_and_counters():
    survivors, n_tokens, pooled_probs, terminal_probs, asrun_kept, asrun_tombstones = merged_f3_survivors()
    pooled = FakeScorer(pooled_probs, POOLED_MAX_TOKENS)
    terminal = FakeScorer(terminal_probs, TERMINAL_MAX_TOKENS)

    kept, tombstones, counters = run_phase_b(survivors, FakeTokenizer(n_tokens), pooled, terminal)

    assert kept.schema.equals(KEPT_SCHEMA)
    assert kept.column("doc_id").to_pylist() == asrun_kept.column("doc_id").to_pylist()
    for name in ("fasttext_score", "n_tokens", "pooled_prob", "modernbert_prob"):
        ours = kept.column(name).to_numpy()
        np.testing.assert_array_equal(ours.view(np.uint32), asrun_kept.column(name).to_numpy().view(np.uint32))
    assert gzip.decompress(tombstones_jsonl_gz_bytes(tombstones)) == asrun_tombstones
    assert {k: getattr(counters, k) for k in F3_COUNTS} == F3_COUNTS
    # The pooled model saw the full 8,192-token documents; the terminal model only their first 2,048 ids.
    assert pooled.longest_input == POOLED_MAX_TOKENS
    assert terminal.longest_input == TERMINAL_MAX_TOKENS


def test_warc_without_survivors_yields_an_empty_kept_table():
    no_docs = np.zeros(0, np.float32)
    kept, tombstones, counters = run_phase_b(
        PRESURVIVOR_SCHEMA.empty_table(),
        FakeTokenizer([]),
        FakeScorer(no_docs, POOLED_MAX_TOKENS),
        FakeScorer(no_docs, TERMINAL_MAX_TOKENS),
    )
    assert kept.num_rows == 0 and kept.schema.equals(KEPT_SCHEMA)
    assert tombstones == []
    assert (counters.n_presurvivors, counters.n_hi_accept, counters.n_band, counters.n_kept) == (0, 0, 0, 0)
