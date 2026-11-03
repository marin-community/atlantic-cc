# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pick the real-document fixture of the model goldens from public Atlantic-CC v1.0 files.

Run once in this repo's environment::

    uv run python tests/golden/models/select_realdocs.py

It reads whole row groups of a few public release parquet files (``atlantic_cc.release.params.HTTPS_ROOT``;
cached under ``.cache/models/release/``), tokenizes the normalized text exactly as production did, and keeps about
2,000 documents stratified by stored pooled probability (deciles), token length, and closeness to
each cascade threshold. The result, ``.cache/models/realdocs/selection.parquet``, holds text and token
ids and is not committed; ``make_goldens.py realdocs`` turns it into the committed fixture
``realdocs.parquet``, which keeps only record ids, row-group locations, hashes and scores.
"""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from atlantic_cc import spec
from atlantic_cc.classify.text import normalize_text
from atlantic_cc.classify.tokenize import load_tokenizer, tokenize_batch
from tests.parity.models.common import release_row_group

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE = REPO_ROOT / ".cache" / "models"
# (release file, row groups): one news month in full plus single row groups of three CC-MAIN crawls
# from different eras (each row group is ~7 consecutive WARCs).
SOURCES = (
    ("crawl=CC-NEWS-2016-08/part-00000-of-00001.parquet", (0, 1, 2, 3, 4)),
    ("crawl=CC-MAIN-2013-20/part-00000-of-00410.parquet", (0,)),
    ("crawl=CC-MAIN-2021-04/part-00000-of-00255.parquet", (0,)),
    ("crawl=CC-MAIN-2024-33/part-00000-of-00189.parquet", (0, 1)),
)
COLUMNS = ["record_id", "crawl", "warc_path", "text", "fasttext_score", "pooled_prob", "modernbert_prob"]
TARGET_DOCS = 2000
NEAR = 0.02  # half-width of the near-threshold strata
NEAR_QUOTA = 150  # documents per near-threshold stratum
LENGTH_EDGES = (128, 512, 2048, spec.POOLED_MAX_TOKENS)  # bins [2,128) [128,512) [512,2048) [2048,8192) [8192]
SEED = 0


def strata(pooled: np.ndarray, terminal: np.ndarray, n_tokens: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "near_pooled_drop": (pooled >= spec.POOLED_DROP_BELOW) & (pooled < spec.POOLED_DROP_BELOW + NEAR),
        "near_pooled_accept_below": (pooled >= spec.POOLED_ACCEPT_AT - NEAR) & (pooled < spec.POOLED_ACCEPT_AT),
        "near_pooled_accept_above": (pooled >= spec.POOLED_ACCEPT_AT) & (pooled < spec.POOLED_ACCEPT_AT + NEAR),
        "near_terminal_keep": (terminal >= spec.TERMINAL_KEEP_AT) & (terminal < spec.TERMINAL_KEEP_AT + NEAR),
        "long_8192": n_tokens >= spec.POOLED_MAX_TOKENS,
        "long_2048_band": (n_tokens > spec.TERMINAL_MAX_TOKENS) & ~np.isnan(terminal),
    }


def select(
    pooled: np.ndarray, terminal: np.ndarray, n_tokens: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the fixture: near-threshold and long-document strata first, then decile x length cells."""
    chosen: dict[int, str] = {}
    for name, mask in strata(pooled, terminal, n_tokens).items():
        candidates = [i for i in np.flatnonzero(mask) if i not in chosen]
        for i in rng.permutation(candidates)[:NEAR_QUOTA]:
            chosen[int(i)] = name
    decile = np.minimum((pooled * 10).astype(int), 9)
    length_bin = np.searchsorted(LENGTH_EDGES, n_tokens, side="right")
    cells = {}
    for i in rng.permutation(len(pooled)):
        if int(i) not in chosen:
            cells.setdefault((int(decile[i]), int(length_bin[i])), []).append(int(i))
    while len(chosen) < TARGET_DOCS and any(cells.values()):
        for key in sorted(cells):
            if cells[key] and len(chosen) < TARGET_DOCS:
                chosen[cells[key].pop()] = f"decile{key[0]}_len{key[1]}"
    order = sorted(chosen)
    return np.array(order), np.array([chosen[i] for i in order])


def main() -> None:
    tables = []
    for name, row_groups in SOURCES:
        for rg in row_groups:
            t = release_row_group(name, rg, COLUMNS)
            n = t.num_rows
            t = t.append_column("source", pa.array([name] * n)).append_column(
                "row_group", pa.array([rg] * n, pa.int32())
            )
            tables.append(t)
    table = pa.concat_tables(tables)
    tok = load_tokenizer()
    texts = [normalize_text(t) for t in table.column("text").to_pylist()]
    ids = tokenize_batch(tok, texts)
    n_tokens = np.array([len(x) for x in ids])
    pooled = table.column("pooled_prob").to_numpy()
    terminal = table.column("modernbert_prob").to_numpy(zero_copy_only=False).astype(np.float64)
    rng = np.random.default_rng(SEED)
    idx, stratum = select(pooled, terminal, n_tokens, rng)
    band = ~np.isnan(terminal[idx])
    out = table.take(pa.array(idx))
    out = out.append_column("stratum", pa.array(stratum))
    out = out.append_column("n_tokens", pa.array(n_tokens[idx], pa.int32()))
    out = out.append_column("input_ids", pa.array([ids[i] for i in idx], pa.list_(pa.int32())))
    dest = CACHE / "realdocs" / "selection.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, dest)
    print(f"{len(table)} candidates -> {len(idx)} selected ({band.sum()} band docs) -> {dest}")
    for name in sorted(set(stratum)):
        print(f"  {name}: {int(np.sum(stratum == name))}")


if __name__ == "__main__":
    main()
