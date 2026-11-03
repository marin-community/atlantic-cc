# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A small deterministic kept corpus that exercises every dedup rule.

Only the standard library and pyarrow: ``tests/golden/dedup/generate_asrun.py`` imports this file from the Marin
environment to run the as-run code on exactly the same documents.

What it holds: exact duplicates inside and across files, near-duplicate chains that need several component rounds,
contained prefixes (c075 removals), missing / empty / blank texts (``\\u3000``, ``\\x1c``), whitespace runs over 128
characters (and one that compacts onto another document), NaN scores, non-ASCII text (Greek final sigma, Turkish
dotted I, CJK, U+2028, NUL) and float values whose JSON form is unusual (1e-05, 1e+16).
"""

import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

KEPT_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("url", pa.string()),
        ("warc_hash", pa.string()),
        ("snapshot", pa.string()),
        ("fasttext_score", pa.float32()),
        ("text", pa.large_string()),
        ("n_tokens", pa.int32()),
        ("pooled_prob", pa.float32()),
        ("modernbert_prob", pa.float32()),
    ]
)
WORDS = [f"w{i}" for i in range(400)] + ["the", "and", "of", "to", "in", "is", "that", "for"]
UNICODE_TEXTS = [
    "ΟΔΥΣΣΕΥΣ ΟΔΟΣ Σ naïve café — 日本語のテキスト",
    "İstanbul ıi İI ß straße STRASSE",
    "line\u2028separator and a NUL \x00 byte",
    "tab\tnew\nline   nbsp\u00a0ideographic\u3000space",
]
N_FILES = 9


def _edit(rng: random.Random, words: list[str], fraction: float) -> list[str]:
    out = list(words)
    for _ in range(max(1, int(len(out) * fraction))):
        out[rng.randrange(len(out))] = rng.choice(WORDS)
    return out


def _texts(seed: int) -> tuple[list[str], list[str]]:
    """A pool of texts that files sample from, and a drifting chain spread over every file."""
    rng = random.Random(seed)
    bases = [[rng.choice(WORDS) for _ in range(rng.randint(60, 250))] for _ in range(40)]
    pool: list[str] = []
    for base in bases:
        pool.append(" ".join(base))
        chain = base
        for _ in range(rng.choice([0, 1, 3])):
            chain = _edit(rng, chain, 0.04)
            pool.append(" ".join(chain))
        # A contained prefix: c075 removes it when it lands in its base's component.
        pool.append(" ".join(base[: int(len(base) * 0.85)]))
    pool.append("x" + " " * 400 + " ".join(bases[0]))
    pool.append("x" + " " * 128 + " ".join(bases[0]))
    pool.append(" ".join(bases[1]) + "\x1f" * 200 + "tail")
    pool.extend(UNICODE_TEXTS)
    drift = [rng.choice(WORDS) for _ in range(300)]
    chain_docs = []
    for _ in range(60):
        drift = _edit(rng, drift, 0.02)
        chain_docs.append(" ".join(drift))
    return pool, chain_docs


def write_corpus(directory: Path, seed: int = 0, n_files: int = N_FILES) -> list[Path]:
    """Write ``n_files`` kept-schema parquet files under ``directory``; returns their paths in sorted order."""
    rng = random.Random(seed + 1)
    pool, chain_docs = _texts(seed)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    special_floats = [1e-05, 1e16, 0.5, 0.0]
    for f in range(n_files):
        texts: list[str | None] = [rng.choice(pool) for _ in range(rng.randint(15, 30))]
        texts += [rng.choice(pool)] * 2 + [None, "", " \n ", "\u3000", "\x1c"] + chain_docs[f::n_files]
        rows = []
        for i, text in enumerate(texts):
            rows.append(
                {
                    "doc_id": f"{f:02d}-{i:03d}",
                    "url": f"https://example.com/{f}/{i}",
                    "warc_hash": f"{f:012x}",
                    "snapshot": "CC-MAIN-2020-10" if i % 2 else "CC-NEWS-2021-04",
                    "fasttext_score": special_floats[i % 4] if i < 4 else rng.random(),
                    "text": text,
                    "n_tokens": len(text.split()) if text else 0,
                    "pooled_prob": float("nan") if i % 5 == 0 else rng.random(),
                    "modernbert_prob": float("nan") if i % 3 == 0 else rng.random(),
                }
            )
        path = directory / f"data-{f:012x}.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=KEPT_SCHEMA), path)
        paths.append(path)
    return sorted(paths)


def write_big_cluster(directory: Path, seed: int, members: int) -> list[Path]:
    """Two files holding one near-duplicate family of ``members`` documents: a solve batch over 256 members."""
    rng = random.Random(seed)
    base = [rng.choice(WORDS) for _ in range(200)]
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in range(2):
        rows = []
        for i in range(members // 2):
            if rng.random() < 0.5:
                words = base[: rng.randint(120, 200)]
            else:
                words = _edit(rng, base, 0.02)
            text = " ".join(words)
            rows.append(
                {
                    "doc_id": f"big-{f}-{i:04d}",
                    "url": f"https://example.org/{f}/{i}",
                    "warc_hash": f"b{f:011x}",
                    "snapshot": "CC-MAIN-2021-04",
                    "fasttext_score": 0.25,
                    "text": text,
                    "n_tokens": len(words),
                    "pooled_prob": 0.5,
                    "modernbert_prob": 0.75,
                }
            )
        path = directory / f"data-b{f:011x}.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=KEPT_SCHEMA), path)
        paths.append(path)
    return sorted(paths)


def solve_rows(seed: int, big_cluster: int) -> list[dict]:
    """Rows ``{cluster_key, id, text, d}`` sorted by ``(cluster_key, id)`` (as-run ``rdedup/test_solve.py``).

    Twelve clusters of truncations, light edits and unrelated texts; cluster ``000`` has ``big_cluster`` members.
    """
    rng = random.Random(seed)
    words = [f"t{i}" for i in range(300)]
    rows = []
    d = 0
    for c in range(12):
        base = [rng.choice(words) for _ in range(rng.randint(20, 120))]
        size = big_cluster if c == 0 else rng.randint(1, 9)
        for _ in range(size):
            kind = rng.random()
            if kind < 0.4:
                text = base[: rng.randint(5, len(base))]
            elif kind < 0.8:
                text = [w if rng.random() > 0.1 else rng.choice(words) for w in base]
            else:
                text = [rng.choice(words) for _ in range(rng.randint(5, 60))]
            rows.append(
                {"cluster_key": f"{c:03d}", "id": f"{rng.getrandbits(128):032x}", "text": " ".join(text), "d": d}
            )
            d += 1
    return sorted(rows, key=lambda r: (r["cluster_key"], r["id"]))
