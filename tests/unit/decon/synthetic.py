# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic synthetic eval records, documents and drop-set samples for as-run goldens.

``tests/golden/decon/generate_asrun_goldens.py`` runs the as-run Marin code on exactly these inputs;
the unit tests run the port on them and compare. Only ``random.Random`` with integer seeds is used,
so the inputs are identical on every platform and Python 3.12 build.
"""

import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Whitespace and alphabetic edge cases: Python's str.split() whitespace (incl. \x1c-\x1f, NBSP,
# U+2028), non-whitespace look-alikes (ZWSP, U+180E), combining marks, CJK, digits only.
EDGE_TOKENS = [
    "a\x1cb",
    "c\x1fd",
    "e\xa0f",
    "g h",
    "i​j",
    "k᠎l",
    "́",
    "x́",
    "日本語",
    "١٢٣",
    "123",
    "--",
    "Ⅻ",
    "ǅ",
    "ß",
    "İ",
    "ﬁ",
    "𝔘",
    "½",
]
SOURCES = ["CC-MAIN-2013-20", "CC-MAIN-2014-23", "CC-MAIN-2019-30", "CC-NEWS", "CC-SUPPLEMENTAL-2026-22"]


def _vocab(rng: random.Random) -> list[str]:
    return [f"w{i}" for i in range(200)] + EDGE_TOKENS + [str(i) for i in range(20)]


def eval_texts(seed: int = 11) -> list[str]:
    rng = random.Random(seed)
    vocab = _vocab(rng)
    texts = []
    for i in range(60):
        n = rng.choice([3, 5, 9, 12, 13, 14, 20, 40, 80])
        paras = [" ".join(rng.choice(vocab) for _ in range(n))]
        if i % 3 == 0:
            paras += [" ".join(rng.choice(vocab) for _ in range(rng.randint(1, 6))) for _ in range(rng.randint(1, 5))]
        texts.append(rng.choice(["\n\n", "\n", " "]).join(paras))
    return texts


def documents(seed: int = 12, n: int = 400) -> list[str]:
    """Documents mixing eval text (whole, partial, embedded) with filler and edge tokens."""
    rng = random.Random(seed)
    vocab = _vocab(rng)
    evals = eval_texts()
    docs = []
    for _ in range(n):
        paras = []
        for _ in range(rng.randint(1, 5)):
            kind = rng.random()
            if kind < 0.3:
                e = rng.choice(evals).split()
                start = rng.randint(0, max(0, len(e) - 1))
                paras.append(" ".join(e[start : start + rng.randint(1, 30)]))
            elif kind < 0.4:
                paras.append(rng.choice(evals))
            else:
                paras.append(" ".join(rng.choice(vocab) for _ in range(rng.randint(0, 50))))
        docs.append(rng.choice(["\n\n", "\n\n\n", "\r\n\r\n", "\n"]).join(paras))
    return docs


def write_sample(root: Path, seed: int = 13) -> list[str]:
    """A drop-set sample ``root/<source>/part-NNNNNN.parquet`` (``id, text``) with boilerplate-heavy sources."""
    rng = random.Random(seed)
    vocab = _vocab(rng)
    evals = eval_texts()
    boiler = [" ".join(f"b{j}t{i}" for i in range(15)) + " " + rng.choice(evals) for j in range(30)]
    for si, source in enumerate(SOURCES):
        (root / source).mkdir(parents=True, exist_ok=True)
        for part in range(rng.randint(2, 6)):
            texts: list[str | None] = []
            for _ in range(rng.randint(300, 1_800)):
                paras = [" ".join(rng.choice(vocab) for _ in range(rng.randint(3, 30)))]
                for bi, b in enumerate(boiler):
                    if rng.random() < (0.012 if bi % len(SOURCES) == si else 0.002):
                        paras.insert(rng.randint(0, len(paras)), b)
                if rng.random() < 0.02:
                    paras.append(rng.choice(evals))
                texts.append(rng.choice(["", None]) if rng.random() < 0.02 else "\n\n".join(paras))
            table = pa.table({"id": [f"{source}-{part}-{i}" for i in range(len(texts))], "text": texts})
            pq.write_table(table, root / source / f"part-{part * 37:06d}.parquet")
    return SOURCES
