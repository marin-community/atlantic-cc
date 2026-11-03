# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the G1 per-WARC goldens from the as-run outputs (maintainers with read access to Marin's GCS).

For each fixture F1-F6 this reads the WARC's ``kept/`` parquet (us-central2) and its reference (the
tombstones and phase-A counters that ``python -m tests.parity.r1_phase_a refs --fixtures`` wrote from
the processing region; see ``tests/parity/r1_asrun.py``) and writes ``tests/golden/per_warc/<warc_hash>/``:

* ``survivors.parquet``: one row per fastText survivor, kept rows (in kept-file order) then tombstones
  (in tombstone-file order): ``doc_id``, ``kept``; for kept rows ``text_sha256_prefix`` (the first
  16 hex digits, 64 bits, of sha256 over the UTF-8 text), ``fasttext_score`` and ``n_tokens``; and
  ``pooled_prob`` / ``modernbert_prob`` as float32 (exact for kept rows; for tombstones
  ``round(float(x), 6)`` recovers the stored 6-decimal value exactly). ``modernbert_prob`` is NaN
  for a pooled accept and null for a pooled drop;
* ``counters.json``: phase-A and phase-B counters plus the raw source records;
* ``meta.json``: WARC path, shard, region, device, processing time, code path, Content-Length-guard
  era and the GCS sources with their MD5s.

Run from the repository root: ``uv run python -m tests.golden.build_per_warc_goldens --refs <dir> [--only F1 ...]``
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from atlantic_cc.warc.identity import snapshot_of, warc_hash
from tests.parity.r1_asrun import gcs_md5, kept_path, read_kept, read_ref
from tests.parity.r1_compare import (
    COUNTERS_FILE,
    GOLDEN_ROOT,
    META_FILE,
    SURVIVORS_FILE,
    TEXT_HASH_HEX_DIGITS,
    text_sha256,
)

logger = logging.getLogger(__name__)

SURVIVOR_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("kept", pa.bool_()),
        ("text_sha256_prefix", pa.string()),
        ("fasttext_score", pa.float32()),
        ("n_tokens", pa.int32()),
        ("pooled_prob", pa.float32()),
        ("modernbert_prob", pa.float32()),
    ]
)


@dataclass(frozen=True)
class Fixture:
    label: str
    description: str
    warc_path: str
    shard: int
    warc_bytes: int


FIXTURES = (
    Fixture(
        "F1",
        "early crawl, smallest; 2M-ladder era (no Content-Length guard)",
        "s3://commoncrawl/crawl-data/CC-MAIN-2014-23/segments/1405997884573.18/warc/"
        "CC-MAIN-20140722025804-00181-ip-10-33-131-23.ec2.internal.warc.gz",
        5000,
        221_920_839,
    ),
    Fixture(
        "F2",
        "early crawl (2013), 8M segment",
        "s3://commoncrawl/crawl-data/CC-MAIN-2013-48/segments/1386165158218/warc/"
        "CC-MAIN-20131204135238-00069-ip-10-33-133-15.ec2.internal.warc.gz",
        31000,
        325_729_542,
    ),
    Fixture(
        "F3",
        "recent CC-MAIN, 8M segment",
        "s3://commoncrawl/crawl-data/CC-MAIN-2026-12/segments/1772687277590.81/warc/"
        "CC-MAIN-20260308022436-20260308052436-00349.warc.gz",
        10000,
        845_424_740,
    ),
    Fixture(
        "F4",
        "CC-NEWS extension",
        "s3://commoncrawl/crawl-data/CC-NEWS/2017/07/CC-NEWS-20170728033529-00050.warc.gz",
        34500,
        336_551_542,
    ),
    Fixture(
        "F5",
        "CC-SUPPLEMENTAL extension; one extractor crash",
        "s3://commoncrawl/projects/cc-open-athena-test/CC-SUPPLEMENTAL-2026-22/segments/20260526134233/warc/"
        "CC-SUPPLEMENTAL-2026-22-20260526134319-20260526174319-00066.warc.gz",
        32001,
        1_279_645_018,
    ),
    Fixture(
        "F6",
        "rung 1: the two-phase code path (no Content-Length guard)",
        "s3://commoncrawl/crawl-data/CC-MAIN-2013-48/segments/1386164021936/warc/"
        "CC-MAIN-20131204133341-00077-ip-10-33-133-15.ec2.internal.warc.gz",
        0,
        621_312_243,
    ),
)


def survivors_table(kept: pa.Table, tombstones: list[dict]) -> pa.Table:
    n_kept, n_tomb = kept.num_rows, len(tombstones)
    columns = {
        "doc_id": kept.column("doc_id").to_pylist() + [t["doc_id"] for t in tombstones],
        "kept": [True] * n_kept + [False] * n_tomb,
        "text_sha256_prefix": [text_sha256(t)[:TEXT_HASH_HEX_DIGITS] for t in kept.column("text").to_pylist()]
        + [None] * n_tomb,
        "fasttext_score": kept.column("fasttext_score").to_pylist() + [None] * n_tomb,
        "n_tokens": kept.column("n_tokens").to_pylist() + [None] * n_tomb,
        "pooled_prob": kept.column("pooled_prob").to_pylist() + [t["pooled_prob"] for t in tombstones],
        "modernbert_prob": kept.column("modernbert_prob").to_pylist() + [t["modernbert_prob"] for t in tombstones],
    }
    return pa.table(columns, schema=SURVIVOR_SCHEMA)


def build(fx: Fixture, out_root: Path, refs_dir: str) -> Path:
    ref = read_ref(refs_dir, warc_hash(fx.warc_path))
    h, kept, counters, tombstones = ref.warc_hash, read_kept(ref.warc_hash), ref.counters, ref.tombstones
    assert "a_download_failed" not in counters, counters["a_download_failed"]
    assert set(kept.column("warc_hash").to_pylist()) <= {h}, "kept rows of another WARC"
    assert set(kept.column("snapshot").to_pylist()) <= {snapshot_of(fx.warc_path)}, "unexpected snapshot"
    assert kept.num_rows == counters["n_kept"], (kept.num_rows, counters["n_kept"])
    assert kept.num_rows + len(tombstones) == counters["n_presurvivors"], "kept + tombstones != presurvivors"

    table = survivors_table(kept, tombstones)
    assert len(set(table.column("doc_id").to_pylist())) == table.num_rows, "duplicate doc_id"
    tomb_pooled = table.column("pooled_prob").to_pylist()[kept.num_rows :]
    assert [round(x, 6) for x in tomb_pooled] == [t["pooled_prob"] for t in tombstones], "float32 lost a value"

    out = out_root / h
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out / SURVIVORS_FILE, compression="zstd", compression_level=19)
    (out / COUNTERS_FILE).write_text(json.dumps({**dict(sorted(counters.items())), "raw": ref.raw}, indent=2) + "\n")
    meta = {
        "fixture": fx.label,
        "description": fx.description,
        "warc_path": fx.warc_path,
        "warc_hash": h,
        "snapshot": snapshot_of(fx.warc_path),
        "warc_bytes": fx.warc_bytes,
        "shard": fx.shard,
        "code_path": ref.code_path,
        "region": counters["region"],
        "device": counters["device"],
        "n_devices": counters["n_devices"],
        "processed_at": ref.processed_at,
        "content_length_guard": ref.content_length_guard,
        "n_kept": kept.num_rows,
        "n_tombstones": len(tombstones),
        "sources": {
            "kept": {"path": kept_path(h), "md5": gcs_md5(kept_path(h))},
            "tombstones": {"path": ref.tombstones_path, "md5": ref.tombstones_md5},
        },
    }
    (out / META_FILE).write_text(json.dumps(meta, indent=2) + "\n")
    size = (out / SURVIVORS_FILE).stat().st_size
    logger.info("%s %s: %d kept + %d tombstones -> %s (%d B)", fx.label, h, kept.num_rows, len(tombstones), out, size)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None, help="Fixture labels to build (default: all).")
    ap.add_argument("--out", type=Path, default=GOLDEN_ROOT)
    ap.add_argument("--refs", required=True, help="Directory of the fixtures' references (r1_phase_a refs).")
    args = ap.parse_args()
    for fx in FIXTURES:
        if args.only is None or fx.label in args.only:
            build(fx, args.out, args.refs)


if __name__ == "__main__":
    main()
