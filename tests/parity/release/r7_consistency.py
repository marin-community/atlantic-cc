# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""R7: the published release against production's per-WARC cascade output, on a seeded sample.

For sampled release row groups (plus every row of a few small files), each record must exist in its
WARC's ``kept/data-<warc_hash>.parquet`` with the same ``url`` and crawl, ``text`` equal to the kept
text after dedup's whitespace compaction (runs of more than 128 whitespace characters cut to 128),
and scores equal to the kept float32 values (``modernbert_prob`` NaN -> null). It also checks:
``xxh3_128(text)`` is unique across the sample; clean files have empty ``decon_matches`` and
contaminated files non-empty ones; within a file each WARC's records are contiguous and WARCs appear
in ascending ``warc_hash`` order. Run in-region (us-central2); writes a JSON report.
"""

import argparse
import json
import logging
import math
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import xxhash

from atlantic_cc.release import storage
from atlantic_cc.release.layout import ReleaseFile, release_files
from atlantic_cc.release.params import VERSION, Kind, manifest_name
from atlantic_cc.warc.identity import warc_hash
from atlantic_cc.worklist.identity import to_s3_url

logger = logging.getLogger(__name__)

RELEASE = "gs://marin-us-central2/documents/baseline_lpv11_fastpipe_v2_1_fused_deduped/8582115warcs/release_1gb"
KEPT = "gs://marin-us-central2/documents/fast_curation/lpv11_fastpipe_v2_1_fused-2a076ddfcb/kept"
ALWAYS = ("CC-NEWS-2016-08", "CC-SUPPLEMENTAL-2026-22")  # every row of these crawls' contaminated files
KEPT_COLUMNS = ["doc_id", "url", "snapshot", "text", "fasttext_score", "pooled_prob", "modernbert_prob"]
# rdedup's compact_whitespace (as-run experiments/fast_curation/rdedup/docs.py:48-64).
MAX_WHITESPACE_RUN_CHARS = 128
_LONG_WHITESPACE = re.compile(r"\s{" + str(MAX_WHITESPACE_RUN_CHARS + 1) + r",}")


def compact_whitespace(text: str) -> str:
    if _LONG_WHITESPACE.search(text) is None:
        return text
    return _LONG_WHITESPACE.sub(lambda m: m.group(0)[:MAX_WHITESPACE_RUN_CHARS], text)


def _kept(path: str, doc_ids: list[str]) -> dict[str, dict]:
    """The kept rows of one WARC with these doc ids (only those rows become Python objects)."""
    table = storage.read_parquet(f"{KEPT}/data-{warc_hash(to_s3_url(path))}.parquet", columns=KEPT_COLUMNS)
    table = table.filter(pc.is_in(table.column("doc_id"), value_set=pa.array(doc_ids, pa.string())))
    return {row["doc_id"]: row for row in table.to_pylist()}


def _same_prob(release_value: float | None, kept_value: float | None) -> bool:
    """Kept NaN (or null) is released as null; anything else must be equal."""
    if kept_value is None or math.isnan(kept_value):
        return release_value is None
    return release_value == kept_value


def check_rows(rows: list[dict], kind: Kind, kept_by_warc: dict[str, dict[str, dict]], problems: Counter) -> None:
    for row in rows:
        kept = kept_by_warc[row["warc_path"]].get(row["record_id"])
        if kept is None:
            problems["record_not_in_kept"] += 1
            continue
        problems["url_mismatch"] += row["url"] != kept["url"]
        problems["crawl_mismatch"] += row["crawl"] != kept["snapshot"]
        problems["text_mismatch"] += row["text"] != compact_whitespace(kept["text"])
        problems["text_compacted"] += row["text"] != kept["text"]  # informational
        problems["fasttext_mismatch"] += row["fasttext_score"] != kept["fasttext_score"]
        problems["pooled_mismatch"] += row["pooled_prob"] != kept["pooled_prob"]
        problems["modernbert_mismatch"] += not _same_prob(row["modernbert_prob"], kept["modernbert_prob"])
        problems["bad_counts"] += row["exact_copies"] < 1 or row["fuzzy_cluster_size"] < 1
        problems["decon_matches_wrong_for_kind"] += bool(row["decon_matches"]) != (kind is Kind.CONTAMINATED)


def warc_order(rows: list[dict]) -> tuple[bool, bool]:
    """(each WARC's records contiguous, WARCs in ascending warc_hash order) for consecutive rows."""
    runs = [rows[0]["warc_path"]] if rows else []
    for row in rows[1:]:
        if row["warc_path"] != runs[-1]:
            runs.append(row["warc_path"])
    hashes = [warc_hash(to_s3_url(p)) for p in runs]
    return len(set(runs)) == len(runs), hashes == sorted(hashes)


def sample_file(f: ReleaseFile, rng: random.Random, row_groups: int, every_row: bool, threads: int) -> dict:
    with storage.open_file(f"{RELEASE}/{f.path}", "rb") as fh:
        pf = pq.ParquetFile(fh)
        n = pf.metadata.num_row_groups
        chosen = list(range(n)) if every_row else sorted(rng.sample(range(n), min(row_groups, n)))
        groups = [pf.read_row_group(i).to_pylist() for i in chosen]
    rows = [r for g in groups for r in g]
    by_warc: dict[str, list[dict]] = {}
    for row in rows:
        by_warc.setdefault(row["warc_path"], []).append(row)

    def check_warc(item: tuple[str, list[dict]]) -> Counter:
        warc, warc_rows = item
        problems: Counter = Counter()
        check_rows(warc_rows, f.kind, {warc: _kept(warc, [r["record_id"] for r in warc_rows])}, problems)
        return problems

    problems: Counter = Counter()
    with ThreadPoolExecutor(threads) as ex:
        for counts in ex.map(check_warc, by_warc.items()):
            problems.update(counts)
    order = [warc_order(g) for g in groups]
    return {
        "path": f.path,
        "row_groups_read": len(chosen),
        "rows": len(rows),
        "warcs": len(by_warc),
        "problems": dict(problems),
        "warcs_contiguous": all(c for c, _ in order),
        "warcs_ascending_hash": all(a for _, a in order),
        "text_hashes": [xxhash.xxh3_128_hexdigest(r["text"].encode()) for r in rows],
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", required=True)
    ap.add_argument("--files", type=int, default=40, help="Random release files to sample.")
    ap.add_argument("--row-groups", type=int, default=2, help="Random row groups per sampled file.")
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    start = time.time()
    manifest = storage.read_json(f"{RELEASE}/{manifest_name(VERSION)}")
    files = release_files(manifest, None, list(Kind))
    rng = random.Random(a.seed)
    always = [f for f in files if f.crawl in ALWAYS and f.kind is Kind.CONTAMINATED] + [
        f for f in files if f.crawl == "CC-NEWS-2016-08"
    ]
    sampled = sorted(set(rng.sample(files, a.files)) - set(always), key=lambda f: f.path)
    results = []
    for f in [*dict.fromkeys(always), *sampled]:
        results.append(sample_file(f, rng, a.row_groups, f in always, a.threads))
        logger.info("%s: %s", f.path, {k: v for k, v in results[-1].items() if k != "text_hashes"})
    hashes = [h for r in results for h in r.pop("text_hashes")]
    totals: Counter = Counter()
    for r in results:
        totals.update(r["problems"])
    failures = {k: v for k, v in totals.items() if v and k != "text_compacted"}
    report = {
        "check": "r7",
        "release": RELEASE,
        "seed": a.seed,
        "files": len(results),
        "crawls": len({r["path"].split("/")[1] for r in results}),
        "rows": sum(r["rows"] for r in results),
        "warcs": sum(r["warcs"] for r in results),
        "problems": dict(totals),
        "text_xxh3_128_duplicates": len(hashes) - len(set(hashes)),
        "all_warcs_contiguous": all(r["warcs_contiguous"] for r in results),
        "all_warcs_ascending_hash": all(r["warcs_ascending_hash"] for r in results),
        "ok": not failures and len(hashes) == len(set(hashes)),
        "wall_seconds": round(time.time() - start, 1),
        "per_file": results,
    }
    text = json.dumps(report, indent=1)
    storage.write_bytes(a.report, text.encode())
    print(json.dumps({k: v for k, v in report.items() if k != "per_file"}, indent=1))


if __name__ == "__main__":
    main()
