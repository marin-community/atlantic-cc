# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""R5 parity: recompute the per-crawl and global drop sets from the as-run sample and compare exactly.

Runs in-region (us-central2) on the cluster. Inputs are the as-run per-source sample
``SAMPLE_DIR/<source>/part-NNNNNN.parquet`` (``id, text``; 127 sources) and the ``full-val`` bloom;
expected outputs are the stored ``dropsets-full-val/_counts/<source>.parquet``,
``<source>/drop.parquet`` and ``_global/drop.parquet``.

    python -m atlantic_cc.decon.dropset_parity slice --slice 0 --num-slices 8 --workers 16
    python -m atlantic_cc.decon.dropset_parity global
"""

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import logging
import multiprocessing
import os
import time

import fsspec
import numpy as np
import pyarrow as pa

from atlantic_cc.decon import dropsets as ds
from atlantic_cc.decon.bloom import PyBloom, load_native_bloom, local_copy
from atlantic_cc.decon.matching import _bloom_hash, _extract_features
from atlantic_cc.decon.params import DROP_HASH_COUNT, GLOBAL_DROP_COUNT, MARK_NGRAM

logger = logging.getLogger(__name__)

POOL = "gs://marin-us-central2/documents/baseline_lpv11_fastpipe_v2_1_fused_deduped/8582115warcs"
DECON = f"{POOL}/decon_v4"
STORED_DROPSETS = f"{DECON}/dropsets-full-val"
BLOOM = f"{DECON}/bloom-full-val/filter.bin"
SAMPLE_DIR = "gs://marin-us-central2/tmp/ttl=14d/decon8m/source_sample_1m"
OUT = "gs://marin-us-central2/tmp/ttl=14d/atlantic_cc_repro/a5/dropsets_sample1m"
PYBLOOM_CHECK_DOCS = 50
DIFF_EXAMPLES = 20

_NATIVE = None
_PYBLOOM = None


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def _sha256_sorted(hashes: list[int]) -> str:
    return hashlib.sha256(np.array(sorted(hashes), dtype=np.uint64).tobytes()).hexdigest()


def _exists(path: str) -> bool:
    fs, p = fsspec.core.url_to_fs(path)
    return fs.exists(p)


def _write_json(path: str, obj: dict) -> None:
    with fsspec.open(path, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)


def _read_json(path: str) -> dict:
    with fsspec.open(path, "r") as fh:
        return json.load(fh)


def _sources(sample_dir: str) -> list[str]:
    fs, root = fsspec.core.url_to_fs(sample_dir.rstrip("/"))
    return sorted(os.path.basename(p) for p in fs.ls(root, detail=False))


def _compare_counts(recomputed: pa.Table, stored: pa.Table) -> dict:
    rh, rd = recomputed["hash"].to_pylist(), recomputed["df"].to_pylist()
    sh, sd = stored["hash"].to_pylist(), stored["df"].to_pylist()
    r, s = dict(zip(rh, rd, strict=True)), dict(zip(sh, sd, strict=True))
    out = {
        "rows_recomputed": len(rh),
        "rows_stored": len(sh),
        "equal_ordered": rh == sh and rd == sd,
        "equal_as_set": r == s,
        "schema_equal": recomputed.schema.equals(stored.schema),
    }
    if not out["equal_as_set"]:
        missing = [h for h in s if h not in r]
        extra = [h for h in r if h not in s]
        df_diff = [(h, r[h], s[h]) for h in r if h in s and r[h] != s[h]]
        out |= {
            "missing_in_recomputed": len(missing),
            "extra_in_recomputed": len(extra),
            "df_differs": len(df_diff),
            "examples_missing": [(h, s[h]) for h in missing[:DIFF_EXAMPLES]],
            "examples_extra": [(h, r[h]) for h in extra[:DIFF_EXAMPLES]],
            "examples_df": df_diff[:DIFF_EXAMPLES],
            "df_sum_recomputed": sum(rd),
            "df_sum_stored": sum(sd),
        }
    return out


def _compare_hash_lists(recomputed: list[int], stored: list[int]) -> dict:
    out = {
        "size_recomputed": len(recomputed),
        "size_stored": len(stored),
        "equal_ordered": recomputed == stored,
        "equal_as_set": set(recomputed) == set(stored),
        "sha256_sorted": _sha256_sorted(stored),
    }
    if not out["equal_as_set"]:
        out |= {
            "missing_in_recomputed": sorted(set(stored) - set(recomputed))[:DIFF_EXAMPLES],
            "extra_in_recomputed": sorted(set(recomputed) - set(stored))[:DIFF_EXAMPLES],
            "n_missing": len(set(stored) - set(recomputed)),
            "n_extra": len(set(recomputed) - set(stored)),
        }
    return out


def _pybloom_check(texts: list[str]) -> dict:
    """Pure-Python reader vs dupekit on every feature of the first docs (hits and misses)."""
    lookups = hits = mismatches = 0
    for text in texts:
        for feature in _extract_features(text, MARK_NGRAM):
            h = _bloom_hash(feature)
            native = h in _NATIVE
            mismatches += native != (h in _PYBLOOM)
            hits += native
            lookups += 1
    return {"docs": len(texts), "lookups": lookups, "native_hits": hits, "mismatches": mismatches}


def _run_source(args: tuple[str, str, str]) -> dict:
    source, sample_dir, out = args
    summary_path = f"{out}/summaries/{source}.json"
    if _exists(summary_path):
        return _read_json(summary_path)
    t0 = time.time()
    fs, root = fsspec.core.url_to_fs(sample_dir.rstrip("/"))
    listing = sorted(fs.ls(f"{root}/{source}", detail=True), key=lambda e: e["name"])
    listing = [e for e in listing if e["name"].endswith(".parquet")]
    parts = [f"gs://{e['name']}" for e in listing]
    t_list = time.time()

    first_texts: list[str] = []
    parts_read = 0

    def read(path: str) -> list[str | None]:
        return ds.read_texts(path)

    def tracked():
        nonlocal parts_read
        for texts in ds.prefetched(parts, read):
            parts_read += 1
            if len(first_texts) < PYBLOOM_CHECK_DOCS:
                first_texts.extend(t for t in texts[: PYBLOOM_CHECK_DOCS - len(first_texts)] if t)
            yield texts

    counts = ds.count_source(tracked(), _NATIVE)
    t_count = time.time()
    local = ds.local_drop_hashes(counts)
    table = ds.counts_table(counts)
    ds.write_parquet(f"{out}/dropsets-full-val/{ds.COUNTS_DIR}/{source}.parquet", table)
    ds.write_parquet(f"{out}/dropsets-full-val/{source}/{ds.DROP_FILE}", ds.drop_table(local))

    stored_counts = ds.read_parquet(f"{STORED_DROPSETS}/{ds.COUNTS_DIR}/{source}.parquet")
    stored_drop = ds.read_parquet(f"{STORED_DROPSETS}/{source}/{ds.DROP_FILE}")["hash"].to_pylist()
    summary = {
        "source": source,
        "parts_listed": len(parts),
        "parts_read": parts_read,
        "sample_bytes_listed": int(sum(e["size"] for e in listing)),
        "sample_max_time_created": max(str(e.get("timeCreated")) for e in listing) if listing else None,
        "sample_min_time_created": min(str(e.get("timeCreated")) for e in listing) if listing else None,
        "docs": counts.docs,
        "local_docs": counts.local_docs,
        "local_threshold": ds.local_threshold(counts.local_docs),
        "counts": _compare_counts(table, stored_counts),
        "drop": _compare_hash_lists(local, stored_drop),
        "pybloom": _pybloom_check(first_texts[:PYBLOOM_CHECK_DOCS]),
        "seconds_list": round(t_list - t0, 1),
        "seconds_count": round(t_count - t_list, 1),
        "seconds_total": round(time.time() - t0, 1),
        "finished": _now(),
        "pid": os.getpid(),
    }
    summary["equal"] = (
        summary["counts"]["equal_as_set"] and summary["drop"]["equal_as_set"] and summary["pybloom"]["mismatches"] == 0
    )
    _write_json(summary_path, summary)
    logger.info(
        "%s: docs %d, counts equal %s (ordered %s), drop equal %s, %.0fs",
        source,
        counts.docs,
        summary["counts"]["equal_as_set"],
        summary["counts"]["equal_ordered"],
        summary["drop"]["equal_as_set"],
        summary["seconds_total"],
    )
    return summary


def run_slice(slice_index: int, num_slices: int, workers: int, sample_dir: str, out: str, sources: list[str]) -> None:
    global _NATIVE, _PYBLOOM
    all_sources = _sources(sample_dir)
    chosen = sources or all_sources[slice_index::num_slices]
    unknown = sorted(set(chosen) - set(all_sources))
    if unknown:
        raise ValueError(f"not in the sample: {unknown}")
    logger.info("slice %d/%d: %d sources: %s", slice_index, num_slices, len(chosen), chosen)
    t0 = time.time()
    local_bloom = local_copy(BLOOM)
    _NATIVE = load_native_bloom(local_bloom)
    _PYBLOOM = PyBloom.open(local_bloom)
    logger.info("bloom loaded in %.0fs (%d bits, k=%d)", time.time() - t0, _NATIVE.size_in_bits, _PYBLOOM.k)
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(min(workers, len(chosen))) as pool:
        for summary in pool.imap_unordered(_run_source, [(s, sample_dir, out) for s in chosen]):
            logger.info("done %s equal=%s", summary["source"], summary["equal"])
    logger.info("slice %d finished in %.0fs", slice_index, time.time() - t0)


def _read_tables(paths: list[str]) -> list[pa.Table]:
    with concurrent.futures.ThreadPoolExecutor(32) as ex:
        return list(ex.map(ds.read_parquet, paths))


def run_global(sample_dir: str, out: str) -> None:
    sources = _sources(sample_dir)
    stored_global = ds.read_parquet(f"{STORED_DROPSETS}/{ds.GLOBAL_SOURCE}/{ds.DROP_FILE}")["hash"].to_pylist()
    report: dict = {"generated": _now(), "sources": len(sources), "stored_global_size": len(stored_global)}

    stored_counts = _read_tables([f"{STORED_DROPSETS}/{ds.COUNTS_DIR}/{s}.parquet" for s in sources])
    from_stored = ds.global_drop_hashes(stored_counts).to_pylist()
    report["global_from_stored_counts"] = _compare_hash_lists(from_stored, stored_global)

    summary_paths = [f"{out}/summaries/{s}.json" for s in sources]
    have = [p for p in summary_paths if _exists(p)]
    summaries = {}
    with concurrent.futures.ThreadPoolExecutor(32) as ex:
        for summary in ex.map(_read_json, have):
            summaries[summary["source"]] = summary
    report["per_source"] = summaries
    report["sources_recomputed"] = len(summaries)
    report["sources_equal"] = sum(s["equal"] for s in summaries.values())
    report["sources_counts_equal_ordered"] = sum(s["counts"]["equal_ordered"] for s in summaries.values())
    report["sources_drop_equal_ordered"] = sum(s["drop"]["equal_ordered"] for s in summaries.values())
    report["unequal_sources"] = sorted(s for s, v in summaries.items() if not v["equal"])
    report["pybloom_lookups"] = sum(s["pybloom"]["lookups"] for s in summaries.values())
    report["pybloom_mismatches"] = sum(s["pybloom"]["mismatches"] for s in summaries.values())

    if len(summaries) == len(sources):
        recomputed = _read_tables([f"{out}/dropsets-full-val/{ds.COUNTS_DIR}/{s}.parquet" for s in sources])
        from_recomputed = ds.global_drop_hashes(recomputed).to_pylist()
        report["global_from_recomputed_counts"] = _compare_hash_lists(from_recomputed, stored_global)
        ds.write_parquet(f"{out}/dropsets-full-val/{ds.GLOBAL_SOURCE}/{ds.DROP_FILE}", ds.drop_table(from_recomputed))

    # D from the stored component tables (the shipped artifact), and with the recomputed lists swapped in.
    stored_lists = _read_tables(ds.drop_tables_dirs(STORED_DROPSETS))
    pool = ds.pool_common_hashes(_read_tables(ds.partition_paths(f"{DECON}/boilerplate", "pool_totals")))
    code = ds.code_common_hashes(_read_tables(ds.partition_paths(f"{DECON}/boilerplate", "code_totals")))
    d_stored = ds.drop_set_union((t["hash"].to_pylist() for t in stored_lists), pool, code)
    report["D_from_stored"] = {
        "size": len(d_stored),
        "expected": DROP_HASH_COUNT,
        "sha256_sorted_uint64_le": hashlib.sha256(d_stored.astype("<u8").tobytes()).hexdigest(),
        "lists": len(stored_lists),
        "pooldf1000": len(pool),
        "codedf100": len(code),
    }
    if "global_from_recomputed_counts" in report:
        local_recomputed = _read_tables([f"{out}/dropsets-full-val/{s}/{ds.DROP_FILE}" for s in sources])
        d_re = ds.drop_set_union([*(t["hash"].to_pylist() for t in local_recomputed), from_recomputed], pool, code)
        report["D_from_recomputed_dropsets"] = {
            "size": len(d_re),
            "equal_to_stored_D": bool(np.array_equal(d_re, d_stored)),
        }
    report["global_expected_size"] = GLOBAL_DROP_COUNT
    _write_json(f"{out}/report.json", report)
    logger.info(
        "global: from stored counts equal=%s; recomputed %d/%d sources, %d equal; D=%d",
        report["global_from_stored_counts"]["equal_as_set"],
        len(summaries),
        len(sources),
        report["sources_equal"],
        len(d_stored),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(processName)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["slice", "global"])
    ap.add_argument("--slice", type=int, default=0)
    ap.add_argument("--num-slices", type=int, default=1)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sources", nargs="*", default=[])
    ap.add_argument("--sample-dir", default=SAMPLE_DIR)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if args.step == "slice":
        run_slice(args.slice, args.num_slices, args.workers, args.sample_dir, args.out, args.sources)
    else:
        run_global(args.sample_dir, args.out)


if __name__ == "__main__":
    main()
