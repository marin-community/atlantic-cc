# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Maintainer checks of the port against the as-run decontamination outputs (Marin GCS, in-region).

``shards``: for deduplicated pool shards, recompute (a) the mark with the ``full-val`` bloom and with S',
compared row by row with ``attributes-full-val/outputs/main/<shard>.parquet`` (``contaminated``,
``max_overlap``, ``matched_hashes``), and (b) the single-pass filter with D and the bloom or S', compared
with the removed-doc sidecar ``removed_parts/<shard>.parquet`` and the ``matched_evals`` of
``removed_docs.parquet``.

    python -m atlantic_cc.decon.verify shards --out gs://.../shards.json data-012345-019 data-000000-000
"""

import argparse
import datetime
import hashlib
import json
import logging
import multiprocessing
import time

import dupekit_native
import fsspec
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from atlantic_cc.decon import dropsets as ds
from atlantic_cc.decon.artifacts import (
    AS_RUN_BLOOM,
    AS_RUN_DECON,
    AS_RUN_INDEXES,
    AS_RUN_POOL,
    EVAL_HIT_NAMES_FILE,
    EVAL_HITS_FILE,
    load_drop_hashes,
    load_eval_hit_names,
    load_eval_hits,
)
from atlantic_cc.decon.bloom import load_native_bloom, local_copy
from atlantic_cc.decon.eval_index import IndexDigest, eval_record_files, file_digest, full_val_capacity
from atlantic_cc.decon.filter import mark_matches, removal_matches
from atlantic_cc.decon.names import decon_matches
from atlantic_cc.decon.params import FALSE_POSITIVE_RATE
from atlantic_cc.decon.pool_steps import read_jsonl

logger = logging.getLogger(__name__)

FILTER_RUN = f"{AS_RUN_DECON}/filter-full-val-pooldf1000-codedf100"
ATTRIBUTES = f"{AS_RUN_DECON}/attributes-full-val/outputs/main"
DEDUPED = f"{AS_RUN_POOL}/deduped_c075"
DEFAULT_SHARDS = ("data-012345-019", "data-012345-002", "data-012345-011", "data-000000-000")
MAX_EXAMPLES = 10

_STATE: dict = {}


def _shard_task(args: tuple[str, str]) -> dict:
    shard, mode = args
    membership = _STATE["bloom"] if mode.endswith("bloom") else _STATE["eval_hits"]
    t0 = time.time()
    out: dict = {"shard": shard, "mode": mode}
    rows = []
    docs = 0
    for record in read_jsonl(f"{DEDUPED}/{shard}.jsonl.gz"):
        docs += 1
        if mode.startswith("mark"):
            score, matched = mark_matches(record["text"], membership)
            rows.append((record["doc_id"], score, matched))
        else:
            matched = removal_matches(record["text"], membership, _STATE["drop"])
            if matched:
                rows.append((record["doc_id"], matched))
    out["docs"] = docs
    if mode.startswith("mark"):
        attrs = ds.read_parquet(f"{ATTRIBUTES}/{shard}.parquet")
        a_id, a_c = attrs["id"].to_pylist(), attrs["contaminated"].to_pylist()
        a_s, a_m = attrs["max_overlap"].to_pylist(), attrs["matched_hashes"].to_pylist()
        ids_equal = [r[0] for r in rows] == a_id
        bad_matched = [i for i, r in enumerate(rows) if r[2] != a_m[i]] if ids_equal else []
        bad_flag = [i for i, r in enumerate(rows) if bool(r[2]) != a_c[i]] if ids_equal else []
        bad_score = [i for i, r in enumerate(rows) if r[1] != a_s[i]] if ids_equal else []
        out |= {
            "attribute_rows": attrs.num_rows,
            "ids_in_order_equal": ids_equal,
            "candidates": sum(a_c),
            "contaminated_mismatches": len(bad_flag),
            "matched_hashes_mismatches": len(bad_matched),
            "max_overlap_mismatches": len(bad_score),
            "examples": [
                {"id": a_id[i], "stored": a_m[i], "recomputed": rows[i][2]} for i in bad_matched[:MAX_EXAMPLES]
            ],
        }
        out["equal"] = ids_equal and not bad_flag and not bad_matched and (mode == "mark-eval-hits" or not bad_score)
    else:
        removed = ds.read_parquet(f"{FILTER_RUN}/removed_parts/{shard}.parquet")
        stored = dict(zip(removed["doc_id"].to_pylist(), removed["matched_hashes"].to_pylist(), strict=True))
        recomputed = dict(rows)
        names = _STATE["names"]
        out |= {
            "stored_removed": len(stored),
            "recomputed_removed": len(recomputed),
            "removed_equal": recomputed == stored,
            "recomputed_decon_matches": {i: decon_matches(m, names) for i, m in recomputed.items()},
        }
        out["equal"] = out["removed_equal"]
    out["seconds"] = round(time.time() - t0, 1)
    return out


def verify_shards(shards: list[str], artifacts: str, workers: int) -> dict:
    local_bloom = local_copy(AS_RUN_BLOOM)
    _STATE.update(
        bloom=load_native_bloom(local_bloom),
        eval_hits=load_eval_hits(f"{artifacts}/{EVAL_HITS_FILE}"),
        names=load_eval_hit_names(f"{artifacts}/{EVAL_HIT_NAMES_FILE}"),
        drop=load_drop_hashes(),
    )
    tasks = [(s, m) for s in shards for m in ("mark-bloom", "mark-eval-hits", "filter-bloom", "filter-eval-hits")]
    with multiprocessing.get_context("fork").Pool(min(workers, len(tasks))) as pool:
        results = list(pool.imap_unordered(_shard_task, tasks))

    stored_evals = _stored_matched_evals(shards)
    fs, _ = fsspec.core.url_to_fs(DEDUPED)
    inputs = {s: fs.info(f"{DEDUPED}/{s}.jsonl.gz") for s in shards}
    for r in results:
        r["input_bytes"] = inputs[r["shard"]]["size"]
        r["input_md5"] = inputs[r["shard"]].get("md5Hash")
        if r["mode"].startswith("filter"):
            names = r.pop("recomputed_decon_matches")
            stored = {i: stored_evals[i] for i in names if i in stored_evals}
            r["decon_matches_equal"] = names == stored
            r["equal"] = r["equal"] and r["decon_matches_equal"]
    return {
        "generated": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "shards": shards,
        "results": sorted(results, key=lambda r: (r["shard"], r["mode"])),
        "all_equal": all(r["equal"] for r in results),
    }


def _stored_matched_evals(shards: list[str]) -> dict[str, list[str]]:
    files = {f"{s}.jsonl.gz" for s in shards}
    with fsspec.open(f"{FILTER_RUN}/removed_docs.parquet", "rb") as fh:
        table = pq.read_table(fh, columns=["doc_id", "source_file", "matched_evals"])
    table = table.filter(pc.is_in(table["source_file"], value_set=pc.cast(list(files), "string")))
    return dict(zip(table["doc_id"].to_pylist(), table["matched_evals"].to_pylist(), strict=True))


EVALS = "gs://marin-us-central2/datakit/decontam/evals"
FILL_ROOT = f"{EVALS}/aa-index-v4.1.1-v3-fill"
VAL_ROOT = f"{EVALS}/val-sets-paloma16-uncheatable7-lima"


def _file_digest_task(args: tuple[str, str]) -> tuple:
    path, label = args
    return file_digest(path, label)


def verify_eval_index(out_dir: str, workers: int) -> dict:
    """Rebuild the full-val index and bloom from the staged as-run corpus; compare; write corpus digests."""
    t0 = time.time()
    full_files = eval_record_files(FILL_ROOT)
    fs, val_root = fsspec.core.url_to_fs(VAL_ROOT)
    val_files = sorted(f"gs://{p}" for p in fs.glob(f"{val_root}/*/eval.parquet"))
    labelled = [(p, "full/" + p.removeprefix(FILL_ROOT + "/")) for p in full_files]
    labelled += [(p, "val/" + p.removeprefix(VAL_ROOT + "/")) for p in val_files]
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        results = pool.map(_file_digest_task, labelled, chunksize=4)
    logger.info("eval-index: digested %d files (%.0fs)", len(results), time.time() - t0)

    full_digest, val_digest = IndexDigest(), IndexDigest()
    full_rows = val_rows = 0
    for (_, label), (summary, _records, hash_sets) in zip(labelled, results, strict=True):
        digest = full_digest if label.startswith("full/") else val_digest
        for eval_id, hashes in hash_sets:
            digest.update(hashes, [eval_id] * hashes.size)
        if label.startswith("full/"):
            full_rows += summary["index_rows"]
        else:
            val_rows += summary["index_rows"]
    stored_digests = {}
    for name, index in zip(("full", "val"), AS_RUN_INDEXES, strict=True):
        d = IndexDigest()
        with fsspec.open(index, "rb") as fh:
            for batch in pq.ParquetFile(fh).iter_batches(columns=["hash", "eval_id"], batch_size=1_000_000):
                d.update_batch(batch)
        stored_digests[name] = d.as_dict()
    logger.info("eval-index: index digests done (%.0fs)", time.time() - t0)

    capacity = full_val_capacity(full_rows, val_rows)
    bf = dupekit_native.Bloom(capacity, FALSE_POSITIVE_RATE)
    for _, (_summary, _records, hash_sets) in zip(labelled, results, strict=True):
        for _, hashes in hash_sets:
            for h in hashes.tolist():
                bf.add(h)
    rebuilt = hashlib.sha256(bf.save_bytes()).hexdigest()
    with fsspec.open(AS_RUN_BLOOM, "rb") as fh:
        stored = hashlib.sha256()
        while chunk := fh.read(64 * 1024 * 1024):
            stored.update(chunk)
    logger.info("eval-index: bloom rebuilt (%.0fs)", time.time() - t0)

    summaries = [r[0] for r in results]
    records = [row for r in results for row in r[1]]
    table = pa.table(
        {
            "file": pa.array([r[0] for r in records], pa.string()).dictionary_encode(),
            "id": [r[1] for r in records],
            "text_sha256": pa.array([r[2] for r in records], pa.binary(32)),
        }
    )
    with fsspec.open(f"{out_dir}/eval_record_digests.parquet", "wb") as fh:
        pq.write_table(table, fh, compression="zstd")
    with fsspec.open(f"{out_dir}/eval_file_digests.json", "w") as fh:
        json.dump(summaries, fh, indent=1)
    report = {
        "generated": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "full_files": len(full_files),
        "val_files": len(val_files),
        "records": len(records),
        "full_index_rows": full_rows,
        "val_index_rows": val_rows,
        "capacity": capacity,
        "index_digest_rebuilt": {"full": full_digest.as_dict(), "val": val_digest.as_dict()},
        "index_digest_stored": stored_digests,
        "index_equal": {
            "full": full_digest.as_dict() == stored_digests["full"],
            "val": val_digest.as_dict() == stored_digests["val"],
        },
        "bloom_sha256_rebuilt": rebuilt,
        "bloom_sha256_stored": stored.hexdigest(),
        "bloom_equal": rebuilt == stored.hexdigest(),
        "seconds": round(time.time() - t0, 1),
    }
    with fsspec.open(f"{out_dir}/eval_index_report.json", "w") as fh:
        json.dump(report, fh, indent=1)
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(processName)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    p = sub.add_parser("shards")
    p.add_argument("shards", nargs="*", default=list(DEFAULT_SHARDS))
    p.add_argument("--artifacts", required=True, help=f"Directory with {EVAL_HITS_FILE} and {EVAL_HIT_NAMES_FILE}.")
    p.add_argument("--out", required=True, help="Report JSON path (fsspec URL).")
    p.add_argument("--workers", type=int, default=16)
    p = sub.add_parser("eval-index", help="Rebuild the full-val index and bloom from the staged corpus.")
    p.add_argument("--out", required=True, help="Output directory (fsspec URL).")
    p.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    if args.step == "eval-index":
        report = verify_eval_index(args.out.rstrip("/"), args.workers)
        logger.info("eval-index: %s", {k: report[k] for k in ("index_equal", "bloom_equal", "records")})
        return
    report = verify_shards(args.shards, args.artifacts.rstrip("/"), args.workers)
    with fsspec.open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    logger.info("shards: all_equal=%s", report["all_equal"])


if __name__ == "__main__":
    main()
