# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""R6: the release regroup port against production's intermediate and published files.

Run in-region (us-central2) on the Marin cluster; every check only reads production's files and writes
under ``--scratch``. Each check writes a JSON report to ``--report`` and prints it:

* ``plan``: list production's pieces with sizes, re-plan the parts, compare with ``parts_plan.json``;
* ``manifest``: rebuild the manifest from the plan, the part footers of ``release_1gb/`` and the decon
  ``export.json``, and the paths files with v1.0's gzip timestamp; compare bytes with the published ones;
* ``parts``: rebuild selected parts from production's pieces; compare with the published parts
  (MD5 of the bytes, footer facts, and row group by row group);
* ``pieces``: rebuild ``warc_paths.parquet`` from the frozen worklist and ``decon_lookup.parquet`` from
  ``removed_docs.parquet``, then the pieces of one batch from ``deduped_c075/``; compare each file with
  production's.
"""

import argparse
import base64
import gzip
import hashlib
import json
import logging
import time

import fsspec
import pyarrow.parquet as pq

from atlantic_cc.release import regroup, storage
from atlantic_cc.release.params import (
    DECON_EXPORT_FILE,
    DECON_LOOKUP_FILE,
    PARTS_PLAN_FILE,
    V1_0_PATHS_GZIP_MTIME,
    VERSION,
    WARC_PATHS_FILE,
    ZSTD_LEVEL,
    manifest_name,
)

logger = logging.getLogger(__name__)

POOL = "gs://marin-us-central2/documents/baseline_lpv11_fastpipe_v2_1_fused_deduped/8582115warcs"
ASRUN_STAGE = "gs://marin-us-central2/tmp/ttl=14d/release_regroup_v3"
ASRUN_RELEASE = f"{POOL}/release_1gb"
ASRUN_SOURCE = f"{POOL}/deduped_c075"
ASRUN_DECON_RUN = f"{POOL}/decon_v4/filter-full-val-pooldf1000-codedf100"
MD5_CHUNK = 8 << 20


def _md5(path: str) -> str:
    """MD5 of an object: from GCS metadata when present, else by reading it."""
    fs, stripped = fsspec.core.url_to_fs(path)
    info = fs.info(stripped)
    if info.get("md5Hash"):
        return base64.b64decode(info["md5Hash"]).hex()
    digest = hashlib.md5()
    with fs.open(stripped, "rb") as f:
        while chunk := f.read(MD5_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _footer(path: str) -> dict:
    with storage.open_file(path, "rb") as f:
        md = pq.ParquetFile(f).metadata
        return {
            "num_rows": md.num_rows,
            "num_row_groups": md.num_row_groups,
            "created_by": md.created_by,
            "row_group_rows": sorted({md.row_group(i).num_rows for i in range(md.num_row_groups)}),
            "compression": md.row_group(0).column(0).compression if md.num_row_groups else None,
            "schema": str(pq.ParquetFile(f).schema_arrow),
        }


def compare_parquet(mine: str, theirs: str) -> dict:
    """Byte, footer and row-group-by-row-group comparison of two Parquet files."""
    result = {"mine": mine, "theirs": theirs, "md5_mine": _md5(mine), "md5_theirs": _md5(theirs)}
    result["bytes_identical"] = result["md5_mine"] == result["md5_theirs"]
    result["footer_mine"], result["footer_theirs"] = _footer(mine), _footer(theirs)
    differing = []
    with storage.open_file(mine, "rb") as fa, storage.open_file(theirs, "rb") as fb:
        a, b = pq.ParquetFile(fa), pq.ParquetFile(fb)
        result["schemas_equal"] = a.schema_arrow.equals(b.schema_arrow)
        if a.metadata.num_row_groups == b.metadata.num_row_groups:
            for i in range(a.metadata.num_row_groups):
                if not a.read_row_group(i).equals(b.read_row_group(i)):
                    differing.append(i)
        else:
            ta, tb = a.read(), b.read()
            result["tables_equal"] = ta.equals(tb)
    result["row_groups_differing"] = differing
    result["rows_identical"] = result.get("tables_equal", not differing) and result["schemas_equal"]
    return result


def check_plan(asrun_stage: str, part_mb: int) -> dict:
    start = time.time()
    listing = regroup.glob_pieces(asrun_stage)
    listed = time.time()
    plan = regroup.plan_parts(listing, asrun_stage, part_mb * 1_000_000)
    asrun_bytes = storage.read_bytes(f"{asrun_stage}/{PARTS_PLAN_FILE}")
    asrun_plan = json.loads(asrun_bytes)
    first_diff = next((i for i, (x, y) in enumerate(zip(plan, asrun_plan, strict=False)) if x != y), None)
    return {
        "check": "plan",
        "asrun_stage": asrun_stage,
        "part_mb": part_mb,
        "pieces_listed": len(listing),
        "piece_bytes": sum(n for _, n in listing),
        "listing_seconds": round(listed - start, 1),
        "parts_planned": len(plan),
        "parts_asrun": len(asrun_plan),
        "plan_equal": plan == asrun_plan,
        "plan_json_bytes_equal": json.dumps(plan).encode() == asrun_bytes,
        "first_difference": first_diff,
        "crawls": len({p["crawl"] for p in plan}),
    }


def check_manifest(asrun_stage: str, release: str, decon_run: str, scratch: str) -> dict:
    plan = storage.read_json(f"{asrun_stage}/{PARTS_PLAN_FILE}")
    rows = regroup.part_rows(release, VERSION, plan)
    export = storage.read_json(f"{decon_run}/{DECON_EXPORT_FILE}")
    document = regroup.manifest_document(plan, rows, VERSION, export, decon_run)
    published = storage.read_bytes(f"{release}/{manifest_name(VERSION)}")
    report = {
        "check": "manifest",
        "files": len(plan),
        "records": sum(rows),
        "by_kind": document["by_kind"],
        "manifest_bytes_equal": regroup.manifest_bytes(document) == published,
        "manifest_equal": document == json.loads(published),
        "paths_files": {},
    }
    for name, data in regroup.paths_files(plan, VERSION, V1_0_PATHS_GZIP_MTIME).items():
        theirs = storage.read_bytes(f"{release}/{name}")
        report["paths_files"][name] = {
            "gzip_bytes_equal": data == theirs,
            "content_equal": gzip.decompress(data) == gzip.decompress(theirs),
            "lines": len(gzip.decompress(data).splitlines()),
        }
    # Per-file record counts: a by-product worth keeping next to the checksums.
    keys = regroup.published_keys(plan, VERSION)
    storage.write_bytes(
        f"{scratch}/manifest_check/part_rows.json", json.dumps(dict(zip(keys, rows, strict=True))).encode()
    )
    return report


def check_parts(asrun_stage: str, release: str, scratch: str, selectors: list[str]) -> dict:
    plan = storage.read_json(f"{asrun_stage}/{PARTS_PLAN_FILE}")
    out = f"{scratch}/parts_rebuild"
    results = []
    for selector in selectors:
        crawl, kind, index = selector.split("/")
        part = next(p for p in plan if p["crawl"] == crawl and p["kind"] == kind and p["part"] == int(index))
        start = time.time()
        written = regroup.write_part(part, out, VERSION, ZSTD_LEVEL)
        result = compare_parquet(regroup.part_key(out, VERSION, part), regroup.part_key(release, VERSION, part))
        result.update(
            selector=selector,
            pieces=len(part["pieces"]),
            records=written["records"],
            seconds=round(time.time() - start, 1),
        )
        results.append(result)
        logger.info("%s: %s", selector, {k: result[k] for k in ("bytes_identical", "rows_identical", "records")})
    return {"check": "parts", "results": results, "all_bytes_identical": all(r["bytes_identical"] for r in results)}


def check_pieces(worklist: str, asrun_stage: str, source: str, decon_run: str, scratch: str, batch_id: int) -> dict:
    stage = f"{scratch}/stage"
    report: dict = {"check": "pieces", "batch": batch_id}
    if not storage.exists(f"{stage}/{WARC_PATHS_FILE}"):
        regroup.write_warc_paths(worklist, stage)
    report["warc_paths"] = compare_parquet(f"{stage}/{WARC_PATHS_FILE}", f"{asrun_stage}/{WARC_PATHS_FILE}")
    if not storage.exists(f"{stage}/{DECON_LOOKUP_FILE}"):
        regroup.write_decon_lookup(decon_run, stage)
    report["decon_lookup"] = compare_parquet(f"{stage}/{DECON_LOOKUP_FILE}", f"{asrun_stage}/{DECON_LOOKUP_FILE}")
    start = time.time()
    regroup.pieces(source, decon_run, stage, [], 1, batch_ids=[batch_id])
    report["seconds"] = round(time.time() - start, 1)
    mine_marker = storage.read_json(regroup.done_marker_path(stage, batch_id))
    asrun_marker = storage.read_json(regroup.done_marker_path(asrun_stage, batch_id))
    report["marker"] = {k: (mine_marker[k] == asrun_marker[k]) for k in ("files", "records", "contaminated", "pieces")}
    report["records"], report["contaminated"] = mine_marker["records"], mine_marker["contaminated"]
    report["pieces"] = []
    for piece in asrun_marker["pieces"]:
        crawl, kind = piece.split("/")
        report["pieces"].append(
            compare_parquet(
                regroup.piece_path(stage, crawl, kind, batch_id), regroup.piece_path(asrun_stage, crawl, kind, batch_id)
            )
        )
    report["pieces_bytes_identical"] = sum(p["bytes_identical"] for p in report["pieces"])
    report["pieces_rows_identical"] = sum(p["rows_identical"] for p in report["pieces"])
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("check", choices=["plan", "manifest", "parts", "pieces"])
    ap.add_argument("--scratch", required=True, help="Writable prefix for rebuilt files.")
    ap.add_argument("--report", required=True)
    ap.add_argument("--asrun-stage", default=ASRUN_STAGE)
    ap.add_argument("--release", default=ASRUN_RELEASE)
    ap.add_argument("--source", default=ASRUN_SOURCE)
    ap.add_argument("--decon-run", default=ASRUN_DECON_RUN)
    ap.add_argument("--worklist", help="Frozen worklist (pieces check).")
    ap.add_argument("--part-mb", type=int, default=1000)
    ap.add_argument(
        "--parts", nargs="+", default=["CC-NEWS-2016-08/part/0", "CC-MAIN-2013-20/part/0"], help="crawl/kind/index"
    )
    ap.add_argument("--batch", type=int, default=3140)
    a = ap.parse_args()
    start = time.time()
    if a.check == "plan":
        report = check_plan(a.asrun_stage, a.part_mb)
    elif a.check == "manifest":
        report = check_manifest(a.asrun_stage, a.release, a.decon_run, a.scratch)
    elif a.check == "parts":
        report = check_parts(a.asrun_stage, a.release, a.scratch, a.parts)
    else:
        report = check_pieces(a.worklist, a.asrun_stage, a.source, a.decon_run, a.scratch, a.batch)
    report["wall_seconds"] = round(time.time() - start, 1)
    text = json.dumps(report, indent=1, default=str)
    storage.write_bytes(a.report, text.encode())
    print(text)


if __name__ == "__main__":
    main()
