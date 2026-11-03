# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``atlantic-cc worklist ...``: rebuild, check and lay out the input WARC list."""

import argparse
import json
import logging

import pyarrow as pa
import pyarrow.parquet as pq

from atlantic_cc.release import storage
from atlantic_cc.warc.identity import warc_hash
from atlantic_cc.worklist import asrun
from atlantic_cc.worklist.crawls import NEWS_LAST_WARC, V1_0_CRAWLS
from atlantic_cc.worklist.frozen import (
    V1_0_FILE_NAME,
    crawl_digests,
    load_frozen,
    read_paths_file,
    write_paths_file,
)
from atlantic_cc.worklist.identity import to_s3_url
from atlantic_cc.worklist.layout import v1_0_layout
from atlantic_cc.worklist.listings import build_worklist, mismatched_crawls, rebuild_v1_0

logger = logging.getLogger(__name__)

LAYOUT_SCHEMA = pa.schema([("shard", pa.int32()), ("warc_path", pa.string()), ("warc_hash", pa.string())])
EMPTY_WARCS_FILE = "v1.0.input.empty_warcs.tsv.gz"
CATALOG_FILE = "v1.0.input.warc_catalog.parquet"


def _rebuild(args: argparse.Namespace) -> None:
    paths = rebuild_v1_0()
    write_paths_file(paths, args.out)
    logger.info("wrote the v1.0 worklist (%d WARCs, digest checked) to %s", len(paths), args.out)


def _build(args: argparse.Namespace) -> None:
    paths = build_worklist(args.crawls, args.news_last_warc)
    write_paths_file(paths, args.out)
    logger.info("wrote %d WARCs of %d crawls to %s", len(paths), len(args.crawls), args.out)


def _verify(args: argparse.Namespace) -> None:
    frozen = load_frozen(args.worklist)  # raises unless it is exactly the frozen list
    rebuilt = build_worklist(V1_0_CRAWLS, NEWS_LAST_WARC)
    report = {
        "worklist": args.worklist,
        "n_warcs": len(frozen),
        "crawls": len(crawl_digests(frozen)),
        "digest": "ok",
        "identical_to_public_listings": rebuilt == frozen,
        "crawls_differing_from_listings": mismatched_crawls(rebuilt),
    }
    print(json.dumps(report, indent=1))
    if not report["identical_to_public_listings"]:
        raise SystemExit(1)


def _layout(args: argparse.Namespace) -> None:
    shards = v1_0_layout(read_paths_file(args.worklist))
    rows = [(s, p) for s, paths in enumerate(shards) for p in paths]
    table = pa.table(
        {
            "shard": [s for s, _ in rows],
            "warc_path": [p for _, p in rows],
            "warc_hash": [warc_hash(to_s3_url(p)) for _, p in rows],
        },
        schema=LAYOUT_SCHEMA,
    )
    with storage.open_file(args.out, "wb") as f:
        pq.write_table(table, f, compression="zstd")
    logger.info("wrote %d WARCs in %d shards to %s", table.num_rows, len(shards), args.out)


def _export_asrun(args: argparse.Namespace) -> None:
    out = storage.normalize_root(args.out_dir)
    summary: dict = {}
    if "worklist" in args.what:
        summary["worklist"] = asrun.export_frozen(f"{args.root}/shards", f"{out}/{V1_0_FILE_NAME}", args.concurrency)
    if "empty-warcs" in args.what:
        summary["empty_warcs"] = asrun.export_empty_warcs(
            f"{args.root}/{asrun.EMPTY_WARCS_AUDIT}", f"{out}/{EMPTY_WARCS_FILE}"
        )
    if "catalog" in args.what:
        shards = v1_0_layout(load_frozen(args.worklist))
        roots = (args.two_phase_root, args.root)
        summary["catalog"] = asrun.export_catalog(
            shards,
            lambda s: roots[0] if s < asrun.TWO_PHASE_SHARDS else roots[1],
            f"{out}/{CATALOG_FILE}",
            args.concurrency,
        )
    print(json.dumps(summary, indent=1))


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("worklist", help="The input WARC list: rebuild, verify, shard layout.")
    sub = parser.add_subparsers(dest="worklist_command", required=True)

    p = sub.add_parser("rebuild", help="Rebuild the frozen v1.0 list from Common Crawl's public listings and check it.")
    p.add_argument("--out", required=True, help=f"Paths file to write, e.g. {V1_0_FILE_NAME}.")
    p.set_defaults(func=_rebuild)

    p = sub.add_parser(
        "build", help="List every WARC of an explicit set of crawls from Common Crawl's public listings."
    )
    p.add_argument(
        "--crawls", nargs="+", required=True, help="CC-MAIN-YYYY-WW, CC-NEWS-YYYY-MM or CC-SUPPLEMENTAL-2026-22."
    )
    p.add_argument(
        "--news-last-warc", help="Last WARC to take from its CC-NEWS month (for a month still being published)."
    )
    p.add_argument("--out", required=True)
    p.set_defaults(func=_build)

    p = sub.add_parser(
        "verify", help="Check a v1.0 list file's digest and compare it crawl by crawl with the public listings."
    )
    p.add_argument("--worklist", required=True, help=f"Local path or URL of {V1_0_FILE_NAME}.")
    p.set_defaults(func=_verify)

    p = sub.add_parser(
        "layout", help="Write production's shard layout of the v1.0 list as Parquet (shard, warc_path, warc_hash)."
    )
    p.add_argument("--worklist", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=_layout)

    p = sub.add_parser(
        "export-asrun", help="(Maintainers) export the frozen list and verification aids from production's records."
    )
    p.add_argument("--root", default=asrun.ASRUN_ROOT, help="Production's fused spec namespace.")
    p.add_argument("--two-phase-root", default=asrun.ASRUN_TWO_PHASE_ROOT, help="Namespace of shards 0-403's catalog.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--what", nargs="+", choices=["worklist", "empty-warcs", "catalog"], required=True)
    p.add_argument("--worklist", help=f"The frozen {V1_0_FILE_NAME} (defines the shards for --what catalog).")
    p.add_argument("--concurrency", type=int, default=32, help="Concurrent object reads.")
    p.set_defaults(func=_export_asrun)
