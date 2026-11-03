# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``atlantic-cc release ...``: build the release from the dedup + decontamination outputs, and use it."""

import argparse
import json
import logging

from atlantic_cc.release import checksums, download, regroup, storage, subset, verify
from atlantic_cc.release.params import (
    HTTPS_ROOT,
    PART_MB,
    VERSION,
    ZSTD_LEVEL,
    Kind,
    checksums_name,
    manifest_name,
)
from atlantic_cc.warc.identity import snapshot_of

logger = logging.getLogger(__name__)

KIND_CHOICES = [k.value for k in Kind]


def _shard(value: str) -> tuple[int, int]:
    """``I/N`` -> (I, N)."""
    i, n = (int(x) for x in value.split("/"))
    if not 0 <= i < n:
        raise argparse.ArgumentTypeError(f"bad shard {value!r}; expected I/N with 0 <= I < N")
    return i, n


def _warc_paths(args: argparse.Namespace) -> None:
    regroup.write_warc_paths(args.worklist, storage.normalize_root(args.stage))


def _pieces(args: argparse.Namespace) -> None:
    shard, num_shards = args.shard
    regroup.pieces(
        storage.normalize_root(args.source),
        storage.normalize_root(args.decon_run),
        storage.normalize_root(args.stage),
        args.drop,
        args.workers,
        limit_files=args.limit_files,
        batch_ids=args.batch_ids,
        shard=shard,
        num_shards=num_shards,
    )


def _parts(args: argparse.Namespace) -> None:
    shard, num_shards = args.shard
    regroup.parts(
        storage.normalize_root(args.out),
        args.version,
        storage.normalize_root(args.stage),
        args.part_mb,
        args.zstd_level,
        args.workers,
        plan_path=args.plan,
        shard=shard,
        num_shards=num_shards,
    )


def _manifest(args: argparse.Namespace) -> None:
    regroup.manifest(
        storage.normalize_root(args.out),
        args.version,
        storage.normalize_root(args.stage),
        storage.normalize_root(args.decon_run) if args.decon_run else None,
        plan_path=args.plan,
        gzip_mtime=args.gzip_mtime,
    )


def _checksums(args: argparse.Namespace) -> None:
    records = checksums.read_done_markers(storage.normalize_root(args.done_markers), args.concurrency)
    md5s = checksums.checksums_from_uploads(records)
    storage.write_bytes(args.out, checksums.format_checksums(md5s).encode())
    logger.info("wrote %d checksums (%.3f TB) to %s", len(md5s), sum(r.bytes for r in records) / 1e12, args.out)


def _load_checksums(path: str | None) -> dict[str, str] | None:
    return checksums.parse_checksums(storage.read_bytes(path).decode()) if path else None


def _verify(args: argparse.Namespace) -> None:
    root = storage.normalize_root(args.root)
    manifest = storage.read_json(args.manifest or f"{root}/{manifest_name(args.version)}")
    report = verify.verify(
        root,
        manifest,
        _load_checksums(args.checksums),
        args.crawls,
        [Kind(k) for k in args.kinds],
        [verify.Check(c) for c in args.checks],
        args.workers,
    )
    print(json.dumps(report.to_dict(), indent=1))
    if not report.ok:
        raise SystemExit(1)


def _download(args: argparse.Namespace) -> None:
    if not args.crawls and not args.all:
        raise SystemExit("choose crawls with --crawls, or --all for the whole release (about 22.8 TB)")
    summary = download.download(
        args.base_url.rstrip("/"),
        storage.normalize_root(args.dest),
        None if args.all else args.crawls,
        [Kind(k) for k in args.kinds],
        args.workers,
        _load_checksums(args.checksums),
        args.version,
    )
    print(json.dumps(summary, indent=1))


def _subset(args: argparse.Namespace) -> None:
    release = storage.normalize_root(args.release)
    output = storage.normalize_root(args.output)
    targets = subset.load_warc_list(args.warcs)
    manifest = storage.read_json(f"{release}/{manifest_name(args.version)}")
    files = subset.subset_files(release, manifest, {snapshot_of(w) for w in targets})
    chunks = subset.chunk_files(files)[: args.limit_chunks]
    logger.info("%d target WARCs; %d release files in %d chunks", len(targets), len(files), len(chunks))
    if args.merge:
        summary = subset.merge(output, release, len(chunks), targets, args.tokenizer)
        print(
            json.dumps(
                {
                    k: summary[k]
                    for k in (
                        "warcs_in_list",
                        "warcs_with_docs",
                        "files_scanned",
                        "row_groups_read",
                        "total",
                        "by_kind",
                    )
                },
                indent=1,
            )
        )
        return
    tokenizer_dir = subset.stage_tokenizer(args.tokenizer) if args.tokenizer else None
    part, parts = args.part
    export = storage.normalize_root(args.export) if args.export else None
    subset.run_part(output, export, chunks, part, parts, args.workers, targets, tokenizer_dir)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("release", help="Build, check, download or subset the crawl-partitioned release.")
    sub = parser.add_subparsers(dest="release_command", required=True)

    p = sub.add_parser("warc-paths", help="Step 1: map <crawl>/<warc_hash> to the WARC path for every worklist WARC.")
    p.add_argument("--worklist", required=True, help="Paths file of the input WARCs (v1.0.input.warc.paths.gz).")
    p.add_argument("--stage", required=True, help="Scratch directory for intermediate files.")
    p.set_defaults(func=_warc_paths)

    p = sub.add_parser("pieces", help="Step 2: split the dedup output into per-crawl, per-kind pieces.")
    p.add_argument("--source", required=True, help="Dedup output directory (deduped_c075/*.jsonl.gz).")
    p.add_argument("--decon-run", required=True, help="Decontamination filter run (removed_docs.parquet, export.json).")
    p.add_argument("--stage", required=True)
    p.add_argument("--workers", type=int, required=True, help="Processes; each holds the 8.6M-entry path map (~3 GB).")
    p.add_argument("--drop", nargs="*", default=[], choices=regroup.RELEASE_SCHEMA.names, help="Fields to leave out.")
    p.add_argument("--limit-files", type=int, help="Use only the first N source files (test runs).")
    p.add_argument("--batch-ids", type=int, nargs="+", help="Build only these batches.")
    p.add_argument("--shard", type=_shard, default=(0, 1), help="I/N: build every N-th batch, starting at I.")
    p.set_defaults(func=_pieces)

    p = sub.add_parser("parts", help="Step 3: pack each crawl's pieces into ~1 GB part files.")
    p.add_argument("--stage", required=True)
    p.add_argument("--out", required=True, help="Release root; parts go to <out>/<version>/crawl=<crawl>/.")
    p.add_argument("--version", default=VERSION)
    p.add_argument("--part-mb", type=int, default=PART_MB, help="Close a part once its pieces reach this many MB.")
    p.add_argument("--zstd-level", type=int, default=ZSTD_LEVEL)
    p.add_argument("--workers", type=int, required=True, help="Processes; a 1 GB part needs ~4 GB of memory.")
    p.add_argument("--plan", help="Where to write the part plan (default <stage>/parts_plan.json).")
    p.add_argument("--shard", type=_shard, default=(0, 1), help="I/N: write every N-th part, starting at I.")
    p.set_defaults(func=_parts)

    p = sub.add_parser(
        "manifest", help="Step 4: write the manifest and paths files; check totals against the decon run."
    )
    p.add_argument("--stage", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--version", default=VERSION)
    p.add_argument(
        "--decon-run", required=True, help="Decontamination run whose export.json totals must match; '' skips."
    )
    p.add_argument("--plan", help="Part plan (default <stage>/parts_plan.json).")
    p.add_argument("--gzip-mtime", type=int, help="Timestamp for the paths files' gzip header (v1.0: 1793254392).")
    p.set_defaults(func=_manifest)

    p = sub.add_parser(
        "checksums", help=f"Write {checksums_name(VERSION)} from the upload done-markers (S3 ETag = MD5)."
    )
    p.add_argument("--done-markers", required=True, help="Directory of <key>.json upload markers.")
    p.add_argument("--out", required=True)
    p.add_argument("--concurrency", type=int, default=32)
    p.set_defaults(func=_checksums)

    p = sub.add_parser("verify", help="Check a copy of the release against its manifest and checksums.")
    p.add_argument("--root", required=True, help=f"Release root, e.g. {HTTPS_ROOT} or a local download directory.")
    p.add_argument("--manifest", help="Manifest to check against (default <root>/<version>.manifest.json).")
    p.add_argument("--checksums", help=f"{checksums_name(VERSION)} (needed for the md5 check).")
    p.add_argument("--crawls", nargs="+", help="Only these crawls (default all).")
    p.add_argument("--kinds", nargs="+", choices=KIND_CHOICES, default=KIND_CHOICES)
    p.add_argument(
        "--checks", nargs="+", choices=[c.value for c in verify.Check], default=[c.value for c in verify.Check]
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--version", default=VERSION)
    p.set_defaults(func=_verify)

    p = sub.add_parser("download", help="Download crawls of the release over HTTPS, resumably.")
    p.add_argument("--dest", required=True)
    p.add_argument("--crawls", nargs="+", help="Crawls to fetch, e.g. CC-MAIN-2024-10 CC-NEWS-2025-01.")
    p.add_argument("--all", action="store_true", help="Fetch every crawl (about 22.8 TB).")
    p.add_argument("--kinds", nargs="+", choices=KIND_CHOICES, default=KIND_CHOICES)
    p.add_argument(
        "--checksums", help=f"Local path or URL of {checksums_name(VERSION)}; default: check each file's ETag."
    )
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--base-url", default=HTTPS_ROOT)
    p.add_argument("--version", default=VERSION)
    p.set_defaults(func=_download)

    p = sub.add_parser("subset", help="Restrict the release to a list of WARCs: counts per WARC, optional export.")
    p.add_argument(
        "--warcs", required=True, help="WARC list: bucket-relative paths or s3://commoncrawl/ URLs, one per line."
    )
    p.add_argument("--output", required=True, help="Directory for chunk stats and summary.json.")
    p.add_argument("--release", default=HTTPS_ROOT, help="Release root (HTTPS, s3://, gs:// or local).")
    p.add_argument("--part", type=_shard, default=(0, 1), help="I/N: scan every N-th chunk, starting at I.")
    p.add_argument("--merge", action="store_true", help="Sum the chunk stats into summary.json.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit-chunks", type=int, help="Smoke test: only the first K chunks.")
    p.add_argument("--tokenizer", help="Count tokens with this Hugging Face tokenizer (no special tokens).")
    p.add_argument("--export", help="Also write the matched clean documents as Parquet under this prefix.")
    p.add_argument("--version", default=VERSION)
    p.set_defaults(func=_subset)
