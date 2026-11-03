# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``atlantic-cc decon ...``: replay decontamination on released or deduplicated text, and rebuild its inputs.

Replay (exact on the released text, see ``docs/decon.md``)::

    atlantic-cc decon filter --kind release --eval-hits eval_hits.npy --names eval_hit_names.parquet \\
        --out decon_out/ 'crawl=CC-NEWS-2016-08/*.parquet'

Rebuild steps (as-run ports; need the deduplicated pool, gated eval data and Marin-scale compute)::

    atlantic-cc decon eval-corpus aa|lmh|download-val|stage-val ...   # best effort
    atlantic-cc decon eval-index --full FULL --val VAL --out INDEX/   # index + full-val bloom
    atlantic-cc decon sample|dropsets|mark|candidate-counts|pool-counts|code-counts|drop-hashes|filter-pool|manifest
"""

import argparse
import json
import logging
import os

import fsspec
import pyarrow.parquet as pq

from atlantic_cc.decon import dropsets as ds
from atlantic_cc.decon import eval_corpus, eval_index, pool_steps, val_sets
from atlantic_cc.decon.artifacts import (
    build_as_run_artifacts,
    load_drop_hashes,
    load_eval_hit_names,
    load_eval_hits,
    load_uint64,
    save_uint64,
    sha256_uint64,
)
from atlantic_cc.decon.bloom import load_native_bloom
from atlantic_cc.decon.filter import InputKind, decon_files

logger = logging.getLogger(__name__)


def _expand(paths: list[str]) -> list[str]:
    out: list[str] = []
    for path in paths:
        if any(c in path for c in "*?["):
            fs, pattern = fsspec.core.url_to_fs(path)
            protocol = path.split("://", 1)[0] + "://" if "://" in path else ""
            out.extend(sorted(protocol + p for p in fs.glob(pattern)))
        else:
            out.append(path)
    return out


def _print(obj) -> None:
    print(json.dumps(obj, indent=1, sort_keys=True, default=str))


def _drop(path: str | None) -> frozenset[int]:
    return frozenset(load_uint64(path).tolist()) if path else load_drop_hashes()


def _filter(args: argparse.Namespace) -> None:
    if (args.eval_hits is None) == (args.bloom is None):
        raise SystemExit("pass exactly one of --eval-hits and --bloom")
    membership = load_eval_hits(args.eval_hits) if args.eval_hits else load_native_bloom(args.bloom)
    names = load_eval_hit_names(args.names) if args.names else {}
    summaries = decon_files(
        _expand(args.inputs), InputKind(args.kind), args.out, membership, _drop(args.drop_hashes), names, args.workers
    )
    totals = {
        "files": len(summaries),
        "docs": sum(s["docs"] for s in summaries),
        "flagged": sum(s["flagged"] for s in summaries),
        "matched_hashes_without_names": sum(s["matched_hashes_without_names"] for s in summaries),
    }
    if args.kind == InputKind.RELEASE:
        totals["stored_flagged"] = sum(s["stored_flagged"] for s in summaries)
        totals["mismatched"] = sum(s["mismatched"] for s in summaries)
    _print(totals)


def _dropsets(args: argparse.Namespace) -> None:
    _print({"global": ds.run_dropsets(args.sample_dir, load_native_bloom(args.bloom), args.out, args.workers)})


def _drop_hashes(args: argparse.Namespace) -> None:
    lists = [ds.read_parquet(p, columns=["hash"])["hash"].to_pylist() for p in ds.drop_tables_dirs(args.dropsets)]
    pool = ds.pool_common_hashes(ds.read_parquet(p) for p in ds.partition_paths(args.boilerplate, "pool_totals"))
    code = ds.code_common_hashes(ds.read_parquet(p) for p in ds.partition_paths(args.boilerplate, "code_totals"))
    drop = ds.drop_set_union(lists, pool, code)
    save_uint64(args.out, drop)
    _print({"count": int(drop.size), "sha256": sha256_uint64(drop)})


def _sample(args: argparse.Namespace) -> None:
    _print({"files": pool_steps.sample(args.input, args.sample_dir, args.files, args.per_file, args.workers)})


def _mark(args: argparse.Namespace) -> None:
    results = pool_steps.mark(args.input, args.attributes, load_native_bloom(args.bloom), args.workers)
    _print(
        {
            "files": len(results),
            "docs": sum(r["docs"] for r in results),
            "candidates": sum(r["candidates"] for r in results),
        }
    )


def _candidate_counts(args: argparse.Namespace) -> None:
    _print(pool_steps.candidate_counts(args.attributes, args.boilerplate, args.workers))


def _pool_counts(args: argparse.Namespace) -> None:
    _print(pool_steps.pool_counts(args.input, args.boilerplate, args.workers))


def _code_counts(args: argparse.Namespace) -> None:
    _print(pool_steps.code_counts(args.code_corpus, args.boilerplate, args.workers))


def _filter_pool(args: argparse.Namespace) -> None:
    results = pool_steps.filter_pool(
        args.input,
        args.attributes,
        args.output,
        args.removed,
        load_native_bloom(args.bloom),
        _drop(args.drop_hashes),
        args.workers,
    )
    _print({k: sum(r[k] for r in results) for k in ("docs", "candidates", "kept", "removed")} | {"files": len(results)})


def _manifest(args: argparse.Namespace) -> None:
    _print({"removed_docs": pool_steps.manifest(args.removed, args.index, args.out)})


def _eval_corpus(args: argparse.Namespace) -> None:
    if args.eval_corpus_command == "aa":
        _print(eval_corpus.prepare_aa(args.root))
    elif args.eval_corpus_command == "lmh":
        manifest = eval_corpus.prepare_lmh(args.root, args.tasks or None)
        _print({k: manifest[k] for k in ("status", "failed")} | {"artifacts": len(manifest["artifacts"])})
    elif args.eval_corpus_command == "download-val":
        token = os.environ.get("HF_TOKEN")
        paloma_revision = val_sets.download_paloma(f"{args.raw}/paloma", token)
        lima_files = val_sets.download_lima(f"{args.raw}/lima_raw", token)
        lima_rows = val_sets.flatten_lima(lima_files, f"{args.raw}/lima/train.jsonl.gz")
        val_sets.download_uncheatable(f"{args.raw}/uncheatable", os.environ.get("GITHUB_TOKEN"))
        _print({"paloma_revision": paloma_revision, "lima_records": lima_rows})
    else:
        _print(
            val_sets.stage_val(
                args.root, f"{args.raw}/paloma", f"{args.raw}/uncheatable", f"{args.raw}/lima/train.jsonl.gz"
            )
        )


def _eval_index(args: argparse.Namespace) -> None:
    _print(eval_index.build_eval_index(args.full, args.val, args.out.rstrip("/")))


def _eval_digests(args: argparse.Namespace) -> None:
    summaries, table = eval_index.corpus_digests(args.full, args.val)
    out = args.out.rstrip("/")
    with fsspec.open(f"{out}/eval_file_digests.json", "w") as fh:
        json.dump(summaries, fh, indent=1)
    with fsspec.open(f"{out}/eval_record_digests.parquet", "wb") as fh:
        pq.write_table(table, fh, compression="zstd")
    report = {"files": len(summaries), "records": table.num_rows}
    if args.compare:
        with fsspec.open(args.compare, "r") as fh:
            report |= eval_index.compare_file_digests(summaries, json.load(fh))
    if args.compare_records:
        with fsspec.open(args.compare_records, "rb") as fh:
            report["records"] = eval_index.compare_record_digests(table, pq.read_table(fh))
    _print(report)


def _export_artifacts(args: argparse.Namespace) -> None:
    _print(build_as_run_artifacts(args.out, check_bloom=not args.skip_bloom_checks))


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add the ``decon`` command and its subcommands."""
    decon = subparsers.add_parser("decon", help="Decontamination (rule v4, eval set full-val).")
    sub = decon.add_subparsers(dest="decon_command", required=True)

    p = sub.add_parser("filter", help="Flag documents and name their matched evals (single pass; exact replay).")
    p.add_argument("inputs", nargs="+", help="Input files or globs (local or fsspec URLs).")
    p.add_argument("--kind", choices=list(InputKind), default=InputKind.RELEASE)
    p.add_argument("--out", required=True, help="Output directory for <file>.decon.parquet/.json.")
    p.add_argument("--eval-hits", help="S' (eval_hits.npy): exact membership for the released text.")
    p.add_argument("--bloom", help="The full-val bloom (filter.bin): membership for any text.")
    p.add_argument("--names", help="hash -> eval names (eval_hit_names.parquet).")
    p.add_argument("--drop-hashes", help="Drop hashes .npy (default: the packaged D).")
    p.add_argument("--workers", type=int, default=1)
    p.set_defaults(func=_filter)

    p = sub.add_parser("sample", help="Drop-set sample from the deduplicated pool (seeded; as-run layout only).")
    p.add_argument("--input", required=True, help="Directory of deduplicated *.jsonl.gz shards.")
    p.add_argument("--sample-dir", required=True)
    p.add_argument("--files", type=int, default=pool_steps.SAMPLE_FILES)
    p.add_argument("--per-file", type=int, default=pool_steps.SAMPLE_PER_FILE)
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=_sample)

    p = sub.add_parser("dropsets", help="Per-source and global boilerplate lists from a drop-set sample.")
    p.add_argument("--sample-dir", required=True, help="<dir>/<source>/part-NNNNNN.parquet with id, text.")
    p.add_argument("--bloom", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=_dropsets)

    p = sub.add_parser("mark", help="Mark every document (one phrase, no drops): per-doc attributes.")
    p.add_argument("--input", required=True)
    p.add_argument("--attributes", required=True)
    p.add_argument("--bloom", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=_mark)

    p = sub.add_parser("candidate-counts", help="Candidate document frequency of every matched phrase (S').")
    p.add_argument("--attributes", required=True)
    p.add_argument("--boilerplate", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=_candidate_counts)

    p = sub.add_parser("pool-counts", help="Exact pool doc and crawl counts of tracked phrases (pooldf1000).")
    p.add_argument("--input", required=True)
    p.add_argument("--boilerplate", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=_pool_counts)

    p = sub.add_parser("code-counts", help="StarCoderData file counts of tracked phrases (codedf100).")
    p.add_argument("--code-corpus", required=True, help="bigcode/starcoderdata@9fc30b5 layout: <language>/*.parquet")
    p.add_argument("--boilerplate", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=_code_counts)

    p = sub.add_parser("drop-hashes", help="Union D of the per-crawl, global, pool and code lists.")
    p.add_argument("--dropsets", required=True, help="Directory of <source>/drop.parquet and _global/drop.parquet.")
    p.add_argument("--boilerplate", required=True, help="Directory with pool_totals/ and code_totals/ partitions.")
    p.add_argument("--out", required=True, help="Output .npy (sorted little-endian uint64).")
    p.set_defaults(func=_drop_hashes)

    p = sub.add_parser("filter-pool", help="As-run two-pass filter over the marked pool (writes a copy).")
    p.add_argument("--input", required=True)
    p.add_argument("--attributes", required=True)
    p.add_argument("--output", required=True, help="Decontaminated copy (kept records).")
    p.add_argument("--removed", required=True, help="Removed-doc sidecars directory.")
    p.add_argument("--bloom", required=True)
    p.add_argument("--drop-hashes", help="Drop hashes .npy (default: the packaged D).")
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=_filter_pool)

    p = sub.add_parser("manifest", help="removed_docs.parquet with matched_evals from the hash indexes.")
    p.add_argument("--removed", required=True)
    p.add_argument("--index", nargs="+", required=True, help="eval_hash_index.parquet and val_index.parquet")
    p.add_argument("--out", required=True)
    p.set_defaults(func=_manifest)

    p = sub.add_parser("eval-corpus", help="Rebuild the eval corpus (best effort; gated sources need HF_TOKEN).")
    ec = p.add_subparsers(dest="eval_corpus_command", required=True)
    q = ec.add_parser("aa", help="The nine AA Intelligence Index v4.1.1 benchmarks.")
    q.add_argument("--root", required=True)
    q = ec.add_parser("lmh", help="lm-eval-harness leaf tasks (needs the pinned lm-eval fork).")
    q.add_argument("--root", required=True)
    q.add_argument("--tasks", nargs="*", default=[])
    q = ec.add_parser("download-val", help="Download Paloma, LIMA (HF, gated) and the pinned Uncheatable dumps.")
    q.add_argument("--raw", required=True, help="Local directory for raw validation data.")
    q = ec.add_parser("stage-val", help="Stage the 24 validation sets as <root>/<name>/eval.parquet.")
    q.add_argument("--raw", required=True)
    q.add_argument("--root", required=True)
    p.set_defaults(func=_eval_corpus)

    p = sub.add_parser("eval-index", help="Hash index and full-val bloom from a staged eval corpus.")
    p.add_argument("--full", required=True, help="Root with aa/ and lmh/.")
    p.add_argument("--val", required=True, help="Root with <name>/eval.parquet.")
    p.add_argument("--out", required=True)
    p.set_defaults(func=_eval_index)

    p = sub.add_parser("eval-digests", help="Per-record and per-file digests of a staged eval corpus.")
    p.add_argument("--full", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--compare", help="Shipped eval_file_digests.json to compare against.")
    p.add_argument("--compare-records", help="Shipped eval_record_digests.parquet to compare against.")
    p.set_defaults(func=_eval_digests)

    p = sub.add_parser(
        "export-artifacts", help="Build D, S' and the S' name map from the as-run Marin tables (maintainers)."
    )
    p.add_argument("--out", required=True, help="Output directory (fsspec URL).")
    p.add_argument("--skip-bloom-checks", action="store_true")
    p.set_defaults(func=_export_artifacts)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="atlantic-cc")
    register(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
