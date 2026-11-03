# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""R1: run phase A over WARCs and compare with the as-run ``kept/`` + ``tombstones/`` (maintainers).

For every WARC: download it from data.commoncrawl.org, run phase A with the as-run extractor
binaries and the as-run fastText model, tokenize the survivors, and compare with the as-run outputs
(``r1_compare.compare``): survivor set == kept ∪ tombstones, kept text byte-identical,
``fasttext_score`` and ``n_tokens`` exact, phase-A counters exact. Exactness is expected on Linux
x86_64 with CPython 3.12 and the pinned wheels.

A WARC's as-run outputs are split between regions (see ``r1_asrun``), so R1 runs in stages, each
where its data lives; ``refs`` and ``run`` check the GCE region they run in:

1. ``refs --region R``: for the listed WARCs that region R processed, read their records and
   tombstones in R and write one small reference per WARC under ``--refs``;
2. ``run``, in us-central2 next to ``kept/``: phase A and the comparison for every listed WARC; writes
   one JSON report, updated after every WARC (a rerun with the same ``--out`` skips WARCs already
   reported). Split a long list over several ``run`` jobs and merge their reports with ``summarize``.

``plan`` lists the WARCs and the regions that processed them (central records only), i.e. which
``refs`` jobs a list needs::

    python -m tests.parity.r1_phase_a plan --fixtures --sample-shards 0,1500 --per-shard 3
    python -m tests.parity.r1_phase_a refs --region us-east5 --fixtures --sample-shards 0,1500 \\
        --per-shard 3 --refs gs://.../r1/refs
    python -m tests.parity.r1_phase_a run --fixtures --sample-shards 0,1500 --per-shard 3 \\
        --refs gs://.../r1/refs --procs 15 --out gs://.../r1/run.json
    python -m tests.parity.r1_phase_a summarize gs://.../r1/run.json ... --out gs://.../r1/r1.json
"""

import argparse
import collections
import dataclasses
import importlib.metadata
import importlib.util
import json
import logging
import os
import platform
import random
import sys
import time
from datetime import UTC, datetime

import fsspec
import requests

from atlantic_cc.classify.fasttext_gate import FastTextGate
from atlantic_cc.classify.text import normalize_text
from atlantic_cc.classify.tokenize import load_tokenizer, tokenize_batch
from atlantic_cc.extract.install import install_artifact
from atlantic_cc.extract.pool import open_extractor_pool
from atlantic_cc.pipeline.phase_a import run_phase_a
from atlantic_cc.warc.download import download_warc
from atlantic_cc.warc.identity import snapshot_of, warc_hash
from tests.parity.r1_asrun import (
    KEPT_REGION,
    central_record,
    gcs_md5,
    load_ref,
    read_kept,
    read_ref,
    reference,
    shard_warc_paths,
    write_ref,
)
from tests.parity.r1_compare import GOLDEN_ROOT, META_FILE, compare

logger = logging.getLogger(__name__)

ASRUN_FASTTEXT = (
    "gs://marin-us-central2/classifiers/useful_fasttext_lpv11/"
    "resiliparse_scale_w640_sub0p22_strat_prep_mc500_TEXT/model.bin"
)
ASRUN_EXTRACTOR = "gs://marin-us-central2/artifacts/resiliparse_rs/latest"
GCE_ZONE_URL = "http://metadata.google.internal/computeMetadata/v1/instance/zone"
PACKAGES = (
    "warcio",
    "charset-normalizer",
    "requests",
    "urllib3",
    "fasttext-wheel",
    "gigatoken",
    "awkward",
    "tokenizers",
    "numpy",
    "pyarrow",
)
MAX_TEXT_DIFFS = 3
DIFF_CONTEXT = 80


def _gce_region() -> str:
    """The GCP region this process runs in (the metadata server reports ``projects/N/zones/<zone>``)."""
    response = requests.get(GCE_ZONE_URL, headers={"Metadata-Flavor": "Google"}, timeout=10)
    response.raise_for_status()
    return response.text.rsplit("/", 1)[1].rsplit("-", 1)[0]


def _require_region(region: str) -> None:
    here = _gce_region()
    if here != region:
        raise SystemExit(f"this stage reads {region} data and must run in {region}, not {here}")


def _fixture_warcs() -> list[tuple[str, int]]:
    metas = [json.loads((d / META_FILE).read_text()) for d in sorted(GOLDEN_ROOT.iterdir()) if (d / META_FILE).exists()]
    return [(m["warc_path"], m["shard"]) for m in sorted(metas, key=lambda m: m["fixture"])]


def _sample_warcs(shards: list[int], per_shard: int, seed: int) -> list[tuple[str, int]]:
    out = []
    for s in shards:
        paths = shard_warc_paths(s)
        out += [(p, s) for p in sorted(random.Random(seed * 1_000_003 + s).sample(paths, min(per_shard, len(paths))))]
    return out


def _work_list(args: argparse.Namespace) -> list[tuple[str, int]]:
    """``(warc_path, shard)`` for the fixtures, the seeded shard samples and explicit WARCs, deduplicated."""
    warcs = _fixture_warcs() if args.fixtures else []
    if args.sample_shards:
        warcs += _sample_warcs([int(s) for s in args.sample_shards.split(",")], args.per_shard, args.seed)
    warcs += [(w.rsplit("@", 1)[0], int(w.rsplit("@", 1)[1])) for w in args.warc]
    unique: dict[str, tuple[str, int]] = {}
    for w, s in warcs:
        unique.setdefault(warc_hash(w), (w, s))
    return list(unique.values())


def _environment(extractor: dict, fasttext_model: str) -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "gce_region": _gce_region(),
        "packages": {p: importlib.metadata.version(p) for p in PACKAGES},
        "brotli_importable": importlib.util.find_spec("brotli") is not None,
        "extractor": extractor,
        "fasttext_model": {
            "path": fasttext_model,
            "md5": gcs_md5(fasttext_model) if fasttext_model.startswith("gs://") else None,
        },
    }


def _text_diffs(presurvivors, ref, ids: list[str]) -> list[dict]:
    texts = dict(zip(presurvivors.column("doc_id").to_pylist(), presurvivors.column("text").to_pylist(), strict=True))
    out = []
    for doc_id in ids[:MAX_TEXT_DIFFS]:
        ours, theirs = texts[doc_id], ref.texts[doc_id]
        at = next(
            (i for i, (a, b) in enumerate(zip(ours, theirs, strict=False)) if a != b), min(len(ours), len(theirs))
        )
        lo = max(0, at - DIFF_CONTEXT // 2)
        out.append(
            {
                "doc_id": doc_id,
                "len": {"ours": len(ours), "asrun": len(theirs)},
                "first_diff_at": at,
                "ours": ours[lo : at + DIFF_CONTEXT // 2],
                "asrun": theirs[lo : at + DIFF_CONTEXT // 2],
            }
        )
    return out


def _read_report(path: str) -> dict | None:
    fs, root = fsspec.core.url_to_fs(path)
    if not fs.exists(root):
        return None
    with fs.open(root, "r") as f:
        return json.load(f)


def _write_report(path: str, report: dict) -> None:
    with fsspec.open(path, "w") as f:
        json.dump(report, f, indent=2)


def _summary(results: list[dict]) -> dict:
    compared = [r for r in results if "comparison" in r]

    def tally(key) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for r in compared:
            e = out.setdefault(key(r), {"n": 0, "n_exact": 0})
            e["n"] += 1
            e["n_exact"] += r["comparison"]["exact"]
        return dict(sorted(out.items()))

    era = {True: "post-guard", False: "pre-guard", None: "guard-unknown"}
    checks = sorted({k for r in compared for k in r["comparison"]["checks"]})
    return {
        "n_warcs": len(results),
        "n_compared": len(compared),
        "n_exact": sum(r["comparison"]["exact"] for r in compared),
        "n_skipped": len(results) - len(compared),
        "by_era": tally(lambda r: f"{r['code_path']} / {era[r['content_length_guard']]}"),
        "by_region": tally(lambda r: f"{r['region']} / {r['device']}"),
        "by_crawl": tally(lambda r: snapshot_of(r["warc_path"]).rsplit("-", 2)[0]),
        "check_pass_counts": {c: sum(r["comparison"]["checks"][c] is True for r in compared) for c in checks},
        "n_survivors": {
            "ours": sum(r["comparison"]["n_survivors"]["ours"] for r in compared),
            "asrun": sum(r["comparison"]["n_survivors"]["asrun"] for r in compared),
        },
        "n_kept_compared": sum(r["comparison"]["n_kept_compared"] for r in compared),
        "a_n_in": sum(r["comparison"]["counters"]["a_n_in"]["asrun"] for r in compared),
        "not_exact": [r["warc_hash"] for r in compared if not r["comparison"]["exact"]],
    }


def plan(args: argparse.Namespace) -> None:
    rows = []
    for warc_path, shard in _work_list(args):
        h = warc_hash(warc_path)
        _, record = central_record(shard, h)
        rows.append({"warc_hash": h, "shard": shard, "region": record["region"], "device": record["device"]})
        print(f"{h} shard {shard:5d} {record['region']:13s} {record['device']:12s} {warc_path}")
    print(json.dumps(collections.Counter(r["region"] for r in rows), indent=1))


def refs(args: argparse.Namespace) -> None:
    _require_region(args.region)
    n = 0
    for warc_path, shard in _work_list(args):
        h = warc_hash(warc_path)
        if central_record(shard, h)[1]["region"] != args.region:
            continue
        ref = load_ref(warc_path, shard)
        write_ref(ref, args.refs)
        n += 1
        logger.info("%s shard %d: %d tombstones (%s)", h, shard, len(ref.tombstones), ref.tombstones_path)
    logger.info("wrote %d references for region %s under %s", n, args.region, args.refs)


def run(args: argparse.Namespace) -> dict:
    _require_region(KEPT_REGION)
    warcs = _work_list(args)
    installed = install_artifact(args.extractor_artifact, os.path.expanduser(args.extractor_dest))
    if args.require_asrun_extractor and not installed.asrun_binaries:
        raise RuntimeError("the installed extractor binaries are not byte-identical to the as-run build")
    report = _read_report(args.out) or {
        "created": datetime.now(UTC).isoformat(),
        "environment": _environment(
            {"artifact": args.extractor_artifact, **dataclasses.asdict(installed)}, args.fasttext_model
        ),
        "warcs": [],
    }
    done = {r["warc_hash"] for r in report["warcs"]}
    todo = [(w, s) for w, s in warcs if warc_hash(w) not in done]
    logger.info("R1: %d WARCs (%d already reported)", len(todo), len(warcs) - len(todo))

    gate = FastTextGate.load(args.fasttext_model)
    tok = load_tokenizer()
    with open_extractor_pool(args.procs, installed.pkg_dir) as pool:
        for i, (warc_path, shard) in enumerate(todo):
            h = warc_hash(warc_path)
            ref = read_ref(args.refs, h)
            entry = {
                "warc_path": warc_path,
                "warc_hash": h,
                "shard": shard,
                "code_path": ref.code_path,
                "content_length_guard": ref.content_length_guard,
                "processed_at": ref.processed_at,
                "device": ref.counters["device"],
                "region": ref.region,
            }
            if "a_download_failed" in ref.counters:
                entry["skipped"] = f"as-run download failed: {ref.counters['a_download_failed']}"
                report["warcs"].append(entry)
                continue
            started = time.monotonic()
            table, counters = run_phase_a(warc_path, gate, pool, fetch=download_warc)
            n_tokens = [
                len(ids) for ids in tokenize_batch(tok, [normalize_text(t) for t in table.column("text").to_pylist()])
            ]
            expected = reference(ref, read_kept(h))
            comparison = compare(table, counters, expected, n_tokens)
            comparison["text_diffs"] = _text_diffs(table, expected, comparison["kept_text_mismatch"]["ids"])
            entry["comparison"] = comparison
            entry["seconds"] = round(time.monotonic() - started, 1)
            report["warcs"].append(entry)
            report["summary"] = _summary(report["warcs"])
            _write_report(args.out, report)
            logger.info(
                "[%d/%d] %s exact=%s survivors %d/%d counters %s (%.0fs)",
                i + 1,
                len(todo),
                h,
                comparison["exact"],
                comparison["n_survivors"]["ours"],
                comparison["n_survivors"]["asrun"],
                comparison["checks"]["counters"],
                entry["seconds"],
            )
    report["summary"] = _summary(report["warcs"])
    report["finished"] = datetime.now(UTC).isoformat()
    _write_report(args.out, report)
    print(json.dumps(report["summary"], indent=2))
    return report


def summarize(args: argparse.Namespace) -> None:
    reports = [_read_report(p) for p in args.reports]
    missing = [p for p, r in zip(args.reports, reports, strict=True) if r is None]
    if missing:
        raise FileNotFoundError(f"no report at {missing}")
    warcs = [w for r in reports for w in r["warcs"]]
    hashes = [w["warc_hash"] for w in warcs]
    assert len(set(hashes)) == len(hashes), "a WARC appears in more than one report"
    merged = {
        "runs": [
            {"report": p, **{k: r.get(k) for k in ("created", "finished", "environment")}}
            for p, r in zip(args.reports, reports, strict=True)
        ],
        "summary": _summary(warcs),
        "warcs": warcs,
    }
    _write_report(args.out, merged)
    print(json.dumps(merged["summary"], indent=2))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    work = argparse.ArgumentParser(add_help=False)
    work.add_argument("--fixtures", action="store_true", help="The WARCs of the committed G1 goldens (F1-F6).")
    work.add_argument("--sample-shards", default="", help="Comma-separated shard ids to sample WARCs from.")
    work.add_argument("--per-shard", type=int, default=3)
    work.add_argument("--seed", type=int, default=0)
    work.add_argument(
        "--warc", action="append", default=[], help="Explicit 's3://commoncrawl/...@<shard>'; repeatable."
    )

    sub.add_parser("plan", parents=[work], help="List the WARCs and their regions.").set_defaults(fn=plan)

    p = sub.add_parser("refs", parents=[work], help="Write the references of the WARCs one region processed.")
    p.add_argument("--region", required=True, help="GCP region, e.g. europe-west4; the job must run there.")
    p.add_argument("--refs", required=True, help="Directory (fsspec URL) for the references.")
    p.set_defaults(fn=refs)

    p = sub.add_parser("run", parents=[work], help="Phase A and the comparison (in us-central2).")
    p.add_argument("--refs", required=True)
    p.add_argument("--fasttext-model", default=ASRUN_FASTTEXT)
    p.add_argument("--extractor-artifact", default=ASRUN_EXTRACTOR)
    p.add_argument("--extractor-dest", default="~/atlantic_cc_extractor", help="Must not be a noexec mount.")
    p.add_argument("--require-asrun-extractor", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--procs", type=int, required=True, help="Extraction processes (the job's CPUs minus one).")
    p.add_argument("--out", required=True, help="Report path (local or fsspec URL).")
    p.set_defaults(fn=run)

    p = sub.add_parser("summarize", help="Merge run reports into one.")
    p.add_argument("reports", nargs="+")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=summarize)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
