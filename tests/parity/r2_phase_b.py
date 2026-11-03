# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""R2 (TPU, JAX backend) / R3 (GPU, PyTorch backend): extract and classify fixture WARCs, then compare
every fastText survivor's scores and decision with production's outputs.

    # run the worker on F3, then compare (arguments after -- go to `atlantic-cc extract-classify`)
    python -m tests.parity.r2_phase_b run --fixtures F3 --output OUT --report OUT/r2_report.json -- \\
        --backend jax --phase-a-procs 4 --fasttext-model ... --pooled-model ... --terminal-model ... --extractor ...

    # compare an existing output tree
    python -m tests.parity.r2_phase_b compare --fixtures F3 --output OUT --report OUT/r2_report.json

The report (JSON) has, per WARC and aggregated (all WARCs, and split by whether our device type
equals the one production used for that WARC): survivor-set agreement, |delta p| distributions
for the pooled and terminal scores, the route confusion matrix, decision agreement, and every flip
with its distance to the threshold that decided it. Reading production's outputs needs Marin GCS
credentials; run in us-central2, where every as-run kept file lives.
"""

import argparse
import datetime
import gzip
import io
import json
import logging
import os
import tempfile

import fsspec
import pyarrow.parquet as pq

from atlantic_cc.io.resolve import store_for
from atlantic_cc.pipeline import cli as pipeline_cli
from atlantic_cc.pipeline.outputs import is_done, kept_path, read_marker, tombstones_path
from tests.parity.r2_asrun import FIXTURES, AsRunWarc, base_warc, load_asrun
from tests.parity.r2_compare import (
    MIN_DECISION_AGREEMENT,
    POOLED_FLIP_EPSILON,
    TERMINAL_FLIP_EPSILON,
    aggregate,
    compare_survivors,
    survivors_from_outputs,
)

logger = logging.getLogger("r2_phase_b")

# Our marker's counter -> production's catalog column (two-phase catalogs lack the phase-A ones).
PHASE_A_COUNTERS = {
    "n_in": "a_n_in",
    "n_prefilter_dropped": "a_n_prefilter_dropped",
    "n_extract_empty": "a_n_extract_empty",
    "n_extract_crashed": "a_n_extract_crashed",
}
PHASE_B_COUNTERS = {name: name for name in ("n_presurvivors", "n_hi_accept", "n_band", "n_kept")}


def run_extract_classify(root: str, warc_urls: list[str], worker_args: list[str]) -> None:
    """``atlantic-cc extract-classify`` over ``warc_urls`` into ``root``; WARCs that raise are left unmarked."""
    parser = argparse.ArgumentParser(prog="atlantic-cc")
    pipeline_cli.register(parser.add_subparsers(dest="command", required=True))
    with tempfile.NamedTemporaryFile("w", suffix=".warc.paths", delete=False) as f:
        f.write("".join(f"{url}\n" for url in warc_urls))
    args = parser.parse_args(["extract-classify", "--warc-list", f.name, "--output", root, *worker_args])
    try:
        args.func(args)
    except SystemExit as e:
        if e.code:
            logger.warning("extract-classify exited with %s; comparing the WARCs it finished", e.code)
    finally:
        os.unlink(f.name)


def compare_warc(root: str, name: str, warc: AsRunWarc):
    store = store_for(root)
    h = warc.warc_hash
    entry = {"name": name, "warc_path": warc.warc_path, "warc_hash": h, "asrun_shard": warc.shard}
    if not is_done(store, root, h):
        return {**entry, "status": "missing"}, None
    marker = read_marker(store, root, h)
    kept = pq.read_table(io.BytesIO(store.read(kept_path(root, h))))
    tombstones_file = tombstones_path(root, h)
    tombstones = (
        []
        if store.stat(tombstones_file) is None
        else [json.loads(line) for line in gzip.decompress(store.read(tombstones_file)).decode().splitlines()]
    )
    asrun = load_asrun(warc)
    comparison = compare_survivors(
        survivors_from_outputs(kept, tombstones), survivors_from_outputs(asrun.kept, asrun.tombstones)
    )
    counters = PHASE_A_COUNTERS | PHASE_B_COUNTERS
    ours_counters = {name: getattr(marker, name) for name in counters}
    asrun_counters = {name: asrun.catalog_row.get(column) for name, column in counters.items()}
    entry |= {
        "status": str(marker.status),
        "backend": marker.backend,
        "device": marker.device,
        "n_devices": marker.n_devices,
        "asrun_device": asrun.catalog_row["device"],
        "asrun_n_devices": asrun.catalog_row["n_devices"],
        "same_device_type": marker.device == asrun.catalog_row["device"],
        "counters": {"ours": ours_counters, "asrun": asrun_counters},
        "phase_a_counters_equal": all(ours_counters[n] == asrun_counters[n] for n in PHASE_A_COUNTERS),
        "seconds": {
            "ours": {k: getattr(marker, f"{k}_seconds") for k in ("phase_a", "tokenize", "pooled", "terminal")},
            "asrun": {k: asrun.catalog_row.get(f"{k}_s") for k in ("tokenize", "pooled", "score", "wall")},
        },
        **comparison.summary,
    }
    return entry, comparison


def build_report(root: str, warcs: dict[str, AsRunWarc]) -> dict:
    entries, same, other = [], [], []
    for name, warc in warcs.items():
        entry, comparison = compare_warc(root, name, warc)
        entries.append(entry)
        if comparison is None:
            logger.warning("%s (%s): no marker under %s", name, warc.warc_hash, root)
            continue
        (same if entry["same_device_type"] else other).append(comparison)
        logger.info(
            "%s %s: %s vs as-run %s; %d common survivors, agreement %.5f, %d flips, pooled |dp| p99 %.2e max %.2e",
            name,
            warc.warc_hash,
            entry["device"],
            entry["asrun_device"],
            comparison.n_common,
            entry["decision_agreement"],
            entry["n_flips"],
            entry["pooled_abs_delta"].get("p99", float("nan")),
            entry["pooled_abs_delta"].get("max", float("nan")),
        )
    return {
        "harness": "tests/parity/r2_phase_b.py",
        "created": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "output": root,
        "gates": {
            "min_decision_agreement": MIN_DECISION_AGREEMENT,
            "pooled_flip_epsilon": POOLED_FLIP_EPSILON,
            "terminal_flip_epsilon": TERMINAL_FLIP_EPSILON,
        },
        "aggregate": {
            "all": aggregate(same + other),
            "same_device_type": aggregate(same),
            "other_device_type": aggregate(other),
        },
        "warcs": entries,
    }


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "compare"])
    parser.add_argument("--fixtures", nargs="*", default=[], choices=sorted(FIXTURES))
    parser.add_argument("--warcs", nargs="*", default=[], help="s3://commoncrawl/ URLs of base-crawl CC-MAIN WARCs")
    parser.add_argument("--output", required=True, help="output root of the worker (local or gs://)")
    parser.add_argument("--report", required=True, help="where to write the JSON report (local or gs://)")
    args, worker_args = parser.parse_known_args(argv)
    if worker_args[:1] == ["--"]:
        worker_args = worker_args[1:]
    warcs = {name: FIXTURES[name] for name in args.fixtures} | {url: base_warc(url) for url in args.warcs}
    if not warcs:
        parser.error("name at least one WARC with --fixtures or --warcs")
    root = args.output.rstrip("/")
    if args.command == "run":
        run_extract_classify(root, [w.warc_path for w in warcs.values()], worker_args)
    elif worker_args:
        parser.error(f"unexpected arguments for compare: {worker_args}")
    report = build_report(root, warcs)
    with fsspec.open(args.report, "w") as f:
        json.dump(report, f, indent=1)
    logger.info("report -> %s; aggregate: %s", args.report, json.dumps(report["aggregate"]["all"]))


if __name__ == "__main__":
    main()
