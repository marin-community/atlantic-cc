# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``atlantic-cc dedup init|run|worker|status|verify``.

atlantic-cc dedup init --run-root RUN --output-root OUT --kept gs://bucket/.../kept/ --expect-files 8582115
atlantic-cc dedup run --run-root RUN --cpus 64 --memory-gb 240          # one host, resumable
atlantic-cc dedup worker --run-root RUN --cpus 16 --memory-gb 60        # any number of hosts (claims)
atlantic-cc dedup status --run-root RUN
atlantic-cc dedup verify --candidate RUN --reference OTHER_RUN --report report.json
"""

import argparse
import json
import logging
import os
import sys

from atlantic_cc.dedup.params import FILES_PER_OUTPUT, UNIT_BYTES
from atlantic_cc.dedup.runtime import SimdPolicy, apply_simd_policy, record_runtime
from atlantic_cc.dedup.tasks import ClaimConfig
from atlantic_cc.spec import PRODUCTION_SPEC_ID

# The command functions import the pipeline (pyarrow, dupekit, msgspec, fsspec) themselves: ``atlantic-cc`` imports
# every stage's cli module to list its commands, so this module must stay cheap to import.

logger = logging.getLogger(__name__)


def _init(args: argparse.Namespace) -> None:
    from atlantic_cc.dedup.runs import PartitionOverrides, init_run  # noqa: PLC0415 - deferred, see above

    init_run(
        run_root=args.run_root,
        output_root=args.output_root,
        kept=args.kept,
        listing=args.listing,
        warc_hashes=args.warc_hashes,
        limit_files=args.limit_files,
        expect_files=args.expect_files,
        unit_bytes=int(args.unit_gb * 1e9),
        node_partitions=args.node_partitions,
        files_per_output=args.files_per_output,
        partitions=PartitionOverrides(exact=args.exact_partitions, lsh=args.lsh_partitions, c075=args.c075_partitions),
        spec_id=args.spec_id,
    )


def _run(args: argparse.Namespace) -> None:
    from atlantic_cc.dedup.local_runner import run_local  # noqa: PLC0415 - deferred, see above
    from atlantic_cc.io.resolve import store_for  # noqa: PLC0415 - deferred, see above

    apply_simd_policy(SimdPolicy(args.simd))
    record_runtime(store_for(args.run_root), args.run_root, "run")
    run_local(args.run_root, args.cpus, args.memory_gb, keep_intermediates=args.keep_intermediates)


def _worker(args: argparse.Namespace) -> None:
    from atlantic_cc.dedup.worker import supervise  # noqa: PLC0415 - deferred, see above
    from atlantic_cc.io.resolve import store_for  # noqa: PLC0415 - deferred, see above

    apply_simd_policy(SimdPolicy(args.simd))
    record_runtime(store_for(args.run_root), args.run_root, "worker")
    claims = ClaimConfig(heartbeat_seconds=args.heartbeat_seconds, stale_seconds=args.stale_seconds)
    supervise(args.run_root, args.cpus, args.memory_gb, claims, keep_intermediates=args.keep_intermediates)


def _status(args: argparse.Namespace) -> None:
    from atlantic_cc.dedup.runs import stage_progress  # noqa: PLC0415 - deferred, see above
    from atlantic_cc.io.resolve import store_for  # noqa: PLC0415 - deferred, see above

    for row in stage_progress(store_for(args.run_root), args.run_root):
        print(json.dumps(row))


def _verify(args: argparse.Namespace) -> None:
    from atlantic_cc.dedup.verify import verify_runs  # noqa: PLC0415 - deferred, see above
    from atlantic_cc.io.resolve import store_for  # noqa: PLC0415 - deferred, see above

    report = verify_runs(args.candidate, args.reference, args.procs, intermediates=not args.skip_intermediates)
    store_for(args.report).write(args.report, json.dumps(report, indent=1, sort_keys=True).encode())
    criteria = {name: result.get("pass", result.get("equal")) for name, result in report["criteria"].items()}
    logger.info("verify pass=%s criteria=%s report=%s", report["pass"], criteria, args.report)
    if not report["pass"]:
        sys.exit(1)


def _add_execution_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--cpus", type=int, required=True, help="Cores this host may use.")
    parser.add_argument("--memory-gb", type=float, required=True, help="RAM budget tasks are scheduled against.")
    parser.add_argument(
        "--simd",
        choices=[p.value for p in SimdPolicy],
        default=SimdPolicy.MASK_AVX512.value,
        help="mask-avx512 (default) reproduces the release's AVX2 tie order in the c075 solver on any x86_64 host.",
    )
    parser.add_argument(
        "--keep-intermediates", action="store_true", help="Keep dead scratch data (for audits); outputs are unchanged."
    )


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add ``dedup`` and its subcommands to the ``atlantic-cc`` parser."""
    parser = subparsers.add_parser("dedup", help="Global exact + MinHash/LSH + c075 containment dedup of kept/.")
    commands = parser.add_subparsers(dest="dedup_command", required=True)

    init = commands.add_parser("init", help="Freeze the inputs and plan of a run.")
    init.add_argument("--run-root", required=True, help="Scratch root: a local directory or gs:// prefix.")
    init.add_argument("--output-root", required=True, help="Receives deduped_c075/ and stats/.")
    init.add_argument("--kept", default=None, help="The kept/ prefix to list (filters --listing when both are given).")
    init.add_argument("--listing", default=None, help="Parquet of (path, size) to use instead of listing --kept.")
    init.add_argument("--warc-hashes", default=None, help="Subset manifest: parquet with warc_hash, or one per line.")
    init.add_argument("--limit-files", type=int, default=None, help="Keep only the first N files (smoke runs).")
    init.add_argument("--expect-files", type=int, default=None)
    init.add_argument("--unit-gb", type=float, default=UNIT_BYTES / 1e9, help="Compressed kept bytes per unit (GB).")
    init.add_argument("--node-partitions", type=int, default=None)
    init.add_argument("--files-per-output", type=int, default=FILES_PER_OUTPUT)
    init.add_argument("--exact-partitions", type=int, default=None, help="Override (scheduling only).")
    init.add_argument("--lsh-partitions", type=int, default=None, help="Override (scheduling only).")
    init.add_argument("--c075-partitions", type=int, default=None, help="Override (scheduling only).")
    init.add_argument("--spec-id", default=PRODUCTION_SPEC_ID, help="Label stored in plan.json.")
    init.set_defaults(func=_init)

    run = commands.add_parser("run", help="Run every remaining stage on this host (resumable).")
    _add_execution_flags(run)
    run.set_defaults(func=_run)

    worker = commands.add_parser("worker", help="Claim and run tasks until the run is sealed (multi-host).")
    _add_execution_flags(worker)
    worker.add_argument("--stale-seconds", type=float, default=ClaimConfig.stale_seconds)
    worker.add_argument("--heartbeat-seconds", type=float, default=ClaimConfig.heartbeat_seconds)
    worker.set_defaults(func=_worker)

    status = commands.add_parser("status", help="Per-stage progress of a run.")
    status.add_argument("--run-root", required=True)
    status.set_defaults(func=_status)

    verify = commands.add_parser("verify", help="Compare two runs exactly (parity criteria 1-5); exit 1 on mismatch.")
    verify.add_argument("--candidate", required=True, help="Run root under test.")
    verify.add_argument("--reference", required=True, help="Run root to compare against.")
    verify.add_argument("--report", required=True, help="Where to write the JSON report (local path or URL).")
    verify.add_argument("--procs", type=int, default=os.cpu_count(), help="Processes hashing output files.")
    verify.add_argument(
        "--skip-intermediates", action="store_true", help="Compare outputs and stats only (criterion 1 skipped)."
    )
    verify.set_defaults(func=_verify)


def main(argv: list[str] | None = None) -> None:
    """Standalone entry point: ``python -m atlantic_cc.dedup.cli dedup ...``."""
    parser = argparse.ArgumentParser(prog="atlantic-cc", description=__doc__)
    register(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
