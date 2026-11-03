# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate the dedup goldens from the AS-RUN Marin code (tag ``lpv11-8m-as-run``). Maintainers only.

Runs inside an export of the tag, with the tag's ``lib/*/src`` ahead of everything else on ``PYTHONPATH``::

    git -C marin archive lpv11-8m-as-run | tar -x -C asrun
    cd asrun && PYTHONPATH=.:lib/marin/src:lib/zephyr/src:lib/fray/src:lib/rigging/src:lib/iris/src:lib/dupekit/src \\
        python <atlantic-cc>/tests/golden/dedup/generate_asrun.py --work /tmp/goldens \\
        --out <atlantic-cc>/tests/golden/dedup/asrun_goldens.json [--platform-only]

It writes digests only (no text): per synthetic pipeline variant the output file names with the sha256 and line
count of each decompressed file, sha256 of every unit's reps / attrs / c075 marks, and the normalized stats; the
removals of the as-run solve loop on the ``test_solve`` clusters; and as-run encoded records with awkward values.
``--platform-only`` regenerates just the solver golden that depends on NumPy's SIMD tie order (batches over 256
members) and prints it, for adding another platform's entry.
"""

import argparse
import contextlib
import dataclasses
import glob
import gzip
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from experiments.baseline_collection.verified_dedup import C075_PARAMS
from experiments.fast_curation.rdedup import blocks, docs, stages, worker
from experiments.fast_curation.rdedup.plan import build_plan, write_plan
from experiments.fast_curation.rdedup.store import LocalStore
from experiments.fast_curation.rdedup.tasks import ClaimConfig
from marin.processing.classification.deduplication import cluster_verify
from numpy._core import _multiarray_umath

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "unit" / "dedup"))
import synthetic  # noqa: E402

PIPELINE_PLAN = {
    "unit_starts": [0, 3, 6, 9],
    "exact_partitions": 3,
    "lsh_partitions": 5,
    "units_per_node_partition": 2,
    "c075_partitions": 3,
    "files_per_output": 2,
}
# Monkeypatched limits that drive the oversized split, the 64 MiB pre-cut, the 8 MiB cut, the longest-first batches
# and the mid-cluster flush on the small synthetic corpus.
SMALL_CAPS = {
    "MAX_CLUSTER_SIZE": 6,
    "TEXT_MAX_DOCUMENT_CHARS": 1300,
    "SOLVE_MAX_DOCUMENT_CHARS": 1000,
    "SOLVE_MAX_CLUSTER_CHARS": 2500,
}
_RUN_SPECIFIC_KEYS = {"deleted_objects", "solve_seconds", "run_root", "output_root"}


def platform_tag() -> str:
    """Which argpartition kernel NumPy dispatches (as ``atlantic_cc.dedup.runtime.tie_order_platform``)."""
    features = _multiarray_umath.__cpu_features__
    if platform.machine() not in ("x86_64", "AMD64"):
        return f"{platform.machine()}-scalar"
    if features.get("AVX512_SKX"):
        return "x86_64-avx512"
    return "x86_64-avx2" if features.get("AVX2") else "x86_64-scalar"


def tie_probe() -> dict[str, list[int]]:
    """Which indices ``np.argpartition`` keeps among ties, in the two shapes the solver calls it with.

    Must equal ``tests/unit/dedup/test_u_dedup_tie_order.py::tie_probe``.
    """
    rng = np.random.default_rng(2026)
    counts = rng.integers(1, 6, 500).astype(np.int64)
    shared = rng.integers(1, 5, 300).astype(np.int64)
    return {
        "rarest_32": sorted(np.argpartition(counts, 32)[:32].tolist()),
        "best_32": sorted(np.argpartition(-shared, 31)[:32].tolist()),
    }


def normalized_stats(stats):
    if isinstance(stats, dict):
        return {
            k: normalized_stats(v) for k, v in stats.items() if not k.startswith("_") and k not in _RUN_SPECIFIC_KEYS
        }
    if isinstance(stats, list):
        return [normalized_stats(v) for v in stats]
    return stats


@contextlib.contextmanager
def patched(module, values: dict):
    """Set module globals for the duration of the block (the as-run tests' monkeypatching)."""
    saved = {name: getattr(module, name) for name in values}
    for name, value in values.items():
        setattr(module, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(module, name, value)


def run_pipeline(work: Path, kept: Path, patches: dict) -> dict:
    """The as-run pipeline (one in-process slot) on ``kept`` with the fixed plan; digests of everything it wrote."""
    with patched(stages, patches):
        return _run_pipeline(work, kept)


def _run_pipeline(work: Path, kept: Path) -> dict:
    paths = sorted(str(p) for p in kept.glob("*.parquet"))
    sizes = [os.path.getsize(p) for p in paths]
    plan = build_plan(
        run_root=str(work / "run"),
        output_root=str(work / "out"),
        spec_id="test",
        sizes=sizes,
        unit_bytes=10**12,
        rules=["c075"],
    )
    plan = dataclasses.replace(plan, **PIPELINE_PLAN)
    write_plan(LocalStore(), plan, paths, sizes)
    assert worker.slot_loop(0, plan.run_root, cpus=1, memory_gb=64, claims=ClaimConfig()) == 0
    store = LocalStore()
    outputs = {}
    for path in sorted(glob.glob(f"{plan.output_root}/deduped_c075/*.jsonl.gz")):
        payload = gzip.decompress(Path(path).read_bytes())
        outputs[os.path.basename(path)] = {"sha256": hashlib.sha256(payload).hexdigest(), "lines": payload.count(b"\n")}
    units = {}
    pipeline = stages.Pipeline(store, plan, procs=1)
    for unit in range(plan.n_units):
        partition = unit // plan.units_per_node_partition
        arrays = {
            "reps": blocks.read_single_block_array(store, pipeline.ctx.done("lshmap", unit)["reps"], stages.REPS_DTYPE),
            "attrs": stages._attrs_of_unit(pipeline.ctx, unit),
            "c075_marks": blocks.read_single_block_array(
                store, pipeline.ctx.done("c075_marks", partition)["units"][str(unit)], stages.KEY_DTYPE
            ),
        }
        units[str(unit)] = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()}
    stats = json.loads(Path(plan.output_root, "stats", "rdedup_stats.json").read_text())
    return {"outputs": outputs, "units": units, "stats": normalized_stats(stats)}


def solver_removals(rows: list[dict], cap: int, document_cap: int) -> dict:
    """Removed ids of the as-run solve loop (``cluster_verify.solve_text_shard``) and of rdedup's port of it."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "text.parquet")
        pq.write_table(
            pa.table(
                {
                    "cluster_key": [r["cluster_key"] for r in rows],
                    "dup_cluster_id": [r["cluster_key"] for r in rows],
                    "id": [r["id"] for r in rows],
                    "text": pa.array([r["text"] for r in rows], type=pa.large_string()),
                    "file_idx": [0] * len(rows),
                    "source_tag": ["source_000"] * len(rows),
                }
            ),
            path,
        )
        limits = {"MAXIMUM_CLUSTER_CHARS": cap, "MAXIMUM_DOCUMENT_CHARS": document_cap}
        ported_limits = {"SOLVE_MAX_CLUSTER_CHARS": cap, "SOLVE_MAX_DOCUMENT_CHARS": document_cap}
        with patched(cluster_verify, limits), patched(stages, ported_limits):
            reference = sorted(m["id"] for m in cluster_verify.solve_text_shard(path, C075_PARAMS))
            solve_stats = stages.SolveStats()
            id_of = {r["d"]: r["id"] for r in rows}
            ported = sorted(id_of[d] for d in stages.solve_sorted_rows([dict(r) for r in rows], solve_stats))
    if ported != reference:
        raise AssertionError("rdedup's solve loop disagrees with cluster_verify.solve_text_shard")
    return {"removed_ids": reference, "stats": dataclasses.asdict(solve_stats)}


def solver_goldens(big_cluster: int) -> dict:
    out = {}
    for cap in (256 * 1024 * 1024, 2500):
        out[str(cap)] = solver_removals(synthetic.solve_rows(cap, big_cluster), cap, 400)
    return out


def record_golden() -> dict:
    """As-run bytes of output records holding NaN, Inf, tiny and huge floats, U+2028, NUL and non-ASCII text."""
    rows = [
        ("a", 1e-05, "plain text", 0.9802150130271912, float("nan")),
        ("b", float("inf"), "line\u2028sep\u2029par \x00 nul \x1f unit", float("nan"), 0.5),
        ("c", 1e16, 'ΟΔΥΣΣΕΥΣ İ ß 日本語 😀 "quoted" back\\slash', float("-inf"), 1.0),
        ("d", 0.1, "x" + " " * 300 + "y", 0.0, 0.0048),
    ]
    table = pa.Table.from_pylist(
        [
            {
                "doc_id": doc_id,
                "url": f"https://example.com/{doc_id}",
                "warc_hash": "0123456789ab",
                "snapshot": "CC-MAIN-2024-10",
                "fasttext_score": score,
                "text": text,
                "n_tokens": 7,
                "pooled_prob": pooled,
                "modernbert_prob": terminal,
            }
            for doc_id, score, text, pooled, terminal in rows
        ],
        schema=synthetic.KEPT_SCHEMA,
    ).drop_columns(["n_tokens"])
    records, texts = docs.output_records(table, np.arange(len(rows)))
    for i, record in enumerate(records):
        record[stages.EXACT_COPIES_FIELD] = i + 1
        record[stages.CLUSTER_SIZE_FIELD] = 4_557_202_931 if i == 2 else 1
    payload = docs.encode_jsonl(records)
    member = gzip.compress(payload, compresslevel=stages.GZIP_LEVEL, mtime=0)
    return {
        "jsonl_hex": payload.hex(),
        "gzip_header_hex": member[:10].hex(),
        "text_lengths": [len(t) for t in texts],
        "nan_is_finite_or_none": [
            r["modernbert_prob"] is None or not math.isnan(r["modernbert_prob"]) for r in records
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--platform-only", action="store_true")
    args = parser.parse_args()
    tag = platform_tag()
    index_path = {"platform": tag, "goldens": solver_goldens(big_cluster=300), "tie_probe": tie_probe()}
    if args.platform_only:
        print(json.dumps(index_path, indent=1, sort_keys=True))
        return
    kept = args.work / "kept"
    synthetic.write_corpus(kept)
    goldens = {
        "generated_on": {"platform": tag, "python": sys.version.split()[0], "numpy": np.__version__},
        "pipeline_plan": PIPELINE_PLAN,
        "small_caps": SMALL_CAPS,
        "pipeline": {
            "default": run_pipeline(args.work / "default", kept, {}),
            "small_caps": run_pipeline(args.work / "small_caps", kept, SMALL_CAPS),
        },
        "solver_exact_scan": solver_goldens(big_cluster=200),
        "solver_index_path": {tag: index_path["goldens"]},
        "tie_probe": {tag: index_path["tie_probe"]},
        "records": record_golden(),
    }
    args.out.write_text(json.dumps(goldens, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v["stats"]["c075"] for k, v in goldens["pipeline"].items()}, indent=1))
    print(json.dumps({k: v["stats"]["stages"]["c075_solve"] for k, v in goldens["pipeline"].items()}, indent=1))


if __name__ == "__main__":
    main()
