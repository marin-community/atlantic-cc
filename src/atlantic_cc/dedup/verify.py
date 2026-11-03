# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare two dedup runs exactly: the parity check of ``tests/parity/dedup``.

A run is named by its run root (``plan.json``, ``inputs.parquet``, ``stages/``, ``data/``); its outputs are found
through the plan's output root. The pass criteria, all exact:

1. survivor and removal sets: per unit, the ``reps`` (unique docs with copy counts), ``attrs`` (component members
   with root, canonical flag and size) and ``c075_marks`` (removed keys) arrays, byte for byte, and
   ``oversized.json``;
2. ``(doc_id, exact_copies, fuzzy_cluster_size)`` of every survivor: implied by 3, and checked record by record in
   any output file that differs;
3. the set of output file names and the decompressed bytes of every output file;
4. ``stats/rdedup_stats.json`` without timing, worker, deletion and path fields (and without partition-dependent
   fields when the two plans used different partition counts);
5. the ``.gz`` bytes: informational, as they also depend on the zlib build.
"""

import gzip
import hashlib
import io
import json
import logging
import multiprocessing
import os
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

from atlantic_cc.dedup import blocks
from atlantic_cc.dedup.params import OUTPUT_DIR
from atlantic_cc.dedup.plan import Plan, inputs_sha256, load_inputs, load_plan
from atlantic_cc.dedup.runtime import runtime_info
from atlantic_cc.dedup.stages import ATTR_DTYPE, KEY_DTYPE, REPS_DTYPE, STATS_PATH
from atlantic_cc.io.resolve import store_for
from atlantic_cc.io.store import Store, read_json

logger = logging.getLogger(__name__)

# Stats fields that legitimately differ between runs of the same inputs.
_RUN_SPECIFIC_KEYS = frozenset({"deleted_objects", "solve_seconds", "run_root", "output_root"})
# Stats fields that depend on partition counts (shuffle index sizes, per-partition request and candidate counts).
_PARTITION_KEYS = frozenset({"exact_index", "copies_index", "sizes_index", "in_records", "requests"})
_READ_CHUNK = 1 << 24
_OUTPUT_SUFFIX = ".jsonl.gz"
_EXAMPLES = 5
_INTERMEDIATE_THREADS = 16


class RunHandle:
    """A run root, its store and its plan; done marks are cached as they are read."""

    def __init__(self, run_root: str) -> None:
        self.run_root = run_root.rstrip("/")
        self.store: Store = store_for(run_root)
        self.plan: Plan = load_plan(self.store, run_root)
        self._done: dict[tuple[str, int], dict] = {}

    def done(self, stage: str, task: int) -> dict:
        if (stage, task) not in self._done:
            self._done[stage, task] = read_json(self.store, f"{self.run_root}/stages/{stage}/done/{task:06d}.json")
        return self._done[stage, task]

    def output_dir(self) -> str:
        return f"{self.plan.output_root}/{OUTPUT_DIR}/"

    def _array(self, meta: dict, dtype: np.dtype) -> np.ndarray:
        # Unpinned: generations are store-specific (the as-run LocalStore counts writes in sidecar files).
        return blocks.read_single_block_array(self.store, {**meta, "generation": None}, dtype)

    def unit_arrays(self, unit: int) -> dict[str, np.ndarray]:
        partition = unit // self.plan.units_per_node_partition
        return {
            "reps": self._array(self.done("lshmap", unit)["reps"], REPS_DTYPE),
            "attrs": self._array(self.done("attrs", partition)["units"][str(unit)], ATTR_DTYPE),
            "c075_marks": self._array(self.done("c075_marks", partition)["units"][str(unit)], KEY_DTYPE),
        }


# ------------------------------------------------------------------------------------- plans and inputs


def _layout(plan: Plan) -> dict:
    return {"n_files": plan.n_files, "unit_starts": plan.unit_starts, "files_per_output": plan.files_per_output}


def _partitions(plan: Plan) -> dict:
    return {
        "exact": plan.exact_partitions,
        "lsh": plan.lsh_partitions,
        "units_per_node_partition": plan.units_per_node_partition,
        "c075": plan.c075_partitions,
    }


def compare_plans(candidate: RunHandle, reference: RunHandle) -> dict:
    return {
        "same_layout": _layout(candidate.plan) == _layout(reference.plan),
        "same_partitions": _partitions(candidate.plan) == _partitions(reference.plan),
        "candidate": {"units": candidate.plan.n_units, **_partitions(candidate.plan)},
        "reference": {"units": reference.plan.n_units, **_partitions(reference.plan)},
    }


def compare_inputs(candidate: RunHandle, reference: RunHandle) -> dict:
    ours, theirs = load_inputs(candidate.store, candidate.run_root), load_inputs(reference.store, reference.run_root)
    out = {"equal": ours == theirs, "files": len(ours), "inputs_sha256": inputs_sha256(ours)}
    if ours != theirs:
        first = next(
            (i for i, (a, b) in enumerate(zip(ours, theirs, strict=False)) if a != b), min(len(ours), len(theirs))
        )
        out.update(reference_files=len(theirs), reference_sha256=inputs_sha256(theirs), first_difference=first)
    return out


# ------------------------------------------------------------------------------------- criterion 1


def _array_difference(ours: np.ndarray, theirs: np.ndarray) -> dict:
    out: dict = {"candidate_rows": len(ours), "reference_rows": len(theirs)}
    keys_ours, keys_theirs = (ours["d"], theirs["d"]) if ours.dtype.names else (ours, theirs)
    only_ours = np.setdiff1d(keys_ours, keys_theirs)
    only_theirs = np.setdiff1d(keys_theirs, keys_ours)
    out.update(
        only_candidate=len(only_ours),
        only_reference=len(only_theirs),
        only_candidate_examples=[int(k) for k in only_ours[:_EXAMPLES]],
        only_reference_examples=[int(k) for k in only_theirs[:_EXAMPLES]],
    )
    if ours.dtype.names:
        common, at_ours, at_theirs = np.intersect1d(keys_ours, keys_theirs, return_indices=True)
        differing = ours[at_ours] != theirs[at_theirs]
        out["common_with_different_fields"] = int(differing.sum())
        out["different_field_examples"] = [int(k) for k in common[differing][:_EXAMPLES]]
    return out


def compare_intermediates(candidate: RunHandle, reference: RunHandle) -> dict:
    """Criterion 1: per-unit reps, attrs and c075 marks, byte for byte, plus ``oversized.json``."""

    def unit_result(unit: int) -> dict[str, dict]:
        ours, theirs = candidate.unit_arrays(unit), reference.unit_arrays(unit)
        out = {}
        for kind in ("reps", "attrs", "c075_marks"):
            equal = ours[kind].dtype == theirs[kind].dtype and ours[kind].tobytes() == theirs[kind].tobytes()
            out[kind] = {"equal": equal, "rows": len(ours[kind])}
            if not equal:
                out[kind].update(_array_difference(ours[kind], theirs[kind]))
        return out

    with ThreadPoolExecutor(_INTERMEDIATE_THREADS) as pool:
        per_unit = list(pool.map(unit_result, range(candidate.plan.n_units)))
    summary: dict = {}
    for kind in ("reps", "attrs", "c075_marks"):
        mismatched = [u for u, r in enumerate(per_unit) if not r[kind]["equal"]]
        summary[kind] = {
            "units": len(per_unit),
            "equal_units": len(per_unit) - len(mismatched),
            "rows": sum(r[kind]["rows"] for r in per_unit),
            "mismatched_units": {str(u): per_unit[u][kind] for u in mismatched[:_EXAMPLES]},
            "mismatched_unit_count": len(mismatched),
        }
    oversized = [read_json(h.store, f"{h.run_root}/data/oversized.json") for h in (candidate, reference)]
    summary["oversized"] = {"equal": oversized[0] == oversized[1], "candidate": oversized[0]}
    if not summary["oversized"]["equal"]:
        summary["oversized"]["reference"] = oversized[1]
    summary["pass"] = summary["oversized"]["equal"] and all(
        summary[kind]["mismatched_unit_count"] == 0 for kind in ("reps", "attrs", "c075_marks")
    )
    return summary


# ------------------------------------------------------------------------------------- criteria 2, 3, 5


def _decompressed_chunks(data: bytes) -> Iterator[bytes]:
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as handle:
        while chunk := handle.read(_READ_CHUNK):
            yield chunk


def output_digest(path: str) -> dict:
    """sha256 of a ``.jsonl.gz`` output and of its decompressed bytes, with sizes and the line count."""
    data = store_for(path).read(path)
    digest = hashlib.sha256()
    size = lines = 0
    for chunk in _decompressed_chunks(data):
        digest.update(chunk)
        size += len(chunk)
        lines += chunk.count(b"\n")
    return {
        "gz_sha256": hashlib.sha256(data).hexdigest(),
        "gz_bytes": len(data),
        "sha256": digest.hexdigest(),
        "bytes": size,
        "lines": lines,
    }


def _survivor_tuples(lines: list[bytes]) -> list[tuple]:
    return [
        (r["doc_id"], r["exact_copies"], r["fuzzy_cluster_size"]) for r in (json.loads(line) for line in lines if line)
    ]


def output_difference(candidate_path: str, reference_path: str) -> dict:
    """Where two output files differ: line counts, first differing records, and the survivor-tuple check."""
    ours = b"".join(_decompressed_chunks(store_for(candidate_path).read(candidate_path))).split(b"\n")
    theirs = b"".join(_decompressed_chunks(store_for(reference_path).read(reference_path))).split(b"\n")
    differing = [i for i, (a, b) in enumerate(zip(ours, theirs, strict=False)) if a != b]
    examples = []
    for i in differing[:_EXAMPLES]:
        a, b = json.loads(ours[i]), json.loads(theirs[i])
        examples.append(
            {
                "line": i,
                "candidate_doc_id": a["doc_id"],
                "reference_doc_id": b["doc_id"],
                "fields": sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k)),
            }
        )
    tuples_ours, tuples_theirs = _survivor_tuples(ours), _survivor_tuples(theirs)
    return {
        "candidate_lines": len(ours) - 1,
        "reference_lines": len(theirs) - 1,
        "differing_lines": len(differing) + abs(len(ours) - len(theirs)),
        "examples": examples,
        "survivor_tuples_equal": tuples_ours == tuples_theirs,
        "survivor_doc_ids_only_candidate": len({t[0] for t in tuples_ours} - {t[0] for t in tuples_theirs}),
        "survivor_doc_ids_only_reference": len({t[0] for t in tuples_theirs} - {t[0] for t in tuples_ours}),
    }


def _digest_pair(candidate_path: str, reference_path: str) -> tuple[dict, dict]:
    return output_digest(candidate_path), output_digest(reference_path)


def compare_outputs(candidate: RunHandle, reference: RunHandle, procs: int) -> dict:
    """Criteria 2, 3 and 5 over every output file."""
    names_ours, names_theirs = (
        {os.path.basename(i.path) for i in h.store.list(h.output_dir()) if i.path.endswith(_OUTPUT_SUFFIX)}
        for h in (candidate, reference)
    )
    common = sorted(names_ours & names_theirs)
    pairs = [(candidate.output_dir() + name, reference.output_dir() + name) for name in common]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max(1, min(procs, len(pairs))), mp_context=context) as pool:
        digests = list(pool.map(_digest_pair, *zip(*pairs, strict=True))) if pairs else []
    mismatched = [name for name, (a, b) in zip(common, digests, strict=True) if a["sha256"] != b["sha256"]]
    gz_equal = sum(1 for a, b in digests if a["gz_sha256"] == b["gz_sha256"])
    details = {
        name: output_difference(candidate.output_dir() + name, reference.output_dir() + name)
        for name in mismatched[:_EXAMPLES]
    }
    names_equal = names_ours == names_theirs
    decompressed_pass = names_equal and not mismatched
    return {
        "3_decompressed_outputs": {
            "pass": decompressed_pass,
            "names_equal": names_equal,
            "files": len(names_ours),
            "reference_files": len(names_theirs),
            "only_candidate": sorted(names_ours - names_theirs)[:_EXAMPLES],
            "only_reference": sorted(names_theirs - names_ours)[:_EXAMPLES],
            "equal_files": len(common) - len(mismatched),
            "mismatched_files": len(mismatched),
            "mismatched_details": details,
            "docs": sum(a["lines"] for a, _ in digests),
            "decompressed_bytes": sum(a["bytes"] for a, _ in digests),
            "candidate_sha256_of_digests": hashlib.sha256(
                "".join(a["sha256"] for a, _ in digests).encode()
            ).hexdigest(),
        },
        "2_survivor_tuples": {
            "pass": decompressed_pass and all(d["survivor_tuples_equal"] for d in details.values()),
            "method": "implied by identical decompressed outputs; checked record by record in differing files",
        },
        "5_gz_bytes": {
            "equal": gz_equal == len(common) and names_equal,
            "equal_files": gz_equal,
            "note": "gzip bytes also depend on the zlib build (header OS byte, deflate implementation)",
        },
    }


# ------------------------------------------------------------------------------------- criterion 4


def normalized_stats(stats: dict, same_partitions: bool) -> dict:
    """``rdedup_stats.json`` without fields that differ between equivalent runs."""
    drop = _RUN_SPECIFIC_KEYS if same_partitions else _RUN_SPECIFIC_KEYS | _PARTITION_KEYS

    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if not k.startswith("_") and k not in drop}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    cleaned = clean(stats)
    if not same_partitions:
        # Round 0 records changed = -1 per node partition.
        cleaned["stages"]["components"]["rounds"][0].pop("changed")
    return cleaned


def _flat(value, prefix: str = "") -> dict[str, object]:
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            out.update(_flat(v, f"{prefix}{k}."))
        return out
    if isinstance(value, list) and value and isinstance(value[0], dict):
        out = {}
        for i, v in enumerate(value):
            out.update(_flat(v, f"{prefix}{i}."))
        return out
    return {prefix.rstrip("."): value}


def compare_stats(candidate: RunHandle, reference: RunHandle, same_partitions: bool) -> dict:
    ours, theirs = (
        normalized_stats(read_json(store_for(p), p), same_partitions)
        for p in (f"{candidate.plan.output_root}/{STATS_PATH}", f"{reference.plan.output_root}/{STATS_PATH}")
    )
    flat_ours, flat_theirs = _flat(ours), _flat(theirs)
    differences = [
        {"key": k, "candidate": flat_ours.get(k), "reference": flat_theirs.get(k)}
        for k in sorted(flat_ours.keys() | flat_theirs.keys())
        if flat_ours.get(k) != flat_theirs.get(k)
    ]
    return {
        "pass": not differences,
        "compared_fields": len(flat_ours),
        "partition_dependent_fields_dropped": not same_partitions,
        "differences": differences[:50],
        "headline": {k: ours.get(k) for k in ("records_in", "exact_duplicates", "unique_docs", "components", "c075")},
    }


# ------------------------------------------------------------------------------------- report


def runtime_records(handle: RunHandle) -> list[dict]:
    """The runtime notes a run left under ``runtime/`` (host, CPU flags, NumPy dispatch), summarized."""
    out = []
    for info in handle.store.list(f"{handle.run_root}/runtime/"):
        if not info.path.endswith(".json"):
            continue
        record = read_json(handle.store, info.path)
        flags = next((line for line in record.get("lscpu", "").splitlines() if line.startswith("Flags:")), "")
        model = next((line for line in record.get("lscpu", "").splitlines() if line.startswith("Model name:")), "")
        out.append(
            {
                "role": record.get("role"),
                "host": record["host"],
                "machine": record["machine"],
                "cpu_model": model.removeprefix("Model name:").strip(),
                "cpu_has_avx512f": " avx512f " in f" {flags} ",
                "numpy_avx512_enabled": [f for f in record["numpy_cpu_enabled"] if f.startswith("AVX512")],
                "NPY_DISABLE_CPU_FEATURES": record.get("NPY_DISABLE_CPU_FEATURES", ""),
            }
        )
    return out


def verify_runs(candidate_root: str, reference_root: str, procs: int, intermediates: bool = True) -> dict:
    """Every pass criterion for ``candidate_root`` against ``reference_root``; ``report["pass"]`` is the verdict."""
    started = time.time()
    candidate, reference = RunHandle(candidate_root), RunHandle(reference_root)
    plans = compare_plans(candidate, reference)
    inputs = compare_inputs(candidate, reference)
    criteria: dict = {}
    if plans["same_layout"] and inputs["equal"]:
        if intermediates:
            criteria["1_survivor_and_removal_sets"] = compare_intermediates(candidate, reference)
        criteria.update(compare_outputs(candidate, reference, procs))
    criteria["4_stats"] = compare_stats(candidate, reference, plans["same_partitions"])
    required = ["2_survivor_tuples", "3_decompressed_outputs", "4_stats"]
    if intermediates:
        required.insert(0, "1_survivor_and_removal_sets")
    verdict = plans["same_layout"] and inputs["equal"] and all(criteria.get(k, {}).get("pass") for k in required)
    return {
        "pass": verdict,
        "candidate": candidate.run_root,
        "reference": reference.run_root,
        "candidate_output_root": candidate.plan.output_root,
        "reference_output_root": reference.plan.output_root,
        "plans": plans,
        "inputs": inputs,
        "criteria": criteria,
        "runtime": {"candidate": runtime_records(candidate), "reference": runtime_records(reference)},
        "verifier": {k: v for k, v in runtime_info().items() if k not in ("numpy_show_runtime", "lscpu")},
        "elapsed_seconds": round(time.time() - started, 1),
    }
