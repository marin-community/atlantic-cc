# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The dedup pipeline as a sequence of stages of independent tasks (as-run ``rdedup/stages.py``).

Stage by stage (``U`` units, ``R_*`` partitions; every arrow is a partitioned shuffle object)::

    pass1      (U)    read kept; normalize, hash, MinHash; collapse byte-identical docs inside the unit
                      -> exact records (with copy counts) by content hash, per-unit stash
    exact      (R_e)  global exact dedup: the minimum doc key per content hash is kept
                      -> dropped doc keys by unit; kept keys' copy counts
    lshmap     (U)    stash minus exact drops = unique docs -> (bucket, doc) by bucket; reps list
                      (each unique doc's exact copy count)
    bucket     (R_b)  every non-singleton bucket -> (member, hub) by node partition
    ccNNN{a,b,c}(R_n) connected components, one FastSV round per three stages, until a round changes no label
    canon_a    (R_n)  member -> component root; canonical order key per member
    canon_b    (R_n)  canonical member per component; component sizes -> every node partition holding members
    attrs      (R_n)  per-unit member table: (doc, component, is_canonical, component size)
    c075_text  (U)    member text keyed by solve group (oversized components split)
    c075_solve (R_c)  containment rule per group -> removed doc keys by node partition
    c075_marks (R_n)  per-unit removal lists
    apply_c075 (U)    re-read kept, write unique docs minus containment removals
    report     (1)    aggregate counters into ``{output_root}/stats/rdedup_stats.json``

Every output document carries ``exact_copies`` (input documents with its content hash) and ``fuzzy_cluster_size``
(unique documents in its connected component; 1 outside any component).

Every task is deterministic: the same inputs give byte-identical outputs, so a retried or duplicated task is
harmless. Every stage reads only paths derived from the plan and the sealed results of the stage before it,
never a listing of data. Only imports, the store layer and orchestration differ from the as-run code; the as-run
``cc_canonical`` rule and its ``apply_cc`` stage are not ported.
"""

import dataclasses
import logging
import multiprocessing
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor

import dupekit_native as dupekit
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from atlantic_cc.dedup import blocks, graph
from atlantic_cc.dedup.minhash import minhash_buckets
from atlantic_cc.dedup.params import (
    APPROX_JACCARD_THRESHOLD,
    C075_FLUSH_CHARS,
    C075_PARAMS,
    CLUSTER_SIZE_FIELD,
    EXACT_COPIES_FIELD,
    MAX_CLUSTER_SIZE,
    MAX_ROWS_PER_FILE,
    NGRAM_SIZE,
    NUM_BANDS,
    NUM_PERMS,
    OUTPUT_DIR,
    SEED,
    SOLVE_PASS_COMPRESSED_BYTES,
    TEXT_MAX_DOCUMENT_CHARS,
)
from atlantic_cc.dedup.plan import Plan, load_inputs
from atlantic_cc.dedup.records import encode_jsonl, gzip_member, output_records, read_kept
from atlantic_cc.dedup.solver import SolveStats, cluster_key, solve_sorted_rows, sorted_rows
from atlantic_cc.dedup.text import (
    canonical_order_keys,
    compact_whitespace,
    content_id,
    doc_keys,
    key_file,
    key_row,
    normalize_file,
)
from atlantic_cc.io.resolve import store_for
from atlantic_cc.io.store import Store, delete_prefix, dumps, read_json

logger = logging.getLogger(__name__)

# ``n``: documents of the unit collapsed into this representative.
EXACT_DTYPE = np.dtype([("hi", "<u8"), ("lo", "<u8"), ("d", "<u8"), ("n", "<u4")])
# A kept doc key and the input documents sharing its content hash, for counts above 1.
COPIES_DTYPE = np.dtype([("d", "<u8"), ("n", "<u4")])
STASH_DTYPE = np.dtype(
    [("d", "<u8"), ("hi", "<u8"), ("lo", "<u8"), ("chars", "<i8"), ("sig", "u1"), ("bk", "<u8", (NUM_BANDS,))]
)
REPS_DTYPE = np.dtype([("d", "<u8"), ("hi", "<u8"), ("lo", "<u8"), ("chars", "<i8"), ("n", "<u4")])
MEMBER_DTYPE = np.dtype([("d", "<u8"), ("root", "<u8")])
CANON_DTYPE = np.dtype([("root", "<u8"), ("k1", "<u8"), ("k2", "<u8"), ("k3", "<u8"), ("d", "<u8"), ("n", "<i8")])
# Component sizes are u8: the 8M run's largest component had 4,557,202,931 members (> u4).
SIZE_DTYPE = np.dtype([("root", "<u8"), ("size", "<u8")])
ATTR_DTYPE = np.dtype([("d", "<u8"), ("root", "<u8"), ("canon", "u1"), ("size", "<u8")])
KEY_DTYPE = np.dtype("<u8")
LABEL_DTYPE = np.dtype([("node", "<u8"), ("label", "<u8")])
CANDIDATE_DTYPE = np.dtype([("t", "<u8"), ("c", "<u8")])
# Component sizes are binned by powers of two in the report.
_SIZE_BINS = 40
_MAX_COUNT = np.iinfo(np.uint32).max

STATS_PATH = "stats/rdedup_stats.json"


@dataclasses.dataclass(frozen=True)
class Ctx:
    """What a task needs: the stores, the plan, and the sealed results of earlier stages.

    ``keep_intermediates`` skips the deletion of dead scratch data at seals (for audits); it changes no output.
    """

    store: Store
    plan: Plan
    sealed: Callable[[str], dict]
    output_store: Store
    keep_intermediates: bool = False

    def data(self, name: str) -> str:
        return f"{self.plan.run_root}/data/{name}"

    def stage_dir(self, name: str) -> str:
        return f"{self.plan.run_root}/stages/{name}"

    def done(self, stage: str, task: int) -> dict:
        return read_json(self.store, f"{self.stage_dir(stage)}/done/{task:06d}.json")

    def inputs(self) -> list[str]:
        return _cached_inputs(self.store, self.plan.run_root)


_INPUTS_CACHE: dict[str, list[str]] = {}


def _cached_inputs(store: Store, run_root: str) -> list[str]:
    if run_root not in _INPUTS_CACHE:
        _INPUTS_CACHE[run_root] = load_inputs(store, run_root)
    return _INPUTS_CACHE[run_root]


def _delete_dead(ctx: Ctx, names: Iterable[str]) -> int:
    """Delete the scratch data of ``names`` (no stage reads it again); returns the object count."""
    if ctx.keep_intermediates:
        return 0
    return sum(delete_prefix(ctx.store, ctx.data(name) + "/") for name in names)


@dataclasses.dataclass(frozen=True)
class StageSpec:
    """One stage: ``n_tasks`` independent tasks, then a seal that sees every task's result."""

    name: str
    n_tasks: int
    run: Callable[[Ctx, int], dict]
    seal: Callable[[Ctx, list[dict]], dict]
    procs: int
    mem_gb: float
    deadline_hours: float

    def slots(self, cpus: int, mem_gb: float) -> int:
        """Concurrent tasks a worker of this shape can run for this stage."""
        return max(1, min(cpus // self.procs, int(mem_gb // self.mem_gb)))


def _pool(procs: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=procs, mp_context=multiprocessing.get_context("spawn"))


def _map_ordered(fn: Callable, args: list[tuple], procs: int) -> Iterator:
    """``fn(*a)`` for each ``a``, in order; in a spawn pool when ``procs > 1``."""
    if procs <= 1 or len(args) <= 1:
        for a in args:
            yield fn(*a)
        return
    with _pool(min(procs, len(args))) as pool:
        futures = [pool.submit(fn, *a) for a in args]
        for f in futures:
            yield f.result()


def _sum(results: Iterable[dict], key: str) -> int:
    return int(sum(r[key] for r in results))


def _as_count(values: np.ndarray, what: str) -> np.ndarray:
    """``values`` as u4 counts; a count beyond u4 fails loudly instead of wrapping."""
    if len(values) and int(values.max()) > _MAX_COUNT:
        raise RuntimeError(f"{what} {int(values.max())} does not fit a u4 count")
    return values.astype(np.uint32)


def _first_of_runs(*keys: np.ndarray) -> np.ndarray:
    """For arrays sorted by ``keys`` (primary last), mask of the first row of each equal run."""
    n = len(keys[0])
    mask = np.ones(n, dtype=bool)
    if n > 1:
        same = np.ones(n - 1, dtype=bool)
        for k in keys:
            same &= k[1:] == k[:-1]
        mask[1:] = ~same
    return mask


# ---------------------------------------------------------------------------------------- pass1


@dataclasses.dataclass
class FileDocs:
    """What pass 1 needs from one kept file, for its kept rows only."""

    file_index: int
    n_rows: int
    rows: np.ndarray  # int64 row index of each kept document
    id_hi: np.ndarray  # uint64
    id_lo: np.ndarray  # uint64
    chars: np.ndarray  # int64 length of the compacted text
    has_signature: np.ndarray  # bool: dupekit produced buckets
    buckets: np.ndarray  # (n_kept, 26) uint64; zeros where has_signature is False
    compacted: int


def pass1_file(store: Store, path: str, file_index: int) -> FileDocs:
    """Normalize, hash and MinHash one kept file."""
    texts = read_kept(store, path, ["text"]).column("text").to_pylist()
    if len(texts) >= MAX_ROWS_PER_FILE:
        raise ValueError(f"{path} has {len(texts)} rows; doc keys hold at most {MAX_ROWS_PER_FILE}")
    rows, normalized, compacted = normalize_file(texts)
    ids = [content_id(t) for t in normalized]
    id_hi = np.fromiter((i >> 64 for i in ids), dtype=np.uint64, count=len(ids))
    id_lo = np.fromiter((i & 0xFFFFFFFFFFFFFFFF for i in ids), dtype=np.uint64, count=len(ids))
    chars = np.fromiter((len(t) for t in normalized), dtype=np.int64, count=len(normalized))
    has_signature, buckets = minhash_buckets(normalized)
    return FileDocs(file_index, len(texts), rows, id_hi, id_lo, chars, has_signature, buckets, compacted)


def _pass1_file(path: str, file_index: int) -> FileDocs:
    return pass1_file(store_for(path), path, file_index)


def run_pass1(ctx: Ctx, unit: int, procs: int) -> dict:
    plan = ctx.plan
    paths = ctx.inputs()
    files = list(plan.unit_files(unit))
    parts = list(_map_ordered(_pass1_file, [(paths[f], f) for f in files], procs))
    keys = np.concatenate([doc_keys(p.file_index, p.rows) for p in parts])
    hi = np.concatenate([p.id_hi for p in parts])
    lo = np.concatenate([p.id_lo for p in parts])
    chars = np.concatenate([p.chars for p in parts])
    sig = np.concatenate([p.has_signature for p in parts])
    buckets = np.concatenate([p.buckets for p in parts]) if parts else np.zeros((0, NUM_BANDS), np.uint64)

    # Keys increase in file-then-row order, so the first key of each content hash is the unit's earliest copy --
    # and the global rule keeps the earliest copy overall.
    order = np.lexsort((keys, lo, hi))
    starts = np.flatnonzero(_first_of_runs(hi[order], lo[order]))
    by_key = np.argsort(order[starts])
    reps = order[starts][by_key]
    copies = np.diff(np.append(starts, len(order)))[by_key]

    exact = np.empty(len(reps), dtype=EXACT_DTYPE)
    exact["hi"], exact["lo"], exact["d"] = hi[reps], lo[reps], keys[reps]
    exact["n"] = _as_count(copies, "in-unit copies")
    writer = blocks.BlockObjectWriter(plan.exact_partitions)
    writer.add_array(exact, (exact["hi"] % np.uint64(plan.exact_partitions)).astype(np.int64))
    exact_meta = writer.finish(ctx.store, blocks.map_object_path(ctx.data("pass1_exact"), unit))

    stash = np.empty(len(reps), dtype=STASH_DTYPE)
    stash["d"], stash["hi"], stash["lo"] = keys[reps], hi[reps], lo[reps]
    stash["chars"], stash["sig"], stash["bk"] = chars[reps], sig[reps], buckets[reps]
    stash_meta = blocks.write_single_block(ctx.store, blocks.map_object_path(ctx.data("stash"), unit), stash)
    return {
        "exact": exact_meta,
        "stash": stash_meta,
        "files": len(files),
        "rows": sum(p.n_rows for p in parts),
        "kept": len(keys),
        "compacted": sum(p.compacted for p in parts),
        "unit_dups": len(keys) - len(reps),
        "unit_reps": len(reps),
        "no_signature": int((~sig[reps]).sum()),
        "chars_kept": int(chars.sum()),
    }


def seal_pass1(ctx: Ctx, results: list[dict]) -> dict:
    index = blocks.build_index(
        ctx.store, ctx.data("pass1_exact"), [r["exact"] for r in results], ctx.plan.exact_partitions
    )
    totals = {
        k: _sum(results, k)
        for k in ("files", "rows", "kept", "compacted", "unit_dups", "unit_reps", "no_signature", "chars_kept")
    }
    if totals["files"] != ctx.plan.n_files:
        raise RuntimeError(f"pass1 covered {totals['files']} files, plan has {ctx.plan.n_files}")
    return {
        **totals,
        "empty_text_filtered": totals["rows"] - totals["kept"],
        "exact_index": index,
        "stash_bytes": _sum((r["stash"] for r in results), "bytes"),
    }


# ---------------------------------------------------------------------------------------- exact


def run_exact(ctx: Ctx, partition: int) -> dict:
    plan = ctx.plan
    records = blocks.read_partition_array(
        ctx.store, ctx.data("pass1_exact"), partition, plan.n_units, plan.exact_partitions, EXACT_DTYPE
    )
    order = np.lexsort((records["d"], records["lo"], records["hi"]))
    first = _first_of_runs(records["hi"][order], records["lo"][order])
    dropped = np.sort(records["d"][order][~first])
    writer = blocks.BlockObjectWriter(plan.n_units)
    writer.add_array(dropped, plan.unit_of_keys(dropped))
    meta = writer.finish(ctx.store, blocks.map_object_path(ctx.data("exact"), partition))

    starts = np.flatnonzero(first)
    n = records["n"][order].astype(np.int64)
    totals = np.add.reduceat(n, starts) if len(n) else np.zeros(0, dtype=np.int64)
    multi = totals > 1
    copies = np.empty(int(multi.sum()), dtype=COPIES_DTYPE)
    copies["d"], copies["n"] = records["d"][order][starts][multi], _as_count(totals[multi], "exact copies")
    copies = np.sort(copies, order="d")
    writer = blocks.BlockObjectWriter(plan.n_units)
    writer.add_array(copies, plan.unit_of_keys(copies["d"]))
    copies_meta = writer.finish(ctx.store, blocks.map_object_path(ctx.data("exact_copies"), partition))
    return {
        "blocks": meta,
        "copies_blocks": copies_meta,
        "records": len(records),
        "unique": int(first.sum()),
        "cross_unit_dups": len(dropped),
        "exact_copies": int(n.sum()),
        "multi_copy_docs": len(copies),
    }


def seal_exact(ctx: Ctx, results: list[dict]) -> dict:
    index = blocks.build_index(ctx.store, ctx.data("exact"), [r["blocks"] for r in results], ctx.plan.n_units)
    copies_index = blocks.build_index(
        ctx.store, ctx.data("exact_copies"), [r["copies_blocks"] for r in results], ctx.plan.n_units
    )
    pass1 = ctx.sealed("pass1")
    unique = _sum(results, "unique")
    cross = _sum(results, "cross_unit_dups")
    if unique + cross != pass1["unit_reps"]:
        raise RuntimeError(f"exact: {unique} unique + {cross} dropped != {pass1['unit_reps']} unit reps")
    copies = _sum(results, "exact_copies")
    if copies != pass1["kept"]:
        raise RuntimeError(f"exact: copy counts sum to {copies}, pass1 kept {pass1['kept']} records")
    return {
        "unique": unique,
        "cross_unit_dups": cross,
        "exact_dups": cross + pass1["unit_dups"],
        "multi_copy_docs": _sum(results, "multi_copy_docs"),
        "index": index,
        "copies_index": copies_index,
    }


# --------------------------------------------------------------------------------------- lshmap


def _read_stash(ctx: Ctx, unit: int) -> np.ndarray:
    return blocks.read_single_block_array(ctx.store, ctx.done("pass1", unit)["stash"], STASH_DTYPE)


def run_lshmap(ctx: Ctx, unit: int) -> dict:
    plan = ctx.plan
    stash = _read_stash(ctx, unit)
    dropped = blocks.read_partition_array(
        ctx.store, ctx.data("exact"), unit, plan.exact_partitions, plan.n_units, KEY_DTYPE
    )
    unique = stash[~np.isin(stash["d"], dropped)]
    if len(unique) != len(stash) - len(dropped):
        raise RuntimeError(f"unit {unit}: {len(dropped)} exact drops but {len(stash) - len(unique)} matched")
    reps = np.empty(len(unique), dtype=REPS_DTYPE)
    for field in ("d", "hi", "lo", "chars"):
        reps[field] = unique[field]
    copies = blocks.read_partition_array(
        ctx.store, ctx.data("exact_copies"), unit, plan.exact_partitions, plan.n_units, COPIES_DTYPE
    )
    at = np.searchsorted(reps["d"], copies["d"])
    if len(copies) and (at.max() >= len(reps) or np.any(reps["d"][at] != copies["d"])):
        raise RuntimeError(f"unit {unit}: exact copy counts for doc keys that are not unique docs of the unit")
    reps["n"] = 1
    reps["n"][at] = copies["n"]
    reps_meta = blocks.write_single_block(ctx.store, blocks.map_object_path(ctx.data("reps"), unit), reps)

    signed = unique[unique["sig"] == 1]
    lsh = np.empty(len(signed) * NUM_BANDS, dtype=graph.LSH_DTYPE)
    lsh["b"] = signed["bk"].reshape(-1)
    lsh["d"] = np.repeat(signed["d"], NUM_BANDS)
    writer = blocks.BlockObjectWriter(plan.lsh_partitions)
    writer.add_array(lsh, (lsh["b"] % np.uint64(plan.lsh_partitions)).astype(np.int64))
    lsh_meta = writer.finish(ctx.store, blocks.map_object_path(ctx.data("lshmap"), unit))
    return {
        "lsh": lsh_meta,
        "reps": reps_meta,
        "unique": len(unique),
        "signed": len(signed),
        "unique_chars": int(unique["chars"].sum()),
        "exact_copies": int(reps["n"].sum(dtype=np.int64)),
    }


def seal_lshmap(ctx: Ctx, results: list[dict]) -> dict:
    index = blocks.build_index(ctx.store, ctx.data("lshmap"), [r["lsh"] for r in results], ctx.plan.lsh_partitions)
    unique = _sum(results, "unique")
    if unique != ctx.sealed("exact")["unique"]:
        raise RuntimeError(f"lshmap kept {unique} unique docs, exact stage counted {ctx.sealed('exact')['unique']}")
    copies = _sum(results, "exact_copies")
    if copies != ctx.sealed("pass1")["kept"]:
        raise RuntimeError(f"unique docs' exact copies sum to {copies}, pass1 kept {ctx.sealed('pass1')['kept']}")
    deleted = _delete_dead(ctx, ("stash", "pass1_exact", "exact", "exact_copies"))
    return {
        "unique": unique,
        "signed": _sum(results, "signed"),
        "unique_chars": _sum(results, "unique_chars"),
        "index": index,
        "deleted_objects": deleted,
    }


# --------------------------------------------------------------------------------------- bucket


def run_bucket(ctx: Ctx, partition: int) -> dict:
    plan = ctx.plan
    lsh = blocks.read_partition_array(
        ctx.store, ctx.data("lshmap"), partition, plan.n_units, plan.lsh_partitions, graph.LSH_DTYPE
    )
    pairs, stats = graph.bucket_hub_pairs(lsh)
    del lsh
    writer = blocks.BlockObjectWriter(plan.node_partitions)
    writer.add_array(pairs, plan.node_partition_of_keys(pairs["x"]))
    meta = writer.finish(ctx.store, blocks.map_object_path(ctx.data("bucket"), partition))
    return {"blocks": meta, **dataclasses.asdict(stats)}


def seal_bucket(ctx: Ctx, results: list[dict]) -> dict:
    index = blocks.build_index(ctx.store, ctx.data("bucket"), [r["blocks"] for r in results], ctx.plan.node_partitions)
    totals = {k: _sum(results, k) for k in ("pairs", "buckets", "multi_buckets", "multi_members")}
    deleted = _delete_dead(ctx, ("lshmap",))
    return {**totals, "largest_bucket": max(r["largest"] for r in results), "index": index, "deleted_objects": deleted}


# ------------------------------------------------------------------------------ components
#
# Round k runs three stages over the node partitions (see graph.py):
#   cc{k}a  apply the candidates of round k-1 (round 0: build edges and initial labels from the bucket pairs);
#           write labels; request the labels of every hub and current label
#   cc{k}b  answer the requests addressed to this partition's nodes
#   cc{k}c  from the answers, emit min-combined candidates to their nodes' partitions
# A round whose ``a`` stage changes no label has converged; its labels are final.


def cc_name(round_index: int, step: str) -> str:
    return f"cc{round_index:03d}{step}"


def _labels_path(ctx: Ctx, round_index: int, partition: int) -> str:
    return blocks.map_object_path(ctx.data(f"cc_labels/{round_index:03d}"), partition)


def _load_edges(ctx: Ctx, partition: int) -> np.ndarray:
    meta = ctx.done(cc_name(0, "a"), partition)["edges"]
    return blocks.read_single_block_array(ctx.store, meta, graph.HUB_DTYPE)


def _load_labels(ctx: Ctx, round_index: int, partition: int) -> np.ndarray:
    meta = ctx.done(cc_name(round_index, "a"), partition)["labels"]
    return blocks.read_single_block_array(ctx.store, meta, LABEL_DTYPE)


def run_cc_a(ctx: Ctx, partition: int, round_index: int) -> dict:
    plan = ctx.plan
    n = plan.node_partitions
    out: dict = {}
    if round_index == 0:
        pairs = blocks.read_partition_array(
            ctx.store, ctx.data("bucket"), partition, plan.lsh_partitions, n, graph.HUB_DTYPE
        )
        g, labels = graph.initial_graph(pairs)
        edges = np.empty(len(g.ex), dtype=graph.HUB_DTYPE)
        edges["x"], edges["h"] = g.ex, g.eh
        out["edges"] = blocks.write_single_block(
            ctx.store, blocks.map_object_path(ctx.data("cc_edges"), partition), edges
        )
        out.update(in_records=len(pairs), edge_count=len(edges), changed=-1)
        nodes = g.nodes
    else:
        previous = _load_labels(ctx, round_index - 1, partition)
        nodes = previous["node"]
        labels = previous["label"].copy()
        received = 0
        stream = blocks.iter_partition_blocks(ctx.store, ctx.data(cc_name(round_index - 1, "c")), partition, n, n)
        for _m, block in stream:
            for frame in blocks.iter_frames(block):
                cand = np.frombuffer(frame, dtype=CANDIDATE_DTYPE)
                graph.apply_candidates(nodes, labels, cand["t"], cand["c"])
                received += len(cand)
        changed = int(np.count_nonzero(labels != previous["label"]))
        out.update(in_records=received, changed=changed)
        g = graph.PartitionGraph(ex=np.empty(0, np.uint64), eh=_load_edges(ctx, partition)["h"], nodes=nodes)
    state = np.empty(len(nodes), dtype=LABEL_DTYPE)
    state["node"], state["label"] = nodes, labels
    out["labels"] = blocks.write_single_block(ctx.store, _labels_path(ctx, round_index, partition), state)
    targets = graph.request_targets(g, labels)
    writer = blocks.BlockObjectWriter(n)
    writer.add_array(targets, plan.node_partition_of_keys(targets))
    out["blocks"] = writer.finish(ctx.store, blocks.map_object_path(ctx.data(cc_name(round_index, "a")), partition))
    out.update(nodes=len(nodes), requests=len(targets))
    return out


def run_cc_b(ctx: Ctx, partition: int, round_index: int) -> dict:
    plan = ctx.plan
    n = plan.node_partitions
    state = _load_labels(ctx, round_index, partition)
    stream = blocks.iter_partition_blocks(ctx.store, ctx.data(cc_name(round_index, "a")), partition, n, n)
    writer = blocks.BlockObjectWriter(n)
    answered = 0
    for requester, block in stream:
        targets = blocks.concat_arrays(
            [np.frombuffer(f, dtype=KEY_DTYPE) for f in blocks.iter_frames(block)], KEY_DTYPE
        )
        if len(targets) == 0:
            continue
        answers = np.empty(len(targets), dtype=LABEL_DTYPE)
        answers["node"], answers["label"] = targets, graph.lookup(state["node"], state["label"], targets)
        writer.add_array(answers, np.full(len(targets), requester, dtype=np.int64))
        answered += len(targets)
    meta = writer.finish(ctx.store, blocks.map_object_path(ctx.data(cc_name(round_index, "b")), partition))
    return {"blocks": meta, "answered": answered}


def run_cc_c(ctx: Ctx, partition: int, round_index: int) -> dict:
    plan = ctx.plan
    n = plan.node_partitions
    state = _load_labels(ctx, round_index, partition)
    edges = _load_edges(ctx, partition)
    g = graph.PartitionGraph(ex=edges["x"], eh=edges["h"], nodes=state["node"])
    labels = state["label"]
    answers = blocks.read_partition_array(ctx.store, ctx.data(cc_name(round_index, "b")), partition, n, n, LABEL_DTYPE)
    order = np.argsort(answers["node"])
    targets, values = graph.candidates(g, labels, answers["node"][order], answers["label"][order])
    cand = np.empty(len(targets), dtype=CANDIDATE_DTYPE)
    cand["t"], cand["c"] = targets, values
    writer = blocks.BlockObjectWriter(n)
    writer.add_array(cand, plan.node_partition_of_keys(targets))
    meta = writer.finish(ctx.store, blocks.map_object_path(ctx.data(cc_name(round_index, "c")), partition))
    return {"blocks": meta, "candidates": len(cand)}


def seal_cc(ctx: Ctx, results: list[dict], round_index: int, step: str) -> dict:
    n = ctx.plan.node_partitions
    name = cc_name(round_index, step)
    index = blocks.build_index(ctx.store, ctx.data(name), [r["blocks"] for r in results], n)
    counts = {k: _sum(results, k) for k in results[0] if isinstance(results[0][k], int) and not k.startswith("_")}
    out = {"round": round_index, "step": step, **counts, "index": index}
    # Each round's inputs are dead once the next step has sealed.
    stale = {
        "a": [cc_name(round_index - 1, "c")] if round_index else ["bucket"],
        "b": [],
        "c": [name[:-1] + "a", name[:-1] + "b"],
    }
    out["deleted_objects"] = _delete_dead(ctx, stale[step])
    if step == "a" and round_index >= 2:
        out["deleted_objects"] += _delete_dead(ctx, (f"cc_labels/{round_index - 2:03d}",))
    if step == "a":
        out["converged"] = round_index > 0 and counts["changed"] == 0
    logger.info("components %s sealed: %s", name, {k: v for k, v in out.items() if k != "index"})
    return out


# ------------------------------------------------------------------------------- canonical


def _load_reps(ctx: Ctx, units: Iterable[int]) -> np.ndarray:
    pieces = [blocks.read_single_block_array(ctx.store, ctx.done("lshmap", u)["reps"], REPS_DTYPE) for u in units]
    return blocks.concat_arrays(pieces, REPS_DTYPE)


def run_canon_a(ctx: Ctx, partition: int) -> dict:
    plan = ctx.plan
    final = ctx.sealed("components")["final_round"]
    state = _load_labels(ctx, final, partition)
    nodes, root = state["node"], state["label"]
    members = np.empty(len(nodes), dtype=MEMBER_DTYPE)
    members["d"], members["root"] = nodes, root
    members_meta = blocks.write_single_block(ctx.store, blocks.map_object_path(ctx.data("members"), partition), members)

    reps = _load_reps(ctx, plan.node_partition_units(partition))
    at = np.searchsorted(reps["d"], nodes)
    if len(nodes) and (at.max() >= len(reps) or np.any(reps["d"][at] != nodes)):
        raise RuntimeError(f"partition {partition}: component members missing from the unique-doc lists")
    k1, k2, k3 = canonical_order_keys(reps["hi"][at], reps["lo"][at])
    # Combine on the map side: only this partition's best member per component (and the member count) crosses the
    # shuffle, so a giant component costs one record per partition, not one per member.
    order = np.lexsort((nodes, k3, k2, k1, root))
    first = _first_of_runs(root[order])
    starts = np.flatnonzero(first)
    best = order[starts]
    records = np.empty(len(best), dtype=CANON_DTYPE)
    records["root"], records["k1"], records["k2"], records["k3"] = root[best], k1[best], k2[best], k3[best]
    records["d"], records["n"] = nodes[best], np.diff(np.append(starts, len(order)))
    writer = blocks.BlockObjectWriter(plan.node_partitions)
    writer.add_array(records, plan.node_partition_of_keys(records["root"]))
    meta = writer.finish(ctx.store, blocks.map_object_path(ctx.data("canon_a"), partition))
    return {
        "blocks": meta,
        "members_meta": members_meta,
        "members": len(nodes),
        "member_chars": int(reps["chars"][at].sum()),
    }


def seal_canon_a(ctx: Ctx, results: list[dict]) -> dict:
    n = ctx.plan.node_partitions
    index = blocks.build_index(ctx.store, ctx.data("canon_a"), [r["blocks"] for r in results], n)
    deleted = _delete_dead(ctx, ("cc_edges", "cc_labels"))
    return {
        "members": _sum(results, "members"),
        "member_chars": _sum(results, "member_chars"),
        "index": index,
        "deleted_objects": deleted,
    }


def run_canon_b(ctx: Ctx, partition: int) -> dict:
    plan = ctx.plan
    n = plan.node_partitions
    pieces, sources = [], []
    for m, block in blocks.iter_partition_blocks(ctx.store, ctx.data("canon_a"), partition, n, n):
        for frame in blocks.iter_frames(block):
            pieces.append(np.frombuffer(frame, dtype=CANON_DTYPE))
            sources.append(np.full(len(pieces[-1]), m, dtype=np.int64))
    r = blocks.concat_arrays(pieces, CANON_DTYPE)
    # The node partition each record came from: it holds members of the record's component.
    source = np.concatenate(sources) if sources else np.zeros(0, dtype=np.int64)
    order = np.lexsort((r["d"], r["k3"], r["k2"], r["k1"], r["root"]))
    r, source = r[order], source[order]
    first = _first_of_runs(r["root"])
    starts = np.flatnonzero(first)
    sizes = np.add.reduceat(r["n"], starts) if len(r) else np.zeros(0, dtype=np.int64)
    canonical = np.sort(r["d"][starts])
    writer = blocks.BlockObjectWriter(n)
    writer.add_array(canonical, plan.node_partition_of_keys(canonical))
    meta = writer.finish(ctx.store, blocks.map_object_path(ctx.data("canon_b"), partition))
    size_records = np.empty(len(r), dtype=SIZE_DTYPE)
    size_records["root"] = r["root"]
    size_records["size"] = np.repeat(sizes, np.diff(np.append(starts, len(r))))
    writer = blocks.BlockObjectWriter(n)
    writer.add_array(size_records, source)
    sizes_meta = writer.finish(ctx.store, blocks.map_object_path(ctx.data("canon_sizes"), partition))
    big = sizes > MAX_CLUSTER_SIZE
    histogram = (
        np.bincount(np.floor(np.log2(sizes)).astype(np.int64), minlength=_SIZE_BINS)
        if len(sizes)
        else np.zeros(_SIZE_BINS)
    )
    return {
        "blocks": meta,
        "sizes_blocks": sizes_meta,
        "components": len(starts),
        "members": int(sizes.sum()),
        "oversized": {
            str(k): int(s) for k, s in zip(r["root"][starts][big].tolist(), sizes[big].tolist(), strict=True)
        },
        "largest": int(sizes.max()) if len(sizes) else 0,
        "size_log2_histogram": [int(c) for c in histogram],
    }


def seal_canon_b(ctx: Ctx, results: list[dict]) -> dict:
    n = ctx.plan.node_partitions
    index = blocks.build_index(ctx.store, ctx.data("canon_b"), [r["blocks"] for r in results], n)
    sizes_index = blocks.build_index(ctx.store, ctx.data("canon_sizes"), [r["sizes_blocks"] for r in results], n)
    oversized: dict[str, int] = {}
    for r in results:
        oversized.update(r["oversized"])
    histogram = np.sum([r["size_log2_histogram"] for r in results], axis=0)
    ctx.store.write(f"{ctx.data('oversized')}.json", dumps(oversized))
    largest = sorted(oversized.values(), reverse=True)[:25]
    return {
        "components": _sum(results, "components"),
        "members": _sum(results, "members"),
        "largest": max(r["largest"] for r in results),
        "largest_sizes": largest,
        "oversized": len(oversized),
        "oversized_members": int(sum(oversized.values())),
        "size_log2_histogram": [int(c) for c in histogram],
        "index": index,
        "sizes_index": sizes_index,
    }


def run_attrs(ctx: Ctx, partition: int) -> dict:
    plan = ctx.plan
    n = plan.node_partitions
    members_meta = ctx.done("canon_a", partition)["members_meta"]
    members = blocks.read_single_block_array(ctx.store, members_meta, MEMBER_DTYPE)
    canonical = blocks.read_partition_array(ctx.store, ctx.data("canon_b"), partition, n, n, KEY_DTYPE)
    attrs = np.empty(len(members), dtype=ATTR_DTYPE)
    attrs["d"], attrs["root"] = members["d"], members["root"]
    attrs["canon"] = np.isin(members["d"], canonical)
    if int(attrs["canon"].sum()) != len(canonical):
        raise RuntimeError(
            f"partition {partition}: {len(canonical)} canonicals but {int(attrs['canon'].sum())} matched"
        )
    sizes = np.sort(
        blocks.read_partition_array(ctx.store, ctx.data("canon_sizes"), partition, n, n, SIZE_DTYPE), order="root"
    )
    if not np.array_equal(sizes["root"], np.unique(members["root"])):
        raise RuntimeError(f"partition {partition}: component sizes do not cover each member component exactly once")
    attrs["size"] = sizes["size"][np.searchsorted(sizes["root"], members["root"])]
    units = plan.unit_of_keys(attrs["d"])
    written = {}
    for unit in plan.node_partition_units(partition):
        mine = np.sort(attrs[units == unit], order="d")
        written[str(unit)] = blocks.write_single_block(ctx.store, blocks.map_object_path(ctx.data("attrs"), unit), mine)
    return {
        "units": written,
        "members": len(attrs),
        "canonicals": len(canonical),
        "canonical_sizes": int(attrs["size"][attrs["canon"] == 1].sum(dtype=np.int64)),
    }


def seal_attrs(ctx: Ctx, results: list[dict]) -> dict:
    members = _sum(results, "members")
    if members != ctx.sealed("canon_b")["members"]:
        raise RuntimeError(f"attrs wrote {members} members, canon_b counted {ctx.sealed('canon_b')['members']}")
    # Each component has one canonical, so its recorded size is counted exactly once.
    canonical_sizes = _sum(results, "canonical_sizes")
    if canonical_sizes != members:
        raise RuntimeError(f"attrs: canonicals' component sizes sum to {canonical_sizes}, not {members} members")
    deleted = _delete_dead(ctx, ("canon_a", "canon_b", "canon_sizes", "members"))
    return {"members": members, "canonicals": _sum(results, "canonicals"), "deleted_objects": deleted}


def attrs_of_unit(ctx: Ctx, unit: int) -> np.ndarray:
    partition = unit // ctx.plan.units_per_node_partition
    meta = ctx.done("attrs", partition)["units"][str(unit)]
    return blocks.read_single_block_array(ctx.store, meta, ATTR_DTYPE)


# ------------------------------------------------------------------------------------ apply


def _apply_file(
    path: str, rows: np.ndarray, hi: np.ndarray, lo: np.ndarray, copies: np.ndarray, cluster_sizes: np.ndarray
) -> tuple[bytes, int, int]:
    """Gzip member holding the surviving rows of one kept file plus ``exact_copies`` and ``fuzzy_cluster_size``."""
    table = read_kept(store_for(path), path)
    records, texts = output_records(table, rows)
    for record, text, h, low, n, size in zip(
        records, texts, hi.tolist(), lo.tolist(), copies.tolist(), cluster_sizes.tolist(), strict=True
    ):
        if content_id(text) != (h << 64) | low:
            raise RuntimeError(f"{path}: a surviving row's text no longer hashes to its pass-1 id")
        record[EXACT_COPIES_FIELD] = n
        record[CLUSTER_SIZE_FIELD] = size
    payload = encode_jsonl(records)
    return gzip_member(payload), len(records), sum(len(t) for t in texts)


def output_path(plan: Plan, unit: int, chunk: int) -> str:
    return f"{plan.output_root}/{OUTPUT_DIR}/data-{unit:06d}-{chunk:03d}.jsonl.gz"


def _cluster_sizes(keys: np.ndarray, attrs: np.ndarray) -> np.ndarray:
    """Component size of each doc key of a unit; 1 for docs in no component."""
    sizes = np.ones(len(keys), dtype=np.uint64)
    at = np.searchsorted(attrs["d"], keys)
    member = at < len(attrs)
    member[member] = attrs["d"][at[member]] == keys[member]
    sizes[member] = attrs["size"][at[member]]
    return sizes


def _write_survivors(ctx: Ctx, unit: int, survivors: np.ndarray, attrs: np.ndarray, procs: int) -> dict:
    """Write ``survivors`` (a REPS_DTYPE slice sorted by doc key) of ``unit`` as JSONL outputs."""
    plan = ctx.plan
    cluster_sizes = _cluster_sizes(survivors["d"], attrs)
    paths = ctx.inputs()
    files = list(plan.unit_files(unit))
    file_of = key_file(survivors["d"]).astype(np.int64)
    bounds = np.searchsorted(file_of, np.asarray([*files, files[-1] + 1], dtype=np.int64))
    tasks, owners = [], []
    for i, f in enumerate(files):
        s = survivors[bounds[i] : bounds[i + 1]]
        if len(s):
            sizes = cluster_sizes[bounds[i] : bounds[i + 1]]
            tasks.append((paths[f], key_row(s["d"]).astype(np.int64), s["hi"], s["lo"], s["n"], sizes))
            owners.append(i // plan.files_per_output)
    written = []
    docs_out = chars_out = 0
    chunks: dict[int, list[bytes]] = {}
    for owner, (member, n_docs, n_chars) in zip(owners, _map_ordered(_apply_file, tasks, procs), strict=True):
        chunks.setdefault(owner, []).append(member)
        docs_out += n_docs
        chars_out += n_chars
    for chunk, members in sorted(chunks.items()):
        path = output_path(plan, unit, chunk)
        ctx.output_store.write(path, b"".join(members))
        written.append(path)
    if docs_out != len(survivors):
        raise RuntimeError(f"unit {unit}: wrote {docs_out} docs for {len(survivors)} survivors")
    return {"docs": docs_out, "chars": chars_out, "outputs": written}


def unit_reps(ctx: Ctx, unit: int) -> np.ndarray:
    return blocks.read_single_block_array(ctx.store, ctx.done("lshmap", unit)["reps"], REPS_DTYPE)


def seal_apply(ctx: Ctx, results: list[dict]) -> dict:
    return {
        "rule": "c075",
        "docs": _sum(results, "docs"),
        "chars": _sum(results, "chars"),
        "outputs": sum(len(r["outputs"]) for r in results),
    }


# ------------------------------------------------------------------------------------- c075


TEXT_SCHEMA = pa.schema(
    [
        pa.field("cluster_key", pa.string()),
        pa.field("id", pa.string()),
        pa.field("text", pa.large_string()),
        pa.field("d", pa.uint64()),
    ]
)


def _c075_text_file(
    path: str, rows: np.ndarray, keys: np.ndarray, roots: np.ndarray, splits: np.ndarray, n_partitions: int
) -> tuple[pa.Table, np.ndarray]:
    table = read_kept(store_for(path), path, ["text"])
    texts = table.column("text").take(pa.array(rows, type=pa.int64())).to_pylist()
    out_keys, out_ids, out_texts = [], [], []
    for text, root, n_splits in zip(texts, roots.tolist(), splits.tolist(), strict=True):
        text = compact_whitespace(text)
        doc_id = format(content_id(text), "032x")
        text = text[:TEXT_MAX_DOCUMENT_CHARS]
        out_keys.append(cluster_key(root, text, doc_id, n_splits))
        out_ids.append(doc_id)
        out_texts.append(text)
    partitions = np.fromiter(
        (dupekit.hash_xxh3_64(k.encode()) % n_partitions for k in out_keys), dtype=np.int64, count=len(out_keys)
    )
    out = pa.table(
        {"cluster_key": out_keys, "id": out_ids, "text": pa.array(out_texts, type=pa.large_string()), "d": keys},
        schema=TEXT_SCHEMA,
    )
    return out, partitions


def run_c075_text(ctx: Ctx, unit: int, procs: int) -> dict:
    plan = ctx.plan
    paths = ctx.inputs()
    attrs = attrs_of_unit(ctx, unit)
    oversized = read_json(ctx.store, f"{ctx.data('oversized')}.json")
    splits_by_root = {int(k): -(-size // MAX_CLUSTER_SIZE) for k, size in oversized.items()}
    splits = np.fromiter((splits_by_root.get(r, 1) for r in attrs["root"].tolist()), dtype=np.int64, count=len(attrs))
    file_of = key_file(attrs["d"]).astype(np.int64)
    tasks = []
    for f in plan.unit_files(unit):
        mine = file_of == f
        if mine.any():
            a = attrs[mine]
            tasks.append(
                (paths[f], key_row(a["d"]).astype(np.int64), a["d"], a["root"], splits[mine], plan.c075_partitions)
            )
    writer = blocks.BlockObjectWriter(plan.c075_partitions)
    pending: list[tuple[pa.Table, np.ndarray]] = []
    pending_chars = members = chars = 0

    def flush() -> None:
        nonlocal pending_chars
        if pending:
            writer.add_table(pa.concat_tables([t for t, _ in pending]), np.concatenate([p for _, p in pending]))
            pending.clear()
        pending_chars = 0

    for table, partitions in _map_ordered(_c075_text_file, tasks, procs):
        n_chars = int(pc.sum(pc.utf8_length(table.column("text"))).as_py() or 0)
        pending.append((table, partitions))
        pending_chars += n_chars
        members += table.num_rows
        chars += n_chars
        if pending_chars > C075_FLUSH_CHARS:
            flush()
    flush()
    meta = writer.finish(ctx.store, blocks.map_object_path(ctx.data("c075_text"), unit))
    if members != len(attrs):
        raise RuntimeError(f"unit {unit}: emitted {members} member texts for {len(attrs)} members")
    return {"blocks": meta, "members": members, "chars": chars}


def seal_c075_text(ctx: Ctx, results: list[dict]) -> dict:
    index = blocks.build_index(
        ctx.store, ctx.data("c075_text"), [r["blocks"] for r in results], ctx.plan.c075_partitions
    )
    return {"members": _sum(results, "members"), "chars": _sum(results, "chars"), "index": index}


def _pass_of(cluster_keys: list[str], n_passes: int) -> np.ndarray:
    """Pass of each cluster key; salted so it is independent of the partition hash."""
    return np.fromiter(
        (dupekit.hash_xxh3_64(b"pass|" + k.encode()) % n_passes for k in cluster_keys),
        dtype=np.int64,
        count=len(cluster_keys),
    )


def _pass_table(ctx: Ctx, partition: int, pass_index: int, n_passes: int) -> pa.Table:
    """This partition's rows whose cluster key falls in ``pass_index``, streamed block by block."""
    plan = ctx.plan
    kept = []
    stream = blocks.iter_partition_blocks(
        ctx.store, ctx.data("c075_text"), partition, plan.n_units, plan.c075_partitions
    )
    for _m, block in stream:
        for frame in blocks.iter_frames(block):
            table = blocks.ipc_to_table(frame)
            if n_passes > 1:
                mask = _pass_of(table.column("cluster_key").to_pylist(), n_passes) == pass_index
                table = table.filter(pa.array(mask))
            if table.num_rows:
                kept.append(table)
    return pa.concat_tables(kept) if kept else TEXT_SCHEMA.empty_table()


def solve_c075_partition(ctx: Ctx, partition: int) -> tuple[np.ndarray, SolveStats, int]:
    """The sorted doc keys the containment rule removes in one c075 partition, with solve stats and pass count.

    The partition is solved in as many passes as its size needs. A pass holds only the groups whose key hashes to
    it. Groups are independent and each is solved whole, in ``(cluster_key, id)`` order, so the removals do not
    depend on the pass count.
    """
    plan = ctx.plan
    compressed = blocks.partition_bytes(ctx.store, ctx.data("c075_text"), partition, plan.n_units, plan.c075_partitions)
    n_passes = max(1, -(-compressed // SOLVE_PASS_COMPRESSED_BYTES))
    stats = SolveStats()
    removed_parts = []
    for pass_index in range(n_passes):
        table = _pass_table(ctx, partition, pass_index, n_passes)
        removed_parts.append(np.fromiter(solve_sorted_rows(sorted_rows(table), stats), dtype=np.uint64))
        del table
    return np.sort(blocks.concat_arrays(removed_parts, KEY_DTYPE)), stats, n_passes


def run_c075_solve(ctx: Ctx, partition: int) -> dict:
    """Solve one partition's groups and shuffle the removed doc keys to their node partitions."""
    plan = ctx.plan
    started = time.monotonic()
    removed, stats, n_passes = solve_c075_partition(ctx, partition)
    writer = blocks.BlockObjectWriter(plan.node_partitions)
    writer.add_array(removed, plan.node_partition_of_keys(removed))
    meta = writer.finish(ctx.store, blocks.map_object_path(ctx.data("c075_solve"), partition))
    return {
        "blocks": meta,
        **dataclasses.asdict(stats),
        "passes": n_passes,
        "solve_seconds": time.monotonic() - started,
    }


def seal_c075_solve(ctx: Ctx, results: list[dict]) -> dict:
    index = blocks.build_index(
        ctx.store, ctx.data("c075_solve"), [r["blocks"] for r in results], ctx.plan.node_partitions
    )
    fields = [f.name for f in dataclasses.fields(SolveStats)]
    totals = {k: _sum(results, k) for k in fields}
    if totals["documents"] != ctx.sealed("c075_text")["members"]:
        raise RuntimeError(
            f"solve saw {totals['documents']} documents, text stage emitted {ctx.sealed('c075_text')['members']}"
        )
    deleted = _delete_dead(ctx, ("c075_text",))
    return {
        **totals,
        "solve_seconds": sum(r["solve_seconds"] for r in results),
        "index": index,
        "deleted_objects": deleted,
    }


def run_c075_marks(ctx: Ctx, partition: int) -> dict:
    plan = ctx.plan
    removed = blocks.read_partition_array(
        ctx.store, ctx.data("c075_solve"), partition, plan.c075_partitions, plan.node_partitions, KEY_DTYPE
    )
    removed = np.sort(removed)
    units = plan.unit_of_keys(removed)
    written = {}
    for unit in plan.node_partition_units(partition):
        written[str(unit)] = blocks.write_single_block(
            ctx.store, blocks.map_object_path(ctx.data("c075_marks"), unit), removed[units == unit]
        )
    return {"units": written, "removed": len(removed)}


def seal_c075_marks(ctx: Ctx, results: list[dict]) -> dict:
    removed = _sum(results, "removed")
    if removed != ctx.sealed("c075_solve")["duplicates"]:
        raise RuntimeError(f"marks hold {removed} removals, solve found {ctx.sealed('c075_solve')['duplicates']}")
    deleted = _delete_dead(ctx, ("c075_solve",))
    return {"removed": removed, "deleted_objects": deleted}


def c075_marks_of_unit(ctx: Ctx, unit: int) -> np.ndarray:
    partition = unit // ctx.plan.units_per_node_partition
    meta = ctx.done("c075_marks", partition)["units"][str(unit)]
    return blocks.read_single_block_array(ctx.store, meta, KEY_DTYPE)


def run_apply_c075(ctx: Ctx, unit: int, procs: int) -> dict:
    reps = unit_reps(ctx, unit)
    removed = c075_marks_of_unit(ctx, unit)
    survivors = reps[~np.isin(reps["d"], removed)]
    if len(reps) - len(survivors) != len(removed):
        raise RuntimeError(f"unit {unit}: {len(removed)} removals but {len(reps) - len(survivors)} matched")
    return _write_survivors(ctx, unit, survivors, attrs_of_unit(ctx, unit), procs)


# ------------------------------------------------------------------------------------ report


def stats_path(plan: Plan) -> str:
    return f"{plan.output_root}/{STATS_PATH}"


def run_report(ctx: Ctx, _task: int) -> dict:
    names = ["pass1", "exact", "lshmap", "bucket", "components", "canon_a", "canon_b", "attrs"]
    names += ["c075_text", "c075_solve", "c075_marks", "apply_c075"]
    sealed = {name: {k: v for k, v in ctx.sealed(name).items() if k != "index"} for name in names}
    pass1, exact, canon_b = sealed["pass1"], sealed["exact"], sealed["canon_b"]
    report = {
        "run_root": ctx.plan.run_root,
        "output_root": ctx.plan.output_root,
        "n_warcs": ctx.plan.n_files,
        "fuzzy_params": {
            "num_perms": NUM_PERMS,
            "num_bands": NUM_BANDS,
            "ngram_size_chars": NGRAM_SIZE,
            "seed": SEED,
            "approx_jaccard_threshold": APPROX_JACCARD_THRESHOLD,
            "text_cap_chars": None,
        },
        "records_in": pass1["kept"],
        "empty_text_filtered": pass1["empty_text_filtered"],
        "exact_duplicates": exact["exact_dups"],
        "unique_docs": exact["unique"],
        "cluster_members": canon_b["members"],
        "components": canon_b["components"],
        "stages": sealed,
        "c075": {
            "docs": sealed["apply_c075"]["docs"],
            "chars": sealed["apply_c075"]["chars"],
            "removed": sealed["c075_marks"]["removed"],
            "rule": dataclasses.asdict(C075_PARAMS),
            "max_cluster_size": MAX_CLUSTER_SIZE,
        },
    }
    ctx.output_store.write(stats_path(ctx.plan), dumps(report))
    return {"report": stats_path(ctx.plan)}


def seal_report(ctx: Ctx, results: list[dict]) -> dict:
    return results[0]


# --------------------------------------------------------------------------------- pipeline


def _stage(name, n, run, seal, procs=1, mem_gb=4.0, deadline_hours=3.0) -> StageSpec:
    return StageSpec(name, n, run, seal, procs, mem_gb, deadline_hours)


def stage_sequence(plan: Plan, sealed: Callable[[str], dict], procs: int) -> Iterator[StageSpec]:
    """The stages in order. Component rounds continue until a round's seal reports convergence; ``sealed`` is only
    consulted for stages the caller has already found sealed."""
    u, rn = plan.n_units, plan.node_partitions
    yield _stage("pass1", u, lambda c, t: run_pass1(c, t, procs), seal_pass1, procs=procs, mem_gb=8, deadline_hours=4)
    yield _stage("exact", plan.exact_partitions, run_exact, seal_exact, mem_gb=6)
    yield _stage("lshmap", u, run_lshmap, seal_lshmap, mem_gb=10)
    yield _stage("bucket", plan.lsh_partitions, run_bucket, seal_bucket, mem_gb=12)
    round_index = 0
    while True:
        for step, run in (("a", run_cc_a), ("b", run_cc_b), ("c", run_cc_c)):
            yield _stage(
                cc_name(round_index, step),
                rn,
                lambda c, t, k=round_index, f=run: f(c, t, k),
                lambda c, rs, k=round_index, st=step: seal_cc(c, rs, k, st),
                mem_gb=10,
            )
            if step == "a" and sealed(cc_name(round_index, "a"))["converged"]:
                break
        else:
            round_index += 1
            continue
        break
    yield _stage("canon_a", rn, run_canon_a, seal_canon_a, mem_gb=6)
    yield _stage("canon_b", rn, run_canon_b, seal_canon_b, mem_gb=6)
    yield _stage("attrs", rn, run_attrs, seal_attrs, mem_gb=4)
    yield _stage("c075_text", u, lambda c, t: run_c075_text(c, t, procs), seal_c075_text, procs=procs, mem_gb=14)
    yield _stage("c075_solve", plan.c075_partitions, run_c075_solve, seal_c075_solve, mem_gb=15)
    yield _stage("c075_marks", rn, run_c075_marks, seal_c075_marks, mem_gb=3)
    yield _stage("apply_c075", u, lambda c, t: run_apply_c075(c, t, procs), seal_apply, procs=procs, mem_gb=8)
    yield _stage("report", 1, run_report, seal_report, mem_gb=2)


def component_rounds_result(ctx_sealed: Callable[[str], dict]) -> dict:
    """The ``components`` pseudo-stage: the converged round and per-round label changes."""
    rounds = []
    k = 0
    while True:
        result = ctx_sealed(cc_name(k, "a"))
        rounds.append({key: v for key, v in result.items() if key != "index" and not key.startswith("_")})
        if result["converged"]:
            return {"final_round": k, "rounds": rounds}
        k += 1


class Pipeline:
    """Finds the frontier stage of a run from sealed markers, caching what it has seen."""

    def __init__(self, store: Store, plan: Plan, procs: int, keep_intermediates: bool = False) -> None:
        self.store = store
        self.plan = plan
        self.procs = procs
        self._sealed: dict[str, dict] = {}
        self.ctx = Ctx(store, plan, self.sealed, store_for(plan.output_root), keep_intermediates)

    def sealed_path(self, name: str) -> str:
        return f"{self.plan.run_root}/stages/{name}/sealed.json"

    def sealed(self, name: str) -> dict:
        if name == "components":
            return component_rounds_result(self.sealed)
        if name not in self._sealed:
            self._sealed[name] = read_json(self.store, self.sealed_path(name))
        return self._sealed[name]

    def frontier(self) -> StageSpec | None:
        """The first stage without a seal, or ``None`` once every stage is sealed."""
        for spec in stage_sequence(self.plan, self.sealed, self.procs):
            if spec.name in self._sealed:
                continue
            if self.store.stat(self.sealed_path(spec.name)) is None:
                return spec
            self.sealed(spec.name)
        return None

    def stages_so_far(self) -> list[StageSpec]:
        """Every stage up to and including the frontier."""
        out = []
        for spec in stage_sequence(self.plan, self.sealed, self.procs):
            out.append(spec)
            if spec.name not in self._sealed and self.store.stat(self.sealed_path(spec.name)) is None:
                break
            self.sealed(spec.name)
        return out
