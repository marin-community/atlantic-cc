# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Creating a dedup run (``init``) and reporting its progress (``status``); as-run ``rdedup/launcher.py``.

``init`` freezes the inputs and the plan under a scratch run root: the kept files are listed (or read from a
``(path, size)`` listing), sorted by full path, optionally narrowed to a WARC-hash manifest, and packed into units.
The sorted path list decides every document key, hence which copy of an exact duplicate survives, and the byte
sizes decide the output file names, so a reproduction must start from the same list.
"""

import dataclasses
import io
import json
import logging
import os

import pyarrow.parquet as pq

from atlantic_cc.dedup.params import FILES_PER_OUTPUT, OUTPUT_DIR, UNIT_BYTES
from atlantic_cc.dedup.plan import Plan, build_plan, inputs_sha256, load_plan, write_plan
from atlantic_cc.dedup.stages import Pipeline
from atlantic_cc.dedup.tasks import ClaimConfig, TaskBoard
from atlantic_cc.io.resolve import store_for
from atlantic_cc.io.store import ObjectInfo, Store, dumps, read_json
from atlantic_cc.spec import PRODUCTION_SPEC_ID

logger = logging.getLogger(__name__)

RUN_MARKER = ".rdedup_run.json"
KEPT_SUFFIX = ".parquet"


@dataclasses.dataclass(frozen=True)
class PartitionOverrides:
    """Partition counts to use instead of the plan formulas. They change scheduling only, never outputs."""

    exact: int | None = None
    lsh: int | None = None
    c075: int | None = None


AS_PLANNED = PartitionOverrides()


def warc_hash_of(path: str) -> str:
    """``data-{warc_hash}.parquet`` -> ``warc_hash``."""
    return os.path.basename(path).removeprefix("data-").removesuffix(KEPT_SUFFIX)


def read_warc_hashes(path: str) -> set[str]:
    """WARC hashes from a manifest: a parquet with a ``warc_hash`` column, or one hash per line (``#`` comments)."""
    data = store_for(path).read(path)
    if path.endswith(".parquet"):
        return set(pq.read_table(io.BytesIO(data), columns=["warc_hash"]).column("warc_hash").to_pylist())
    return {line.strip() for line in data.decode().splitlines() if line.strip() and not line.startswith("#")}


def read_listing(path: str) -> list[ObjectInfo]:
    """``(path, size)`` rows of a parquet listing, e.g. a previous run's ``inputs.parquet``."""
    table = pq.read_table(io.BytesIO(store_for(path).read(path)), columns=["path", "size"])
    return [
        ObjectInfo(p, 0, 0.0, size)
        for p, size in zip(table.column("path").to_pylist(), table.column("size").to_pylist(), strict=True)
    ]


def list_kept(kept: str) -> list[ObjectInfo]:
    return [info for info in store_for(kept).list(kept) if info.path.endswith(KEPT_SUFFIX)]


def init_run(
    *,
    run_root: str,
    output_root: str,
    kept: str | None = None,
    listing: str | None = None,
    warc_hashes: str | None = None,
    limit_files: int | None = None,
    expect_files: int | None = None,
    unit_bytes: int = UNIT_BYTES,
    node_partitions: int | None = None,
    files_per_output: int = FILES_PER_OUTPUT,
    partitions: PartitionOverrides = AS_PLANNED,
    spec_id: str = PRODUCTION_SPEC_ID,
) -> Plan:
    """Freeze the inputs and plan of a run under ``run_root``; re-running with identical settings is a no-op.

    Args:
        run_root: Scratch root of the run (a local directory, ``gs://`` or another fsspec URL).
        output_root: Where ``deduped_c075/`` and ``stats/`` are written.
        kept: The ``kept/`` prefix to list; with ``listing``, only listed paths under it are used.
        listing: A parquet of ``(path, size)`` to use instead of listing ``kept``.
        warc_hashes: Keep only the kept files of these WARC hashes (parquet ``warc_hash`` column or text lines).
        limit_files: Keep only the first N files after sorting (smoke runs).
        expect_files: Fail unless exactly this many files remain.
        unit_bytes: Compressed kept bytes per unit; the release used 8e9.
        node_partitions: Connected-component partitions (default: units // 4).
        files_per_output: Kept files per output file; the release used 32.
        partitions: Overrides for the exact, LSH and c075 partition counts.
        spec_id: Free-form label stored in the plan.
    """
    if kept is None and listing is None:
        raise ValueError("give the kept/ prefix to list, a listing parquet, or both")
    if listing is not None:
        files = [info for info in read_listing(listing) if kept is None or info.path.startswith(kept)]
    else:
        files = list_kept(kept)
    if not files:
        raise FileNotFoundError(f"no kept parquet under {kept or listing}")
    files.sort(key=lambda info: info.path)
    if warc_hashes is not None:
        wanted = read_warc_hashes(warc_hashes)
        files = [info for info in files if warc_hash_of(info.path) in wanted]
        missing = wanted - {warc_hash_of(info.path) for info in files}
        if missing:
            raise FileNotFoundError(f"{len(missing)} manifest WARCs have no kept file, e.g. {sorted(missing)[:5]}")
    if limit_files is not None:
        files = files[:limit_files]
    if expect_files is not None and len(files) != expect_files:
        raise ValueError(f"found {len(files)} kept files, expected {expect_files}")

    output_store = store_for(output_root)
    marker = f"{output_root.rstrip('/')}/{RUN_MARKER}"
    if output_store.stat(marker) is not None:
        owner = read_json(output_store, marker)["run_root"]
        if owner != run_root.rstrip("/"):
            raise ValueError(f"{output_root} belongs to run {owner}; pick a new output root")
    else:
        for folder in (f"{OUTPUT_DIR}/", "stats/"):
            if output_store.list(f"{output_root.rstrip('/')}/{folder}"):
                raise ValueError(f"{output_root}/{folder} already holds objects from another run")

    paths = [info.path for info in files]
    sizes = [info.size for info in files]
    plan = build_plan(
        run_root=run_root,
        output_root=output_root,
        spec_id=spec_id,
        sizes=sizes,
        unit_bytes=unit_bytes,
        node_partitions=node_partitions,
        files_per_output=files_per_output,
    )
    plan = dataclasses.replace(
        plan,
        exact_partitions=partitions.exact or plan.exact_partitions,
        lsh_partitions=partitions.lsh or plan.lsh_partitions,
        c075_partitions=partitions.c075 or plan.c075_partitions,
    )
    store = store_for(run_root)
    write_plan(store, plan, paths, sizes)
    if output_store.stat(marker) is None:
        output_store.write(marker, dumps({"run_root": plan.run_root, "inputs_sha256": inputs_sha256(paths)}))
    summary = {
        "files": plan.n_files,
        "bytes": sum(sizes),
        "units": plan.n_units,
        "exact_partitions": plan.exact_partitions,
        "lsh_partitions": plan.lsh_partitions,
        "node_partitions": plan.node_partitions,
        "c075_partitions": plan.c075_partitions,
        "inputs_sha256": inputs_sha256(paths),
    }
    logger.info("plan written to %s: %s", plan.run_root, json.dumps(summary))
    return plan


def stage_progress(store: Store, run_root: str) -> list[dict]:
    """Progress of every stage up to the frontier: done / in flight / total, failures, and seal state."""
    plan = load_plan(store, run_root)
    pipeline = Pipeline(store, plan, procs=1)
    rows = []
    for spec in pipeline.stages_so_far():
        board = TaskBoard(store, pipeline.ctx.stage_dir(spec.name), spec.n_tasks, ClaimConfig())
        sealed = board.is_sealed()
        row = {"stage": spec.name, "tasks": spec.n_tasks, "sealed": sealed}
        if not sealed:
            state = board.state()
            in_flight = sum(
                1
                for task, (_generation, updated) in state.claims.items()
                if task not in state.done and state.now - updated <= board.config.stale_seconds
            )
            row.update(
                done=len(state.done), in_flight=in_flight, failures=len(store.list(f"{board.stage_dir}/failures/"))
            )
        rows.append(row)
    return rows
