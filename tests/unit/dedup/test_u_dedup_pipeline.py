# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""U-DEDUP rules 3, 6, 7 and 8 end to end: the standalone pipeline against the as-run pipeline's goldens, exact
duplicate provenance, copy counts and cluster sizes, and invariance to partitioning, passes, units and retries."""

import collections
import dataclasses
import glob
import gzip
import hashlib
import json
import os
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from atlantic_cc.dedup import blocks, solver, stages, worker
from atlantic_cc.dedup.local_runner import run_local
from atlantic_cc.dedup.plan import build_plan, load_plan, write_plan
from atlantic_cc.dedup.tasks import ClaimConfig
from atlantic_cc.dedup.verify import normalized_stats
from atlantic_cc.io.local import LocalStore
from atlantic_cc.schemas import KEPT_SCHEMA
from tests.unit.dedup import synthetic

GOLDEN = Path(__file__).parents[2] / "golden" / "dedup"
ASRUN = json.loads((GOLDEN / "asrun_goldens.json").read_text())
SMALL_CAPS_TARGETS = {
    "MAX_CLUSTER_SIZE": stages,
    "TEXT_MAX_DOCUMENT_CHARS": stages,
    "SOLVE_MAX_DOCUMENT_CHARS": solver,
    "SOLVE_MAX_CLUSTER_CHARS": solver,
}


def _plan(root: Path, kept: Path, **overrides):
    paths = sorted(str(p) for p in kept.glob("*.parquet"))
    sizes = [os.path.getsize(p) for p in paths]
    plan = build_plan(
        run_root=str(root / "run"), output_root=str(root / "out"), spec_id="test", sizes=sizes, unit_bytes=10**12
    )
    plan = dataclasses.replace(plan, **overrides)
    write_plan(LocalStore(), plan, paths, sizes)
    return plan


def _outputs(plan) -> dict[str, bytes]:
    return {os.path.basename(p): Path(p).read_bytes() for p in sorted(glob.glob(f"{plan.output_root}/deduped_c075/*"))}


def _records(plan) -> list[dict]:
    return [json.loads(line) for payload in _outputs(plan).values() for line in gzip.decompress(payload).splitlines()]


def _digests(plan) -> dict:
    store = LocalStore()
    pipeline = stages.Pipeline(store, load_plan(store, plan.run_root), procs=1)
    units = {}
    for unit in range(plan.n_units):
        arrays = {
            "reps": stages.unit_reps(pipeline.ctx, unit),
            "attrs": stages.attrs_of_unit(pipeline.ctx, unit),
            "c075_marks": stages.c075_marks_of_unit(pipeline.ctx, unit),
        }
        units[str(unit)] = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()}
    outputs = {}
    for name, payload in _outputs(plan).items():
        data = gzip.decompress(payload)
        outputs[name] = {"sha256": hashlib.sha256(data).hexdigest(), "lines": data.count(b"\n")}
    stats = json.loads(Path(plan.output_root, stages.STATS_PATH).read_text())
    return {"outputs": outputs, "units": units, "stats": normalized_stats(stats, same_partitions=True)}


@pytest.mark.parametrize("variant", ["default", "small_caps"])
def test_pipeline_matches_the_as_run_pipeline(tmp_path, monkeypatch, variant):
    """Same synthetic corpus and plan as the as-run run behind the golden: every output byte, every unit's
    reps / attrs / removal list and the stats agree. ``small_caps`` shrinks the limits so the oversized split, both
    text cuts, longest-first batches and mid-cluster flushes all happen."""
    if variant == "small_caps":
        for name, value in ASRUN["small_caps"].items():
            monkeypatch.setattr(SMALL_CAPS_TARGETS[name], name, value)
    kept = tmp_path / "kept"
    synthetic.write_corpus(kept)
    plan = _plan(tmp_path / variant, kept, **ASRUN["pipeline_plan"])
    run_local(plan.run_root, cpus=1, memory_gb=64)
    assert _digests(plan) == ASRUN["pipeline"][variant]


def _write_kept(directory: Path, files: list[list[str | None]]) -> None:
    directory.mkdir(parents=True)
    for f, texts in enumerate(files):
        rows = [
            {
                "doc_id": f"{f}-{i}",
                "url": f"https://example.com/{f}/{i}",
                "warc_hash": f"{f:012x}",
                "snapshot": "CC-MAIN-2020-10",
                "fasttext_score": 0.5,
                "text": text,
                "n_tokens": len(text.split()) if text else 0,
                "pooled_prob": 0.25,
                "modernbert_prob": 0.75,
            }
            for i, text in enumerate(texts)
        ]
        pq.write_table(pa.Table.from_pylist(rows, schema=KEPT_SCHEMA), directory / f"data-{f:012x}.parquet")


def _random_words(rng: random.Random, n: int) -> list[str]:
    """Words of random letters: unrelated texts share almost no character 5-grams."""
    return ["".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(6)) for _ in range(n)]


def test_exact_duplicates_keep_the_copy_with_the_smallest_doc_key(tmp_path):
    rng = random.Random(3)
    fillers = [" ".join(_random_words(rng, 40)) for _ in range(12)]
    same = " ".join(_random_words(rng, 50))
    _write_kept(tmp_path / "kept", [[*fillers[0:2], same, *fillers[2:4], same], [same, *fillers[4:8]], fillers[8:12]])
    plan = _plan(tmp_path, tmp_path / "kept", unit_starts=[0, 1, 2, 3])
    run_local(plan.run_root, cpus=1, memory_gb=64)
    survivors = [r for r in _records(plan) if r["text"] == same]
    # Copies at (file 0, row 5), (file 1, row 0) and (file 0, row 2): the earliest file, then the lowest row wins.
    assert [(r["doc_id"], r["exact_copies"]) for r in survivors] == [("0-2", 3)]
    report = json.loads(Path(plan.output_root, stages.STATS_PATH).read_text())
    assert report["stages"]["pass1"]["unit_dups"] == 1 and report["stages"]["exact"]["cross_unit_dups"] == 1


def _edit(rng: random.Random, words: list[str], fraction: float) -> list[str]:
    out = list(words)
    for _ in range(max(1, int(len(out) * fraction))):
        out[rng.randrange(len(out))] = rng.choice(out)
    return out


def test_output_docs_carry_exact_copy_counts_and_fuzzy_cluster_sizes(tmp_path):
    rng = random.Random(7)
    fillers = iter(" ".join(_random_words(rng, 80)) for _ in range(60))
    a, b = (" ".join(_random_words(rng, 120)) for _ in range(2))
    base = _random_words(rng, 300)
    family = [" ".join(_edit(rng, base, 0.005)) for _ in range(4)]
    spaced = _random_words(rng, 100)
    # Whitespace runs over 128 characters are cut to 128, so these two normalize to one text.
    long_gap = " ".join(spaced[:50]) + " " * 300 + " ".join(spaced[50:])
    gap_128 = " ".join(spaced[:50]) + " " * 128 + " ".join(spaced[50:])
    files = [
        [a, a, family[0], long_gap] + [next(fillers) for _ in range(8)],
        [a, b] + [next(fillers) for _ in range(8)],
        [family[1], gap_128] + [next(fillers) for _ in range(8)],
        [family[2], b] + [next(fillers) for _ in range(8)],
        [family[2], "", None] + [next(fillers) for _ in range(8)],
        [a, family[3]] + [next(fillers) for _ in range(8)],
    ]
    _write_kept(tmp_path / "kept", files)
    plan = _plan(tmp_path, tmp_path / "kept", unit_starts=[0, 2, 4, 6], files_per_output=1)
    run_local(plan.run_root, cpus=1, memory_gb=64)
    report = json.loads(Path(plan.output_root, stages.STATS_PATH).read_text())
    assert report["stages"]["pass1"]["unit_dups"] > 0 and report["stages"]["exact"]["cross_unit_dups"] > 0
    expected_copies = {a: 4, b: 2, family[2]: 2, gap_128: 2}
    by_text = {}
    for record in _records(plan):
        assert record["text"] not in by_text
        by_text[record["text"]] = record
    assert [t for t in by_text if t in family]
    for text, record in by_text.items():
        assert record["exact_copies"] == expected_copies.get(text, 1)
        assert record["fuzzy_cluster_size"] == (4 if text in family else 1)


def test_partition_and_pass_counts_change_no_output_byte(tmp_path, monkeypatch):
    kept = tmp_path / "kept"
    synthetic.write_corpus(kept, seed=5)
    unit_starts = [0, 2, 4, 7, 9]
    few = _plan(tmp_path / "few", kept, unit_starts=unit_starts, exact_partitions=1, lsh_partitions=1)
    run_local(few.run_root, cpus=1, memory_gb=64)
    monkeypatch.setattr(stages, "SOLVE_PASS_COMPRESSED_BYTES", 2048)  # 2 to 8 solve passes per partition
    many = _plan(
        tmp_path / "many",
        kept,
        unit_starts=unit_starts,
        exact_partitions=4,
        lsh_partitions=7,
        units_per_node_partition=1,
        c075_partitions=5,
    )
    run_local(many.run_root, cpus=1, memory_gb=64)
    report = json.loads(Path(many.output_root, stages.STATS_PATH).read_text())
    assert report["c075"]["removed"] > 0
    assert min(r["passes"] for r in done_marks(many, "c075_solve")) > 1
    assert _outputs(many) == _outputs(few)


def test_unit_size_changes_only_the_file_grouping(tmp_path):
    kept = tmp_path / "kept"
    synthetic.write_corpus(kept, seed=6)
    one = _plan(tmp_path / "one", kept, unit_starts=[0, 9])
    three = _plan(tmp_path / "three", kept, unit_starts=[0, 3, 6, 9], files_per_output=2)
    for plan in (one, three):
        run_local(plan.run_root, cpus=1, memory_gb=64)
    assert sorted(_outputs(one)) != sorted(_outputs(three))
    canonical = collections.Counter(json.dumps(r, sort_keys=True) for r in _records(one))
    assert canonical == collections.Counter(json.dumps(r, sort_keys=True) for r in _records(three))


def test_failed_tasks_are_retried_by_the_claims_runner_with_identical_outputs(tmp_path, monkeypatch):
    kept = tmp_path / "kept"
    synthetic.write_corpus(kept, seed=3, n_files=6)
    clean = _plan(tmp_path / "clean", kept, unit_starts=[0, 2, 4, 6], exact_partitions=2, lsh_partitions=3)
    run_local(clean.run_root, cpus=1, memory_gb=64)

    failed: set[tuple] = set()
    names = ("run_pass1", "run_bucket", "run_cc_a", "run_cc_c", "run_c075_solve")
    originals = {name: getattr(stages, name) for name in names}

    def flaky(name):
        def wrapper(ctx, task, *args):
            key = (name, task, *args)
            if key not in failed:
                failed.add(key)
                raise RuntimeError(f"injected failure in {name}")
            return originals[name](ctx, task, *args)

        return wrapper

    for name in originals:
        monkeypatch.setattr(stages, name, flaky(name))
    monkeypatch.setattr(worker, "IDLE_POLL_SECONDS", 0.02)
    monkeypatch.setattr(worker, "FAILURE_BACKOFF_SECONDS", (0.0, 0.0))
    flaky_plan = _plan(tmp_path / "flaky", kept, unit_starts=[0, 2, 4, 6], exact_partitions=2, lsh_partitions=3)
    # A failed task's claim must go stale before it is retried; keep that wait short.
    claims = ClaimConfig(heartbeat_seconds=0.02, stale_seconds=0.1)
    assert worker.slot_loop(0, flaky_plan.run_root, cpus=1, memory_gb=64, claims=claims) == 0
    assert len(failed) >= 10
    assert _outputs(flaky_plan) == _outputs(clean)
    assert all(r["_attempt"] == 2 for r in done_marks(flaky_plan, "pass1"))


def done_marks(plan, stage: str) -> list[dict]:
    store = LocalStore()
    return [json.loads(store.read(p)) for p in sorted(glob.glob(f"{plan.run_root}/stages/{stage}/done/*.json"))]


def test_local_runner_resumes_from_its_done_marks(tmp_path, monkeypatch):
    kept = tmp_path / "kept"
    synthetic.write_corpus(kept, seed=4, n_files=6)
    reference = _plan(tmp_path / "reference", kept, unit_starts=[0, 3, 6])
    run_local(reference.run_root, cpus=1, memory_gb=64)

    original = stages.run_apply_c075

    def crash_on_unit_1(ctx, unit, procs):
        if unit == 1:
            raise RuntimeError("host lost")
        return original(ctx, unit, procs)

    monkeypatch.setattr(stages, "run_apply_c075", crash_on_unit_1)
    plan = _plan(tmp_path / "resumed", kept, unit_starts=[0, 3, 6])
    with pytest.raises(RuntimeError, match="host lost"):
        run_local(plan.run_root, cpus=1, memory_gb=64)
    finished_before = [r["_finished"] for r in done_marks(plan, "pass1")]
    monkeypatch.setattr(stages, "run_apply_c075", original)
    run_local(plan.run_root, cpus=1, memory_gb=64)
    assert [r["_finished"] for r in done_marks(plan, "pass1")] == finished_before
    assert _outputs(plan) == _outputs(reference)


def test_shuffle_blocks_round_trip_by_partition(tmp_path):
    store = LocalStore()
    writer = blocks.BlockObjectWriter(3)
    records = pa.array(range(10), type=pa.uint64()).to_numpy()
    writer.add_array(records, records.astype("int64") % 3)
    meta = writer.finish(store, str(tmp_path / "map" / "000000.bin"))
    index = blocks.build_index(store, str(tmp_path / "map"), [meta], 3)
    assert index["n_maps"] == 1
    got = [blocks.read_partition_array(store, str(tmp_path / "map"), p, 1, 3, records.dtype).tolist() for p in range(3)]
    assert got == [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]]
