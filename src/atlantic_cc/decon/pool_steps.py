# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pool-scale steps of the as-run decontamination, ported from Marin ``decon_8m.py`` (tag ``lpv11-8m-as-run``).

These rebuild the inputs that the shipped artifacts already contain, so replaying the release does not
need them: the drop-set sample, the mark (per-doc attributes; its matched hashes give S'), candidate
counts (S' with document frequencies), pool counts (pooldf1000), code counts (codedf100), the two-pass
filter and the manifest that names matched evals. Algorithms, seeds, partitioning and output schemas are
the as-run ones; Zephyr orchestration is replaced by a local fork-based process pool (one task per file
or batch, outputs written per task, so a rerun can skip finished outputs). Running them over the full
pool needs Marin-scale compute (the mark scored 9.95B documents).
"""

import concurrent.futures
import gzip
import json
import logging
import multiprocessing
import os
import random
from collections import Counter
from collections.abc import Callable, Container, Iterable, Iterator, Sequence

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from atlantic_cc.decon import dropsets as ds
from atlantic_cc.decon.filter import removal_matches
from atlantic_cc.decon.matching import _bloom_hash, _document_overlap_and_matches, _extract_features
from atlantic_cc.decon.names import eval_name
from atlantic_cc.decon.params import (
    CODE_FILES_PER_LANGUAGE,
    CODE_LANGUAGES,
    COUNT_FILES_PER_TASK,
    COUNT_PARTITIONS,
    MARK_NGRAM,
    POOL_COUNT_FILES_PER_TASK,
    SAMPLE_FILES,
    SAMPLE_PER_FILE,
    SAMPLE_SEED,
    TRACK_MIN_CANDIDATE_DF,
)

logger = logging.getLogger(__name__)

ID_FIELD = "doc_id"
INPUT_SUFFIX = ".jsonl.gz"
FLAGGED_SAMPLE_PER_FILE = 8
INDEX_BATCH_ROWS = 1_000_000
READ_THREADS = 32

ATTR_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("contaminated", pa.bool_()),
        ("max_overlap", pa.float64()),
        ("matched_hashes", pa.list_(pa.uint64())),
    ]
)
FLAGGED_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("text", pa.string()),
        ("max_overlap", pa.float64()),
        ("matched_hashes", pa.list_(pa.uint64())),
    ]
)
REMOVED_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("url", pa.string()),
        ("snapshot", pa.string()),
        ("chars", pa.int64()),
        ("matched_hashes", pa.list_(pa.uint64())),
        ("source_file", pa.string()),
    ]
)

_SHARED: dict = {}


def _glob(pattern: str) -> list[str]:
    fs, path = fsspec.core.url_to_fs(pattern)
    protocol = pattern.split("://", 1)[0] + "://" if "://" in pattern else ""
    return sorted(protocol + p for p in fs.glob(path))


def _exists(path: str) -> bool:
    fs, p = fsspec.core.url_to_fs(path)
    return fs.exists(p)


def read_jsonl(path: str) -> Iterator[dict]:
    with fsspec.open(path, "rb") as raw, gzip.open(raw, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def input_files(input_dir: str) -> list[str]:
    """Deduplicated pool shards (``**/*.jsonl.gz``), sorted."""
    return _glob(f"{input_dir.rstrip('/')}/**/*{INPUT_SUFFIX}")


def _pool_map(fn: Callable, items: Sequence, workers: int, **shared) -> list:
    """Run ``fn`` over *items* with a fork-based pool; *shared* objects are inherited copy-on-write."""
    _SHARED.update(shared)
    if workers <= 1 or len(items) <= 1:
        return [fn(item) for item in items]
    with multiprocessing.get_context("fork").Pool(min(workers, len(items))) as pool:
        return list(pool.imap_unordered(fn, items))


# ---- sample (drop-set sample) ---------------------------------------------------------------------


def sample_file(path: str, k: int, seed: int) -> list[dict]:
    """``k``-doc reservoir of one pool file with ``random.Random(seed)``, then shuffled (as-run ``_sample_file``)."""
    rng = random.Random(seed)
    keep: list[dict] = []
    for n, record in enumerate(read_jsonl(path)):
        row = {"id": record[ID_FIELD], "text": record["text"], "snapshot": record["snapshot"]}
        if len(keep) < k:
            keep.append(row)
        elif (j := rng.randrange(n + 1)) < k:
            keep[j] = row
    rng.shuffle(keep)
    return keep


def _sample_task(item: tuple[int, str]) -> dict:
    i, path = item
    sample_dir, per_file = _SHARED["sample_dir"], _SHARED["per_file"]
    by_source: dict[str, list[dict]] = {}
    for row in sample_file(path, per_file, i):
        by_source.setdefault(ds.drop_set_source(row["snapshot"]), []).append({"id": row["id"], "text": row["text"]})
    for source, rows in by_source.items():
        ds.write_parquet(f"{sample_dir}/{source}/part-{i:06d}.parquet", pa.Table.from_pylist(rows))
    return {"file": path, "sources": len(by_source)}


def sample(
    input_dir: str,
    sample_dir: str,
    n_files: int = SAMPLE_FILES,
    per_file: int = SAMPLE_PER_FILE,
    workers: int = 8,
) -> int:
    """Drop-set sample: ``per_file`` random docs from each of ``n_files`` random pool files, split by source.

    The file choice depends on the exact sorted path list of the deduplicated pool (as-run: the
    ``deduped_c075`` layout, 274,767 files), so it cannot be rebuilt from the regrouped release.
    """
    files = random.Random(SAMPLE_SEED).sample(input_files(input_dir), n_files)
    results = _pool_map(
        _sample_task, list(enumerate(files)), workers, sample_dir=sample_dir.rstrip("/"), per_file=per_file
    )
    return len(results)


# ---- mark -----------------------------------------------------------------------------------------


def attributes_path(attributes_root: str, input_path: str, kind: str = "main") -> str:
    name = os.path.basename(input_path).removesuffix(INPUT_SUFFIX) + ".parquet"
    return f"{attributes_root}/outputs/{kind}/{name}"


def mark_file(input_path: str, attributes_root: str, bf: Container[int]) -> dict:
    """Mark every doc of one shard: one required phrase, no drop set (as-run ``_mark_shard``)."""
    rows, reservoir = [], []
    rng = random.Random(input_path)
    n_flagged = 0
    for record in read_jsonl(input_path):
        score, matched = _document_overlap_and_matches(record["text"], bf, MARK_NGRAM)
        rows.append(
            {"id": record[ID_FIELD], "contaminated": bool(matched), "max_overlap": score, "matched_hashes": matched}
        )
        if not matched:
            continue
        n_flagged += 1
        flagged = {"id": record[ID_FIELD], "text": record["text"], "max_overlap": score, "matched_hashes": matched}
        if len(reservoir) < FLAGGED_SAMPLE_PER_FILE:
            reservoir.append(flagged)
        elif (j := rng.randrange(n_flagged)) < FLAGGED_SAMPLE_PER_FILE:
            reservoir[j] = flagged
    if reservoir:
        ds.write_parquet(
            attributes_path(attributes_root, input_path, "flagged_sample"),
            pa.Table.from_pylist(reservoir, schema=FLAGGED_SCHEMA),
        )
    # Written last: an existing main file means the input is fully marked.
    ds.write_parquet(attributes_path(attributes_root, input_path), pa.Table.from_pylist(rows, schema=ATTR_SCHEMA))
    return {"file": input_path, "docs": len(rows), "candidates": n_flagged}


def _mark_task(input_path: str) -> dict:
    return mark_file(input_path, _SHARED["attributes_root"], _SHARED["bf"])


def mark(input_dir: str, attributes_root: str, bf: Container[int], workers: int = 8) -> list[dict]:
    """Mark every shard that has no attributes yet."""
    attributes_root = attributes_root.rstrip("/")
    done = {os.path.basename(p) for p in _glob(f"{attributes_root}/outputs/main/*.parquet")}
    files = [f for f in input_files(input_dir) if os.path.basename(attributes_path(attributes_root, f)) not in done]
    logger.info("mark: %d files left (%d done)", len(files), len(done))
    return _pool_map(_mark_task, files, workers, attributes_root=attributes_root, bf=bf)


# ---- candidate counts (S' with document frequencies) ----------------------------------------------


def _partitioned_write(root: str, kind: str, name: str, table: pa.Table) -> None:
    part_of = pc.cast(table["hash"], pa.uint64()).to_numpy() % COUNT_PARTITIONS
    for part in range(COUNT_PARTITIONS):
        sel = np.flatnonzero(part_of == part)
        ds.write_parquet(f"{root}/{kind}/part={part:03d}/{name}.parquet", table.take(sel))


def _candidate_batch_task(item: tuple[int, list[str]]) -> dict:
    batch_id, paths = item
    counts: Counter[int] = Counter()
    for path in paths:
        table = ds.read_parquet(path).select(["contaminated", "matched_hashes"])
        for row in table.filter(table["contaminated"]).column("matched_hashes").to_pylist():
            counts.update(set(row))
    hashes = np.fromiter(counts.keys(), dtype=np.uint64, count=len(counts))
    dfs = np.fromiter(counts.values(), dtype=np.int64, count=len(counts))
    _partitioned_write(
        _SHARED["root"], "candidate_batches", f"batch-{batch_id:05d}", pa.table({"hash": hashes, "df": dfs})
    )
    return {"batch": batch_id, "hashes": len(counts)}


def _read_partition(root: str, kind: str, part: int) -> pa.Table:
    files = _glob(f"{root}/{kind}/part={part:03d}/*.parquet")
    with concurrent.futures.ThreadPoolExecutor(READ_THREADS) as ex:
        return pa.concat_tables(list(ex.map(ds.read_parquet, files)))


def _candidate_reduce_task(part: int) -> dict:
    root = _SHARED["root"]
    table = _read_partition(root, "candidate_batches", part)
    totals = table.group_by("hash").aggregate([("df", "sum")]).rename_columns(["hash", "df"])
    ds.write_parquet(f"{root}/candidate_totals/part={part:03d}.parquet", totals)
    return {"part": part, "hashes": totals.num_rows}


def candidate_counts(attributes_root: str, boilerplate_root: str, workers: int = 8) -> dict:
    """Per phrase, the number of mark candidates whose ``matched_hashes`` contain it (``candidate_totals``)."""
    root = boilerplate_root.rstrip("/")
    attrs = _glob(f"{attributes_root.rstrip('/')}/outputs/main/*.parquet")
    batches = [
        (i // COUNT_FILES_PER_TASK, attrs[i : i + COUNT_FILES_PER_TASK])
        for i in range(0, len(attrs), COUNT_FILES_PER_TASK)
    ]
    _pool_map(_candidate_batch_task, batches, workers, root=root)
    _pool_map(_candidate_reduce_task, list(range(COUNT_PARTITIONS)), workers, root=root)
    df = np.concatenate(
        [ds.read_parquet(p).column("df").to_numpy() for p in ds.partition_paths(root, "candidate_totals")]
    )
    summary = {
        "attribute_files": len(attrs),
        "distinct_phrases": int(df.size),
        "phrases_at_least": {str(n): int((df >= n).sum()) for n in (50, 100, 500, 1_000, 4_000, 10_000, 100_000)},
    }
    with fsspec.open(f"{root}/candidate_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    return summary


def tracked_phrases(boilerplate_root: str) -> np.ndarray:
    """Sorted hashes of phrases in at least ``TRACK_MIN_CANDIDATE_DF`` candidate docs."""
    arrays = []
    for path in ds.partition_paths(boilerplate_root, "candidate_totals"):
        table = ds.read_parquet(path)
        arrays.append(table.filter(pc.greater_equal(table["df"], TRACK_MIN_CANDIDATE_DF)).column("hash").to_numpy())
    return np.sort(np.concatenate(arrays))


def in_sorted(values: np.ndarray, sorted_set: np.ndarray) -> np.ndarray:
    """Membership mask of *values* in the sorted array *sorted_set* (binary search, no re-sort)."""
    idx = np.searchsorted(sorted_set, values)
    idx[idx == sorted_set.size] = 0
    return sorted_set[idx] == values


# ---- pool counts (pooldf1000) ---------------------------------------------------------------------


def _pool_count_task(item: tuple[int, list[str]]) -> dict:
    batch_id, paths = item
    tracked = _SHARED["tracked"]
    counts: Counter[tuple[int, str]] = Counter()
    docs = 0
    for path in paths:
        for record in read_jsonl(path):
            docs += 1
            feats = np.fromiter(
                {_bloom_hash(f) for f in _extract_features(record["text"], MARK_NGRAM)}, dtype=np.uint64
            )
            hits = feats[in_sorted(feats, tracked)]
            source = ds.drop_set_source(record["snapshot"])
            counts.update((int(h), source) for h in hits)
    hashes = np.fromiter((h for h, _ in counts), dtype=np.uint64, count=len(counts))
    sources = [s for _, s in counts]
    dfs = np.fromiter(counts.values(), dtype=np.int64, count=len(counts))
    table = pa.table({"hash": hashes, "crawl": pa.array(sources, pa.string()), "df": dfs})
    _partitioned_write(_SHARED["root"], "pool_batches", f"batch-{batch_id:06d}", table)
    return {"batch": batch_id, "docs": docs, "pairs": len(counts)}


def _pool_reduce_task(part: int) -> dict:
    root = _SHARED["root"]
    table = _read_partition(root, "pool_batches", part)
    per_crawl = table.group_by(["hash", "crawl"]).aggregate([("df", "sum")])
    totals = per_crawl.group_by("hash").aggregate([("df_sum", "sum"), ("crawl", "count")])
    ds.write_parquet(f"{root}/pool_totals/part={part:03d}.parquet", totals.rename_columns(["hash", "df", "crawls"]))
    return {"part": part, "hashes": totals.num_rows}


def pool_counts(input_dir: str, boilerplate_root: str, workers: int = 8) -> dict:
    """Exact pool-wide doc and crawl counts of every tracked phrase (``pool_totals``), then the reduce."""
    root = boilerplate_root.rstrip("/")
    files = input_files(input_dir)
    n = POOL_COUNT_FILES_PER_TASK
    batches = [(i // n, files[i : i + n]) for i in range(0, len(files), n)]
    done = {
        int(os.path.basename(p).removeprefix("batch-").removesuffix(".parquet"))
        for p in _glob(f"{root}/pool_batches/part={COUNT_PARTITIONS - 1:03d}/*.parquet")
    }
    todo = [b for b in batches if b[0] not in done]
    _pool_map(_pool_count_task, todo, workers, root=root, tracked=tracked_phrases(root))
    _pool_map(_pool_reduce_task, list(range(COUNT_PARTITIONS)), workers, root=root)
    tables = [ds.read_parquet(p) for p in ds.partition_paths(root, "pool_totals")]
    df = np.concatenate([t.column("df").to_numpy() for t in tables])
    crawls = np.concatenate([t.column("crawls").to_numpy() for t in tables])
    summary = {
        "tracked_phrases": int(df.size),
        "phrases_at_least_3_crawls_and_df": {
            str(n): int(((df >= n) & (crawls >= 3)).sum()) for n in (50, 500, 1_000, 4_000, 10_000, 100_000)
        },
    }
    with fsspec.open(f"{root}/pool_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    return summary


# ---- code counts (codedf100) ----------------------------------------------------------------------


def code_files(code_corpus: str) -> list[tuple[str, str]]:
    """``(language, file)`` pairs: ``random.Random(language).sample`` of each language's sorted parquet files."""
    items = []
    for language in CODE_LANGUAGES:
        files = _glob(f"{code_corpus.rstrip('/')}/{language}/*.parquet")
        if not files:
            raise FileNotFoundError(f"no StarCoderData files for {language}")
        chosen = random.Random(language).sample(files, min(CODE_FILES_PER_LANGUAGE, len(files)))
        items += [(language, f) for f in chosen]
    return items


def _code_count_task(item: tuple[int, str, str]) -> dict:
    task_id, language, path = item
    tracked = _SHARED["tracked"]
    counts: Counter[int] = Counter()
    files = 0
    with fsspec.open(path, "rb") as fh:
        for batch in pq.ParquetFile(fh).iter_batches(columns=["content"], batch_size=5_000):
            for content in batch.column("content").to_pylist():
                if not content:
                    continue
                files += 1
                feats = np.fromiter({_bloom_hash(f) for f in _extract_features(content, MARK_NGRAM)}, dtype=np.uint64)
                counts.update(int(h) for h in feats[in_sorted(feats, tracked)])
    hashes = np.fromiter(counts.keys(), dtype=np.uint64, count=len(counts))
    dfs = np.fromiter(counts.values(), dtype=np.int64, count=len(counts))
    table = pa.table({"hash": hashes, "language": pa.array([language] * len(counts)), "df": dfs})
    _partitioned_write(_SHARED["root"], "code_batches", f"task-{task_id:05d}", table)
    return {"task": task_id, "language": language, "files": files}


def _code_reduce_task(part: int) -> dict:
    root = _SHARED["root"]
    table = _read_partition(root, "code_batches", part)
    per_lang = table.group_by(["hash", "language"]).aggregate([("df", "sum")])
    totals = per_lang.group_by("hash").aggregate([("df_sum", "sum"), ("language", "count")])
    ds.write_parquet(f"{root}/code_totals/part={part:03d}.parquet", totals.rename_columns(["hash", "df", "languages"]))
    return {"part": part, "hashes": totals.num_rows}


def code_counts(code_corpus: str, boilerplate_root: str, workers: int = 8) -> dict:
    """Files and languages of a StarCoderData sample containing each tracked phrase (``code_totals``)."""
    root = boilerplate_root.rstrip("/")
    tasks = [(i, language, f) for i, (language, f) in enumerate(code_files(code_corpus))]
    _pool_map(_code_count_task, tasks, workers, root=root, tracked=tracked_phrases(root))
    _pool_map(_code_reduce_task, list(range(COUNT_PARTITIONS)), workers, root=root)
    tables = [ds.read_parquet(p) for p in ds.partition_paths(root, "code_totals")]
    df = np.concatenate([x.column("df").to_numpy() for x in tables])
    langs = np.concatenate([x.column("languages").to_numpy() for x in tables])
    summary = {
        "source_files": len(tasks),
        "phrases_seen": int(df.size),
        "phrases_at_least_df": {str(n): int((df >= n).sum()) for n in (10, 50, 100, 500, 1_000)},
        "phrases_at_least_df_2_languages": {str(n): int(((df >= n) & (langs >= 2)).sum()) for n in (10, 50, 100, 500)},
    }
    with fsspec.open(f"{root}/code_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    return summary


# ---- two-pass filter and manifest -----------------------------------------------------------------


def filter_file(
    input_path: str, attributes_root: str, output_dir: str, removed_dir: str, bf: Container[int], drop: frozenset[int]
) -> dict:
    """Re-score the mark candidates of one shard; write the kept records and the removed-doc sidecar."""
    table = ds.read_parquet(attributes_path(attributes_root, input_path)).select(["id", "contaminated"])
    candidates = {r["id"] for r in table.to_pylist() if r["contaminated"]}
    name = os.path.basename(input_path)
    removed = []
    kept = n = kept_chars = removed_chars = seen_candidates = 0
    with fsspec.open(f"{output_dir}/{name}", "wb") as raw, gzip.open(raw, "wt", encoding="utf-8") as out:
        for record in read_jsonl(input_path):
            n += 1
            matched: list[int] = []
            if record[ID_FIELD] in candidates:
                seen_candidates += 1
                matched = removal_matches(record["text"], bf, drop)
            if not matched:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
                kept_chars += len(record["text"])
                continue
            removed_chars += len(record["text"])
            removed.append(
                {
                    "doc_id": record[ID_FIELD],
                    "url": record.get("url"),
                    "snapshot": record.get("snapshot"),
                    "chars": len(record["text"]),
                    "matched_hashes": matched,
                    "source_file": name,
                }
            )
    if n != table.num_rows:
        raise RuntimeError(f"{input_path}: {n} docs but {table.num_rows} attribute rows")
    if seen_candidates != len(candidates):
        raise RuntimeError(f"{input_path}: {len(candidates)} candidate ids but {seen_candidates} seen")
    ds.write_parquet(
        f"{removed_dir}/{name.removesuffix(INPUT_SUFFIX)}.parquet", pa.Table.from_pylist(removed, schema=REMOVED_SCHEMA)
    )
    return {
        "file": name,
        "docs": n,
        "candidates": len(candidates),
        "kept": kept,
        "removed": len(removed),
        "chars_kept": kept_chars,
        "chars_removed": removed_chars,
    }


def _filter_task(input_path: str) -> dict:
    s = _SHARED
    return filter_file(input_path, s["attributes_root"], s["output_dir"], s["removed_dir"], s["bf"], s["drop"])


def filter_pool(
    input_dir: str,
    attributes_root: str,
    output_dir: str,
    removed_dir: str,
    bf: Container[int],
    drop: frozenset[int],
    workers: int = 8,
) -> list[dict]:
    """Two-pass filter over every shard without a sidecar (the sidecar is written after the output)."""
    done = {os.path.basename(p) for p in _glob(f"{removed_dir.rstrip('/')}/*.parquet")}
    todo = [
        x for x in input_files(input_dir) if f"{os.path.basename(x).removesuffix(INPUT_SUFFIX)}.parquet" not in done
    ]
    shared = {
        "attributes_root": attributes_root.rstrip("/"),
        "output_dir": output_dir.rstrip("/"),
        "removed_dir": removed_dir.rstrip("/"),
        "bf": bf,
        "drop": drop,
    }
    return _pool_map(_filter_task, todo, workers, **shared)


def matched_eval_names(index_paths: Iterable[str], wanted: pa.Array) -> dict[int, set[str]]:
    """``hash -> eval names`` for the *wanted* hashes, from hash-index parquet files (as-run ``manifest``)."""
    evals_of: dict[int, set[str]] = {}
    for index in index_paths:
        with fsspec.open(index, "rb") as fh:
            for batch in pq.ParquetFile(fh).iter_batches(columns=["hash", "eval_id"], batch_size=INDEX_BATCH_ROWS):
                hits = batch.filter(pc.is_in(batch.column("hash"), value_set=wanted))
                for h, eval_id in zip(hits.column("hash").to_pylist(), hits.column("eval_id").to_pylist(), strict=True):
                    evals_of.setdefault(h, set()).add(eval_name(eval_id))
    return evals_of


def manifest(removed_dir: str, index_paths: Sequence[str], out_path: str) -> int:
    """Concatenate the removed-doc sidecars and add ``matched_evals`` (sorted eval names per doc)."""
    parts = _glob(f"{removed_dir.rstrip('/')}/*.parquet")
    with concurrent.futures.ThreadPoolExecutor(64) as ex:
        tables = list(ex.map(ds.read_parquet, parts))
    removed = pa.concat_tables(tables) if tables else REMOVED_SCHEMA.empty_table()
    wanted = pc.unique(pc.list_flatten(removed.column("matched_hashes")))
    evals_of = matched_eval_names(index_paths, wanted)
    matched_evals = [
        sorted({e for h in hashes for e in evals_of.get(h, ())})
        for hashes in removed.column("matched_hashes").to_pylist()
    ]
    removed = removed.append_column("matched_evals", pa.array(matched_evals, pa.list_(pa.string())))
    ds.write_parquet(out_path, removed)
    return removed.num_rows
