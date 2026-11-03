# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The hash lists that make decontamination replayable on the released text.

* ``drop_hashes`` (D, 66,558 uint64): the union of the four boilerplate drop lists. Shipped in this
  package (``data/drop_hashes.npy``).
* ``eval_hits`` (S', 10,944,079 uint64): every eval phrase the as-run mark matched anywhere in the pool
  (``decon_v4/boilerplate/candidate_totals``). On the released documents, membership in S' gives the
  same decisions and matched hashes as the ``full-val`` bloom (see ``docs/decon.md``). Release asset.
* ``eval_hit_names``: ``hash -> sorted eval names`` for every S' hash, joined from the two as-run hash
  indexes (``bloom-full/eval_hash_index.parquet`` and ``bloom-full-val/val_index.parquet``). Release asset.

Arrays are stored sorted, little-endian uint64; digests are sha256 over those bytes.
"""

import hashlib
import json
import logging
import os
import time
from collections.abc import Iterator, Mapping
from importlib import resources

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from atlantic_cc.decon import dropsets as ds
from atlantic_cc.decon.bloom import PyBloom, load_native_bloom, local_copy
from atlantic_cc.decon.names import eval_name
from atlantic_cc.decon.params import DROP_HASH_COUNT, EVAL_HIT_COUNT

logger = logging.getLogger(__name__)

DROP_HASHES_RESOURCE = "data/drop_hashes.npy"
DROP_HASHES_SHA256 = "5d71024672f39cd7e52fe4ae97c6d846481cbcf290aacf5d9cbd4ed787795fb8"
EVAL_HITS_FILE = "eval_hits.npy"
EVAL_HIT_NAMES_FILE = "eval_hit_names.parquet"
INDEX_BATCH_ROWS = 1_000_000
NAMES_SCHEMA = pa.schema([("hash", pa.uint64()), ("eval_names", pa.list_(pa.string()))])

# As-run sources (Marin GCS, us-central2; maintainers only).
AS_RUN_POOL = "gs://marin-us-central2/documents/baseline_lpv11_fastpipe_v2_1_fused_deduped/8582115warcs"
AS_RUN_DECON = f"{AS_RUN_POOL}/decon_v4"
AS_RUN_BLOOM = f"{AS_RUN_DECON}/bloom-full-val/filter.bin"
AS_RUN_INDEXES = (
    f"{AS_RUN_DECON}/bloom-full/eval_hash_index.parquet",
    f"{AS_RUN_DECON}/bloom-full-val/val_index.parquet",
)
AS_RUN_DROPSETS = f"{AS_RUN_DECON}/dropsets-full-val"
AS_RUN_BOILERPLATE = f"{AS_RUN_DECON}/boilerplate"


def sha256_uint64(values: np.ndarray) -> str:
    """sha256 of a uint64 array as little-endian bytes."""
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<u8").tobytes()).hexdigest()


def save_uint64(path: str, values: np.ndarray) -> None:
    with fsspec.open(path, "wb") as fh:
        np.save(fh, np.ascontiguousarray(values, dtype="<u8"), allow_pickle=False)


def load_uint64(path: str, expected_sha256: str | None = None) -> np.ndarray:
    with fsspec.open(path, "rb") as fh:
        values = np.load(fh, allow_pickle=False)
    if values.dtype != np.dtype("<u8"):
        raise ValueError(f"{path}: dtype {values.dtype}, expected little-endian uint64")
    if expected_sha256 is not None and sha256_uint64(values) != expected_sha256:
        raise ValueError(f"{path}: sha256 {sha256_uint64(values)} != {expected_sha256}")
    return values


def load_drop_hashes() -> frozenset[int]:
    """D, the 66,558 boilerplate drop hashes shipped with this package (digest-checked)."""
    with resources.files("atlantic_cc.decon").joinpath(DROP_HASHES_RESOURCE).open("rb") as fh:
        values = np.load(fh, allow_pickle=False)
    if len(values) != DROP_HASH_COUNT or sha256_uint64(values) != DROP_HASHES_SHA256:
        raise ValueError("packaged drop_hashes.npy does not match the pinned digest")
    return frozenset(values.tolist())


def load_eval_hits(path: str, expected_sha256: str | None = None) -> frozenset[int]:
    """S' as a frozenset of Python ints (about 1 GB of memory for 10.9M hashes)."""
    return frozenset(load_uint64(path, expected_sha256).tolist())


class EvalNameMap(Mapping[int, list[str]]):
    """``hash -> sorted eval names`` backed by the sorted Arrow table (about 100 MB instead of GBs of dicts)."""

    def __init__(self, table: pa.Table) -> None:
        hashes = table["hash"].to_numpy()
        if hashes.size and not (np.diff(hashes) > 0).all():
            raise ValueError("eval name map must be sorted by hash without duplicates")
        self._hashes = hashes
        self._names = table["eval_names"].combine_chunks()

    def _index(self, h: int) -> int | None:
        i = int(np.searchsorted(self._hashes, np.uint64(h)))
        return i if i < self._hashes.size and int(self._hashes[i]) == h else None

    def __getitem__(self, h: int) -> list[str]:
        i = self._index(h)
        if i is None:
            raise KeyError(h)
        return self._names[i].as_py()

    def __contains__(self, h: object) -> bool:
        return isinstance(h, int) and 0 <= h < 2**64 and self._index(h) is not None

    def __iter__(self) -> Iterator[int]:
        return iter(self._hashes.tolist())

    def __len__(self) -> int:
        return int(self._hashes.size)


def load_eval_hit_names(path: str) -> EvalNameMap:
    with fsspec.open(path, "rb") as fh:
        return EvalNameMap(pq.read_table(fh))


def drop_hashes_from_as_run(dropsets_root: str = AS_RUN_DROPSETS, boilerplate: str = AS_RUN_BOILERPLATE) -> np.ndarray:
    """Recompute D from the stored component tables (as-run ``_drop_hashes(full-val, per-crawl, 1000, 100)``)."""
    lists = [ds.read_parquet(p, columns=["hash"])["hash"].to_pylist() for p in ds.drop_tables_dirs(dropsets_root)]
    pool = ds.pool_common_hashes(ds.read_parquet(p) for p in ds.partition_paths(boilerplate, "pool_totals"))
    code = ds.code_common_hashes(ds.read_parquet(p) for p in ds.partition_paths(boilerplate, "code_totals"))
    return ds.drop_set_union(lists, pool, code)


def eval_hits_from_as_run(boilerplate: str = AS_RUN_BOILERPLATE) -> np.ndarray:
    """S': the sorted distinct hashes of ``candidate_totals`` (every mark-matched eval phrase)."""
    arrays = [
        ds.read_parquet(p, columns=["hash"])["hash"].to_numpy()
        for p in ds.partition_paths(boilerplate, "candidate_totals")
    ]
    return np.unique(np.concatenate(arrays))


def eval_hit_names(index_paths: tuple[str, ...], hits: np.ndarray) -> pa.Table:
    """``hash -> sorted eval names`` for every hash in *hits* (hashes absent from both indexes get ``[]``)."""
    value_set = pa.array(hits, pa.uint64())
    pairs: list[pa.Table] = []
    name_cache: dict[str, str] = {}
    for index in index_paths:
        with fsspec.open(index, "rb") as fh:
            for batch in pq.ParquetFile(fh).iter_batches(columns=["hash", "eval_id"], batch_size=INDEX_BATCH_ROWS):
                matched = batch.filter(pc.is_in(batch.column("hash"), value_set=value_set))
                if matched.num_rows == 0:
                    continue
                ids = matched.column("eval_id").dictionary_encode()
                names = [name_cache.setdefault(i, eval_name(i)) for i in ids.dictionary.to_pylist()]
                pairs.append(
                    pa.table(
                        {
                            "hash": matched.column("hash"),
                            "name": pa.DictionaryArray.from_arrays(ids.indices, pa.array(names, pa.string())).cast(
                                pa.string()
                            ),
                        }
                    )
                )
        logger.info("joined %s", index)
    grouped = pa.concat_tables(pairs).group_by("hash").aggregate([("name", "distinct")])
    names_of = dict(zip(grouped["hash"].to_pylist(), grouped["name_distinct"].to_pylist(), strict=True))
    hash_list = hits.tolist()
    return pa.table(
        {
            "hash": pa.array(hash_list, pa.uint64()),
            "eval_names": pa.array([sorted(names_of.get(h, ())) for h in hash_list], pa.list_(pa.string())),
        },
        schema=NAMES_SCHEMA,
    )


def build_as_run_artifacts(out_dir: str, check_bloom: bool = True) -> dict:
    """Build D, S' and the S' name map from the as-run GCS tables (in-region job; maintainers only)."""
    t0 = time.time()
    report: dict = {}
    drop = drop_hashes_from_as_run()
    save_uint64(f"{out_dir}/drop_hashes.npy", drop)
    report["drop_hashes"] = {"count": int(drop.size), "sha256": sha256_uint64(drop), "expected_count": DROP_HASH_COUNT}

    hits = eval_hits_from_as_run()
    save_uint64(f"{out_dir}/{EVAL_HITS_FILE}", hits)
    report["eval_hits"] = {"count": int(hits.size), "sha256": sha256_uint64(hits), "expected_count": EVAL_HIT_COUNT}
    logger.info("D %d, S' %d (%.0fs)", drop.size, hits.size, time.time() - t0)

    names = eval_hit_names(AS_RUN_INDEXES, hits)
    with fsspec.open(f"{out_dir}/{EVAL_HIT_NAMES_FILE}", "wb") as fh:
        pq.write_table(names, fh, compression="zstd")
    lengths = pc.list_value_length(names["eval_names"]).to_numpy()
    distinct_names = pc.unique(pc.list_flatten(names["eval_names"]))
    report["eval_hit_names"] = {
        "rows": names.num_rows,
        "hashes_without_names": int((lengths == 0).sum()),
        "max_names_per_hash": int(lengths.max()),
        "distinct_eval_names": len(distinct_names),
        "pairs": int(lengths.sum()),
    }
    logger.info("names: %s (%.0fs)", report["eval_hit_names"], time.time() - t0)

    report["drop_subset_of_eval_hits"] = bool(np.isin(drop, hits).all())
    if check_bloom:
        local = local_copy(AS_RUN_BLOOM)
        native, py = load_native_bloom(local), PyBloom.open(local)
        hit_list = hits.tolist()
        report["eval_hits_in_bloom_pybloom"] = sum(h in py for h in hit_list)
        report["eval_hits_in_bloom_native"] = sum(h in native for h in hit_list)
        report["drop_in_bloom_pybloom"] = sum(h in py for h in drop.tolist())
        index_rows = index_in_bloom = 0
        for index in AS_RUN_INDEXES:
            with fsspec.open(index, "rb") as fh:
                for batch in pq.ParquetFile(fh).iter_batches(columns=["hash"], batch_size=INDEX_BATCH_ROWS):
                    column = batch.column("hash").to_pylist()
                    index_rows += len(column)
                    index_in_bloom += sum(h in native for h in column)
        report["index_rows"] = index_rows
        report["index_rows_in_bloom_native"] = index_in_bloom
        logger.info("bloom checks done (%.0fs)", time.time() - t0)
    report["seconds"] = round(time.time() - t0, 1)
    with fsspec.open(f"{out_dir}/artifacts_report.json", "w") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
    return report


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with fsspec.open(path, "rb") as fh:
        while chunk := fh.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def package_data_path() -> str:
    return os.path.join(os.path.dirname(__file__), DROP_HASHES_RESOURCE)
