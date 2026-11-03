# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-WARC outputs of extraction + classification, and the marker that commits them.

Layout under an output root (a local directory or a ``gs://``/``s3://`` prefix)::

    kept/data-{warc_hash}.parquet         kept documents, schemas.KEPT_SCHEMA, zstd level 12
    tombstones/data-{warc_hash}.jsonl.gz  one line per dropped fastText survivor; absent if none
    markers/data-{warc_hash}.json         written last: a WARC is done iff its marker exists
    attempts/data-{warc_hash}.json        how often the WARC was started, to abandon poison WARCs

The kept and tombstone files are byte-compatible with production's (``write_table_v3`` and
``write_tombstones_band``): re-serializing an as-run kept table gives the identical file, and
tombstones decompress to identical bytes (the gzip header carries the write time).
"""

import dataclasses
import gzip
import io
import json
from enum import StrEnum

import pyarrow as pa
import pyarrow.parquet as pq

from atlantic_cc.io.store import Store, dumps, read_json
from atlantic_cc.schemas import KEPT_SCHEMA, KEPT_ZSTD_LEVEL

KEPT_DIR = "kept"
TOMBSTONES_DIR = "tombstones"
MARKERS_DIR = "markers"
ATTEMPTS_DIR = "attempts"


def kept_path(root: str, warc_hash: str) -> str:
    return f"{root}/{KEPT_DIR}/data-{warc_hash}.parquet"


def tombstones_path(root: str, warc_hash: str) -> str:
    return f"{root}/{TOMBSTONES_DIR}/data-{warc_hash}.jsonl.gz"


def marker_path(root: str, warc_hash: str) -> str:
    return f"{root}/{MARKERS_DIR}/data-{warc_hash}.json"


def attempts_path(root: str, warc_hash: str) -> str:
    return f"{root}/{ATTEMPTS_DIR}/data-{warc_hash}.json"


def kept_parquet_bytes(kept: pa.Table) -> bytes:
    """The kept parquet file: pyarrow defaults (one row group per WARC) with zstd level 12."""
    if not kept.schema.equals(KEPT_SCHEMA):
        raise ValueError(f"kept table schema differs from schemas.KEPT_SCHEMA:\n{kept.schema}")
    buffer = io.BytesIO()
    pq.write_table(kept, buffer, compression="zstd", compression_level=KEPT_ZSTD_LEVEL)
    return buffer.getvalue()


def tombstones_jsonl_gz_bytes(rows: list[dict]) -> bytes:
    """gzip (level 9, no file name) of one ``json.dumps(row)`` line per row, each ending in a newline."""
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb") as gz:
        for row in rows:
            gz.write((json.dumps(row) + "\n").encode("utf-8"))
    return buffer.getvalue()


def write_warc_outputs(store: Store, root: str, warc_hash: str, kept: pa.Table, tombstones: list[dict]) -> None:
    """Write the kept file (always, even with zero rows) and the tombstones (only if there are any)."""
    store.write(kept_path(root, warc_hash), kept_parquet_bytes(kept))
    path = tombstones_path(root, warc_hash)
    if tombstones:
        store.write(path, tombstones_jsonl_gz_bytes(tombstones))
    else:
        # An interrupted earlier attempt on different hardware may have dropped documents this one keeps.
        store.delete(path)


@dataclasses.dataclass(frozen=True)
class PhaseBCounters:
    """Phase B's document counts and stage wall-clock seconds for one WARC (``phase_b.run_phase_b``)."""

    n_presurvivors: int
    n_hi_accept: int
    n_band: int
    n_kept: int
    tokenize_seconds: float
    pooled_seconds: float
    terminal_seconds: float


class WarcStatus(StrEnum):
    OK = "ok"
    DOWNLOAD_FAILED = "download_failed"  # the WARC could not be fetched; no documents
    ABANDONED = "abandoned"  # started MAX_WARC_ATTEMPTS times without finishing; no documents


@dataclasses.dataclass(frozen=True)
class WarcMarker:
    """What one WARC produced, where and how fast; times are wall-clock seconds."""

    warc_path: str
    warc_hash: str
    status: WarcStatus
    error: str  # why a failed WARC has no documents; "" when the status is ok
    attempt: int
    # phase A (decode, extract, fastText)
    n_in: int
    n_prefilter_dropped: int
    n_extract_empty: int
    n_extract_crashed: int
    # phase B (pooled band, terminal model)
    n_presurvivors: int
    n_hi_accept: int
    n_band: int
    n_kept: int
    phase_a_seconds: float
    tokenize_seconds: float
    pooled_seconds: float
    terminal_seconds: float
    write_seconds: float
    # where phase B ran, e.g. backend "jax", device "TPU v4", 4 devices
    backend: str
    device: str
    n_devices: int


def write_marker(store: Store, root: str, marker: WarcMarker) -> None:
    store.write(marker_path(root, marker.warc_hash), dumps(dataclasses.asdict(marker)))


def read_marker(store: Store, root: str, warc_hash: str) -> WarcMarker:
    fields = read_json(store, marker_path(root, warc_hash))
    return WarcMarker(**{**fields, "status": WarcStatus(fields["status"])})


def is_done(store: Store, root: str, warc_hash: str) -> bool:
    return store.stat(marker_path(root, warc_hash)) is not None


def record_attempt(store: Store, root: str, warc_hash: str) -> int:
    """Count one more start of this WARC and return the count (1 on the first start)."""
    path = attempts_path(root, warc_hash)
    attempts = 1 if store.stat(path) is None else read_json(store, path)["attempts"] + 1
    store.write(path, dumps({"attempts": attempts}))
    return attempts
