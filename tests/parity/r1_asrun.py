# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Where the as-run per-WARC outputs live, and how to read them (maintainers with Marin GCS access).

* Central records, small, in ``gs://marin-us-central1``: the shard lists; the fused path's per-WARC
  ``_b_marks`` marker (counters, region, device; its write time is the processing time); for rung 1
  (shards 0-403, the two-phase path) the shard's catalog row.
* ``kept/data-{h}.parquet``: every WARC, rung 1 included, consolidated in us-central2 under the fused
  namespace. The ``kept_path`` that markers and catalog rows record names the bucket of the region
  that processed the WARC, where the file was first written; that copy no longer exists.
* In that bucket (e.g. ``gs://marin-eu-west4`` for region ``europe-west4``), under the code path's
  namespace: ``tombstones/data-{h}.jsonl.gz`` (absent when nothing was dropped).
* Rung 1 ran phase A on CPU workers and phase B on TPUs, not always in the same region, and its
  catalog records the phase-B region. Its phase-A counters ``timing_a/data-{h}.json`` (284 B) are in
  the buckets of the regions that ran phase A, sometimes twice; every copy carries the same counters
  (checked on 10 WARCs), and for 1 of those 10 the phase-B region has none.

Regional files are read by a job in their region (:func:`load_ref`, which condenses them into a small
:class:`AsRunRef`), ``kept/`` by a job in us-central2 (:func:`read_kept`). The one exception is a
rung-1 ``timing_a`` missing from the phase-B region, read from another region's bucket.
"""

import gzip
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq

from atlantic_cc.warc.identity import warc_hash
from tests.parity.r1_compare import KeptDoc, Reference, text_sha256

FUSED_NS = "documents/fast_curation/lpv11_fastpipe_v2_1_fused-2a076ddfcb"
RUNG1_NS = "documents/fast_curation/lpv11_fastpipe_v2_1-944ca6bc38"
CENTRAL_BUCKET = "gs://marin-us-central1"  # shard lists, markers, catalogs
# Every bucket production wrote per-WARC files to; us-central2 first (rung 1's CPU phase-A home).
PRODUCTION_BUCKETS = (
    "gs://marin-us-central2",
    "gs://marin-us-central1",
    "gs://marin-us-east1",
    "gs://marin-us-east5",
    "gs://marin-us-west4",
    "gs://marin-eu-west4",
)
KEPT_REGION = "us-central2"
KEPT_PREFIX = f"gs://marin-us-central2/{FUSED_NS}/kept"
RUNG1_LAST_SHARD = 403
# The Content-Length guard is absent from bundles up to 2026-09-07 (the 2M ladder ended then) and
# present in every bundle of the 8M segment (from 2026-09-26) and in the reopen passes.
GUARD_ABSENT_BEFORE = datetime(2026, 9, 8, tzinfo=UTC)
GUARD_PRESENT_FROM = datetime(2026, 9, 18, tzinfo=UTC)
PHASE_A_COUNTERS = ("n_in", "n_prefilter_dropped", "n_extract_empty", "n_extract_crashed")
RUNG1_ROW_COUNTERS = ("n_presurvivors", "n_hi_accept", "n_band", "n_kept", "device", "n_devices", "region")
MARKER_EXTRA_FIELDS = ("device", "n_devices", "region", "a_download_failed")


@dataclass
class AsRunRef:
    """One WARC's as-run record apart from ``kept/``, as read in the region that processed it."""

    warc_path: str
    warc_hash: str
    shard: int
    region: str  # GCP region of the worker, e.g. "europe-west4"
    processed_at: str  # ISO 8601
    counters: dict  # a_n_* and n_* counters, device, n_devices, region (and a_download_failed if set)
    raw: dict  # the source records, with their paths
    tombstones_path: str
    tombstones_md5: str | None  # None when the file is absent (nothing was dropped)
    tombstones: list[dict]  # the tombstone file's rows, in file order

    @property
    def code_path(self) -> str:
        return "fused" if self.shard > RUNG1_LAST_SHARD else "two-phase (rung 1)"

    @property
    def content_length_guard(self) -> bool | None:
        processed_at = datetime.fromisoformat(self.processed_at)
        if processed_at < GUARD_ABSENT_BEFORE:
            return False
        if processed_at >= GUARD_PRESENT_FROM:
            return True
        return None


def _json(path: str) -> dict:
    with fsspec.open(path, "r") as f:
        return json.load(f)


def _updated(path: str) -> datetime:
    fs, root = fsspec.core.url_to_fs(path)
    return datetime.fromisoformat(fs.info(root)["updated"].replace("Z", "+00:00"))


def gcs_md5(path: str) -> str:
    fs, root = fsspec.core.url_to_fs(path)
    return fs.info(root)["md5Hash"]


def shard_warc_paths(shard: int) -> list[str]:
    with fsspec.open(f"{CENTRAL_BUCKET}/{FUSED_NS}/shards/shard-{shard:05d}.parquet", "rb") as f:
        return pq.read_table(f, columns=["warc_path"]).column("warc_path").to_pylist()


def _namespace(shard: int) -> str:
    return FUSED_NS if shard > RUNG1_LAST_SHARD else RUNG1_NS


def central_record(shard: int, h: str) -> tuple[str, dict]:
    """``(path, record)``: the fused marker, or the rung-1 catalog row, of WARC ``h``."""
    if shard > RUNG1_LAST_SHARD:
        path = f"{CENTRAL_BUCKET}/{FUSED_NS}/_b_marks/shard-{shard:05d}/data-{h}"
        return path, _json(path)
    path = f"{CENTRAL_BUCKET}/{RUNG1_NS}/catalog/shard-{shard:05d}.parquet"
    with fsspec.open(path, "rb") as f:
        (row,) = [r for r in pq.read_table(f).to_pylist() if r["warc_hash"] == h]
    return path, row


def region_bucket(record: dict, shard: int, h: str) -> str:
    """The processing region's bucket, taken from the recorded ``kept_path`` (not from the region name)."""
    recorded = record["kept_path"]
    bucket = "gs://" + recorded.removeprefix("gs://").split("/", 1)[0]
    assert recorded == f"{bucket}/{_namespace(shard)}/kept/data-{h}.parquet", recorded
    return bucket


def rung1_timing_path(bucket: str, h: str) -> str:
    """A rung-1 WARC's ``timing_a``: the copy in the phase-B region's ``bucket``, else another region's."""
    for candidate in (bucket, *(b for b in PRODUCTION_BUCKETS if b != bucket)):
        path = f"{candidate}/{RUNG1_NS}/timing_a/data-{h}.json"
        fs, root = fsspec.core.url_to_fs(path)
        if fs.exists(root):
            return path
    raise FileNotFoundError(f"no timing_a/data-{h}.json in {PRODUCTION_BUCKETS}")


def load_ref(warc_path: str, shard: int) -> AsRunRef:
    """Read one WARC's central record and its regional files; run this in the record's region."""
    h = warc_hash(warc_path)
    record_path, record = central_record(shard, h)
    bucket = region_bucket(record, shard, h)
    ns = _namespace(shard)
    if shard > RUNG1_LAST_SHARD:
        counters = {k: v for k, v in record.items() if k.startswith(("a_n_", "n_")) or k in MARKER_EXTRA_FIELDS}
        raw = {"marker": {"path": record_path, "record": record}}
        processed_at = _updated(record_path)
    else:
        timing_path = rung1_timing_path(bucket, h)
        timing = _json(timing_path)
        counters = {f"a_{k}": timing[k] for k in PHASE_A_COUNTERS} | {k: record[k] for k in RUNG1_ROW_COUNTERS}
        raw = {
            "catalog_row": {"path": record_path, "record": record},
            "timing_a": {"path": timing_path, "record": timing},
        }
        processed_at = _updated(timing_path)

    tombstones_path = f"{bucket}/{ns}/tombstones/data-{h}.jsonl.gz"
    fs, root = fsspec.core.url_to_fs(tombstones_path)
    tombstones, md5 = [], None
    if fs.exists(root):
        md5 = fs.info(root)["md5Hash"]
        with fs.open(root, "rb") as f:
            tombstones = [json.loads(line) for line in gzip.decompress(f.read()).decode().splitlines()]
    return AsRunRef(
        warc_path=warc_path,
        warc_hash=h,
        shard=shard,
        region=record["region"],
        processed_at=processed_at.isoformat(),
        counters=counters,
        raw=raw,
        tombstones_path=tombstones_path,
        tombstones_md5=md5,
        tombstones=tombstones,
    )


def ref_path(refs_dir: str, h: str) -> str:
    return f"{refs_dir.rstrip('/')}/{h}.json.gz"


def write_ref(ref: AsRunRef, refs_dir: str) -> None:
    with fsspec.open(ref_path(refs_dir, ref.warc_hash), "wb") as f:
        f.write(gzip.compress(json.dumps(asdict(ref)).encode(), mtime=0))


def read_ref(refs_dir: str, h: str) -> AsRunRef:
    with fsspec.open(ref_path(refs_dir, h), "rb") as f:
        return AsRunRef(**json.loads(gzip.decompress(f.read())))


def kept_path(h: str) -> str:
    return f"{KEPT_PREFIX}/data-{h}.parquet"


def read_kept(h: str) -> pa.Table:
    """The WARC's consolidated ``kept/`` table (2-25 MB; read it in us-central2)."""
    with fsspec.open(kept_path(h), "rb") as f:
        return pq.read_table(f)


def reference(ref: AsRunRef, kept: pa.Table) -> Reference:
    ids = kept.column("doc_id").to_pylist()
    texts = kept.column("text").to_pylist()
    scores = kept.column("fasttext_score").to_numpy()
    n_tokens = kept.column("n_tokens").to_pylist()
    kept_docs = {d: KeptDoc(text_sha256(t), s, n) for d, t, s, n in zip(ids, texts, scores, n_tokens, strict=True)}
    names = ("a_n_in", "a_n_prefilter_dropped", "a_n_extract_empty", "a_n_extract_crashed", "n_presurvivors")
    return Reference(
        kept=kept_docs,
        tombstones=[t["doc_id"] for t in ref.tombstones],
        counters={k: ref.counters[k] for k in names},
        texts=dict(zip(ids, texts, strict=True)),
    )
