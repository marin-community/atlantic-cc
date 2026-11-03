# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Where production's per-WARC outputs live on Marin's GCS, and the fixture WARCs F1-F6.

Every WARC's ``kept/`` file was consolidated to us-central2. Its tombstones stayed in the region of
the worker that processed it, which the shard's catalog row records (``region``, and the TPU in
``device``). Shards 0-403 ran production's two-phase path, whose catalog and tombstones live under
its own namespace. Reading these needs Marin GCS credentials; run in us-central2 for the kept files.
"""

import gzip
import io
import json
import os
from dataclasses import dataclass

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq

from atlantic_cc.warc.identity import warc_hash
from atlantic_cc.worklist.identity import hash_shard

FUSED_NAMESPACE = "documents/fast_curation/lpv11_fastpipe_v2_1_fused-2a076ddfcb"
TWO_PHASE_NAMESPACE = "documents/fast_curation/lpv11_fastpipe_v2_1-944ca6bc38"
TWO_PHASE_SHARDS = 404  # shards 0-403
KEPT_BUCKET = "gs://marin-us-central2"
CATALOG_BUCKET = "gs://marin-us-central1"
BASE_SHARDS = 32_000
REGION_BUCKETS = {
    "us-east5": "gs://marin-us-east5",
    "us-east1": "gs://marin-us-east1",
    "us-central2": "gs://marin-us-central2",
    "us-west4": "gs://marin-us-west4",
    "us-central1": "gs://marin-us-central1",
    "europe-west4": "gs://marin-eu-west4",
}
KEPT_SCORE_COLUMNS = ["doc_id", "fasttext_score", "n_tokens", "pooled_prob", "modernbert_prob"]


@dataclass(frozen=True)
class AsRunWarc:
    warc_path: str
    shard: int  # production's shard; for base-crawl WARCs hash_shard(warc_path, 32000)

    @property
    def warc_hash(self) -> str:
        return warc_hash(self.warc_path)


# One fixture per era, crawl family and TPU type.
FIXTURES = {
    "F1": AsRunWarc(  # CC-MAIN-2014-23, pre-Content-Length-guard; TPU v5e, us-west4
        "s3://commoncrawl/crawl-data/CC-MAIN-2014-23/segments/1405997884573.18/warc/"
        "CC-MAIN-20140722025804-00181-ip-10-33-131-23.ec2.internal.warc.gz",
        5000,
    ),
    "F2": AsRunWarc(  # CC-MAIN-2013-48; TPU v6e, us-east5
        "s3://commoncrawl/crawl-data/CC-MAIN-2013-48/segments/1386165158218/warc/"
        "CC-MAIN-20131204135238-00069-ip-10-33-133-15.ec2.internal.warc.gz",
        31000,
    ),
    "F3": AsRunWarc(  # CC-MAIN-2026-12; TPU v4, us-central2
        "s3://commoncrawl/crawl-data/CC-MAIN-2026-12/segments/1772687277590.81/warc/"
        "CC-MAIN-20260308022436-20260308052436-00349.warc.gz",
        10000,
    ),
    "F4": AsRunWarc(  # CC-NEWS 2017-07; TPU v5p, us-central1
        "s3://commoncrawl/crawl-data/CC-NEWS/2017/07/CC-NEWS-20170728033529-00050.warc.gz",
        34500,
    ),
    "F5": AsRunWarc(  # CC-SUPPLEMENTAL-2026-22, one extractor crash; TPU v4, us-central2
        "s3://commoncrawl/projects/cc-open-athena-test/CC-SUPPLEMENTAL-2026-22/segments/20260526134233/warc/"
        "CC-SUPPLEMENTAL-2026-22-20260526134319-20260526174319-00066.warc.gz",
        32001,
    ),
    "F6": AsRunWarc(  # CC-MAIN-2013-48 through the two-phase path; TPU v6e, us-east5
        "s3://commoncrawl/crawl-data/CC-MAIN-2013-48/segments/1386164021936/warc/"
        "CC-MAIN-20131204133341-00077-ip-10-33-133-15.ec2.internal.warc.gz",
        0,
    ),
}


def base_warc(warc_path: str) -> AsRunWarc:
    """A WARC of the 119 base CC-MAIN crawls, whose production shard follows from its URL."""
    return AsRunWarc(warc_path, hash_shard(warc_path, BASE_SHARDS))


def _namespace(shard: int) -> str:
    return TWO_PHASE_NAMESPACE if shard < TWO_PHASE_SHARDS else FUSED_NAMESPACE


def kept_url(warc: AsRunWarc) -> str:
    return f"{KEPT_BUCKET}/{FUSED_NAMESPACE}/kept/data-{warc.warc_hash}.parquet"


def catalog_url(warc: AsRunWarc) -> str:
    return f"{CATALOG_BUCKET}/{_namespace(warc.shard)}/catalog/shard-{warc.shard:05d}.parquet"


def tombstones_url(warc: AsRunWarc, region: str) -> str:
    return f"{REGION_BUCKETS[region]}/{_namespace(warc.shard)}/tombstones/data-{warc.warc_hash}.jsonl.gz"


@dataclass(frozen=True)
class AsRunOutputs:
    catalog_row: dict  # region, device, n_devices, counters and stage times
    kept: pa.Table  # KEPT_SCORE_COLUMNS, WARC order
    tombstones: list[dict]  # WARC order


def _read_bytes(url: str) -> bytes:
    with fsspec.open(url, "rb") as f:
        return f.read()


def read_catalog_row(warc: AsRunWarc) -> dict:
    rows = [
        r
        for r in pq.read_table(io.BytesIO(_read_bytes(catalog_url(warc)))).to_pylist()
        if r["warc_hash"] == warc.warc_hash
    ]
    if len(rows) != 1:
        raise ValueError(f"{len(rows)} catalog rows for {warc.warc_hash} in {catalog_url(warc)}")
    return rows[0]


def read_tombstone_rows(url: str) -> list[dict]:
    """The rows of a tombstone file; production wrote none when no survivor was dropped."""
    fs, path = fsspec.core.url_to_fs(url)
    if not fs.exists(path):
        return []
    with fs.open(path, "rb") as f:
        return [json.loads(line) for line in gzip.decompress(f.read()).decode("utf-8").splitlines()]


def load_asrun(warc: AsRunWarc) -> AsRunOutputs:
    row = read_catalog_row(warc)
    with fsspec.open(kept_url(warc), "rb") as f:
        kept = pq.read_table(f, columns=KEPT_SCORE_COLUMNS)
    return AsRunOutputs(row, kept, read_tombstone_rows(tombstones_url(warc, row["region"])))


def fetch_asrun(warc: AsRunWarc, dest: str) -> None:
    """Copy the WARC's kept file, tombstones and catalog row to ``dest`` (small objects only)."""
    os.makedirs(dest, exist_ok=True)
    row = read_catalog_row(warc)
    h = warc.warc_hash
    with open(os.path.join(dest, f"kept-{h}.parquet"), "wb") as f:
        f.write(_read_bytes(kept_url(warc)))
    tombstones = tombstones_url(warc, row["region"])
    fs, path = fsspec.core.url_to_fs(tombstones)
    if fs.exists(path):
        with open(os.path.join(dest, f"tombstones-{h}.jsonl.gz"), "wb") as f:
            f.write(_read_bytes(tombstones))
    with open(os.path.join(dest, f"catalog-{h}.json"), "w") as f:
        json.dump(row, f)
