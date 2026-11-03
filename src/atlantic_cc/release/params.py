# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Release layout, schema and packaging constants of Atlantic-CC v1.0.

Layout under the release root (``s3://commoncrawl/projects/atlantic-cc/``)::

    v1.0/crawl=<crawl>/part-NNNNN-of-MMMMM.parquet          decon_matches == []
    v1.0/crawl=<crawl>/contaminated-NNNNN-of-MMMMM.parquet  decon_matches != []
    v1.0.manifest.json            per-crawl record and file counts
    v1.0.parquet.paths.gz         every data file, as bucket keys
    v1.0.clean.parquet.paths.gz   part-* files only
    v1.0.checksums.md5            md5sum of every published file

The manifest and paths files sit next to ``v1.0/`` so that hive readers of ``v1.0/`` see only
``crawl=`` partitions. Part boundaries depend on the byte sizes of the intermediate pieces, which
depend on the pyarrow version (23.0.1) and its default zstd level; the record content and order do
not.
"""

from enum import StrEnum

import pyarrow as pa

from atlantic_cc.spec import CC_HTTPS_PREFIX, CC_S3_PREFIX, DATASET_VERSION

VERSION = DATASET_VERSION
# Where the release lives in the Common Crawl bucket; paths files list keys under it. This is the only
# place in the code that names the release location: if Common Crawl moves the release, change it
# here (and its two mentions in README.md and docs/release_format.md).
PUBLISHED_PREFIX = "projects/atlantic-cc"
HTTPS_ROOT = f"{CC_HTTPS_PREFIX}{PUBLISHED_PREFIX}"
S3_ROOT = f"{CC_S3_PREFIX}{PUBLISHED_PREFIX}"

# --- regroup ---------------------------------------------------------------------------------------
FILES_PER_TASK = 50  # dedup output files per pieces batch
PART_MB = 1000  # Common Crawl asked for ~1 GB files; measured on piece bytes, so parts are ~0.8 GB
ZSTD_LEVEL = 9  # parts; pieces use pyarrow's default zstd level
WRITE_BATCH_ROWS = 10_000  # rows per piece write and per part row group
LOOKUP_ROW_GROUP_ROWS = 20_000
READ_THREADS = 64

DECON_FIELD = "decon_matches"
WARC_PATH_FIELD = "warc_path"
RELEASE_SCHEMA = pa.schema(
    [
        ("record_id", pa.string()),
        ("url", pa.string()),
        ("crawl", pa.string()),
        (WARC_PATH_FIELD, pa.string()),
        ("text", pa.string()),
        ("fasttext_score", pa.float64()),
        ("pooled_prob", pa.float64()),
        ("modernbert_prob", pa.float64()),
        ("exact_copies", pa.int64()),
        ("fuzzy_cluster_size", pa.int64()),
        (DECON_FIELD, pa.list_(pa.string())),
    ]
)
# Release field -> dedup-output field, for the fields copied straight from the record.
COLUMNS = {
    "record_id": "doc_id",
    "url": "url",
    "crawl": "snapshot",
    "text": "text",
    "fasttext_score": "fasttext_score",
    "pooled_prob": "pooled_prob",
    "modernbert_prob": "modernbert_prob",
    "exact_copies": "exact_copies",
    "fuzzy_cluster_size": "fuzzy_cluster_size",
}


class Kind(StrEnum):
    """Release file kind; the value is the file name prefix."""

    CLEAN = "part"
    CONTAMINATED = "contaminated"


# --- stage and output names ----------------------------------------------------------------------
WARC_PATHS_FILE = "warc_paths.parquet"
DECON_LOOKUP_FILE = "decon_lookup.parquet"
PARTS_PLAN_FILE = "parts_plan.json"
PIECES_DIR = "pieces"
DONE_DIR = "_done"
REMOVED_DOCS_FILE = "removed_docs.parquet"  # in the decon run
DECON_EXPORT_FILE = "export.json"  # in the decon run


def manifest_name(version: str) -> str:
    return f"{version}.manifest.json"


def paths_name(version: str) -> str:
    return f"{version}.parquet.paths.gz"


def clean_paths_name(version: str) -> str:
    return f"{version}.clean.parquet.paths.gz"


def checksums_name(version: str) -> str:
    return f"{version}.checksums.md5"


def part_file_name(kind: str, part: int, of: int) -> str:
    return f"{kind}-{part:05d}-of-{of:05d}.parquet"


# --- the published v1.0 ------------------------------------------------------------------------------
V1_0_RECORDS = 9_951_767_085
V1_0_RECORDS_BY_KIND = {Kind.CLEAN: 9_950_417_677, Kind.CONTAMINATED: 1_349_408}
V1_0_FILES = 27_985
V1_0_FILES_BY_KIND = {Kind.CLEAN: 27_736, Kind.CONTAMINATED: 249}
V1_0_CRAWLS = 249
# The as-run manifest step gzipped the paths files with the current time; this reproduces their bytes.
V1_0_PATHS_GZIP_MTIME = 1_793_254_392
