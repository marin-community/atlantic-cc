# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""G5: the published v1.0 release (manifest, paths files, schema, checksums, CC-NEWS-2016-08).

Offline tests compare the code's outputs with facts recorded from the published files; the ``network``
test downloads CC-NEWS-2016-08 (50 MB) from data.commoncrawl.org with the download tool and checks it.
"""

import gzip
import hashlib
import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from atlantic_cc.release import checksums, download, regroup, verify
from atlantic_cc.release.layout import release_files
from atlantic_cc.release.params import (
    HTTPS_ROOT,
    RELEASE_SCHEMA,
    V1_0_CRAWLS,
    V1_0_FILES_BY_KIND,
    V1_0_PATHS_GZIP_MTIME,
    V1_0_RECORDS,
    V1_0_RECORDS_BY_KIND,
    VERSION,
    Kind,
)
from atlantic_cc.warc.identity import warc_hash
from atlantic_cc.worklist.identity import to_s3_url

GOLDEN = Path(__file__).parents[2] / "golden" / "release"


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads((GOLDEN / "v1.0.manifest.json").read_text())


@pytest.fixture(scope="module")
def facts() -> dict:
    return json.loads((GOLDEN / "v1.0.published.json").read_text())


def _plan(manifest: dict) -> list[dict]:
    """The part plan's naming fields, recovered from the manifest's file counts."""
    plan = []
    for f in release_files(manifest, None, list(Kind)):
        part, of = f.path.rsplit("/", 1)[1].removesuffix(".parquet").split("-")[1::2]
        plan.append({"crawl": f.crawl, "kind": str(f.kind), "part": int(part), "of": int(of)})
    return plan


def test_manifest_totals_add_up(manifest):
    crawls = manifest["crawls"].values()
    for kind in Kind:
        assert sum(c[kind]["records"] for c in crawls) == manifest["by_kind"][kind] == V1_0_RECORDS_BY_KIND[kind]
        assert sum(c[kind]["files"] for c in crawls) == V1_0_FILES_BY_KIND[kind]
    assert manifest["records"] == V1_0_RECORDS
    assert len(manifest["crawls"]) == V1_0_CRAWLS


def test_paths_files_regenerate_from_the_manifest(manifest, facts):
    for name, data in regroup.paths_files(_plan(manifest), VERSION, V1_0_PATHS_GZIP_MTIME).items():
        published = facts["paths_files"][name]
        content = gzip.decompress(data)
        assert hashlib.sha256(content).hexdigest() == published["content_sha256"]
        assert content.decode().split()[:3] == published["head"]
        assert int.from_bytes(data[4:8], "little") == published["gzip_mtime"]
        assert hashlib.sha256(data).hexdigest() == published["sha256"]  # zlib 1.2.x deflate, as published


def test_manifest_document_is_serialized_like_the_published_one(manifest):
    assert regroup.manifest_bytes(manifest) == (GOLDEN / "v1.0.manifest.json").read_bytes()


def test_release_schema_matches_the_published_parquet_schema(facts):
    row = dict.fromkeys(RELEASE_SCHEMA.names)
    row["decon_matches"] = []
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pylist([row], schema=RELEASE_SCHEMA), buf, compression="zstd")
    written = pq.ParquetFile(io.BytesIO(buf.getvalue()))
    assert str(written.schema).split("\n", 1)[1] == facts["schema"]["parquet"]
    # Read back, the list child is called "element" (Parquet's compliant nested naming).
    assert str(written.schema_arrow) == facts["schema"]["arrow"]
    assert written.metadata.created_by == facts["schema"]["created_by"]


def test_checksums_cover_the_published_manifest(facts):
    head = checksums.parse_checksums("\n".join(facts["checksums"]["head"]))
    manifest_md5 = hashlib.md5((GOLDEN / "v1.0.manifest.json").read_bytes()).hexdigest()
    assert head[f"{VERSION}.manifest.json"] == manifest_md5 == facts["manifest"]["md5"]


def test_cc_news_2016_08_records_are_grouped_by_warc_in_warc_hash_order(facts):
    runs = facts["cc_news_2016_08"]["part-00000-of-00001"]["warc_runs"]
    hashes = [warc_hash(to_s3_url(path)) for path, _, _ in runs]
    assert hashes == [h for _, _, h in runs] == sorted(hashes)
    assert sum(n for _, n, _ in runs) == facts["cc_news_2016_08"]["part-00000-of-00001"]["rows"]


@pytest.mark.network
def test_download_and_verify_cc_news_2016_08(facts, tmp_path):
    summary = download.download(HTTPS_ROOT, str(tmp_path), ["CC-NEWS-2016-08"], list(Kind), 2, None, VERSION)
    assert summary["files"] == 2
    manifest = json.loads((tmp_path / f"{VERSION}.manifest.json").read_text())
    md5s = {f"{VERSION}/crawl=CC-NEWS-2016-08/{name}.parquet": f["md5"] for name, f in facts["cc_news_2016_08"].items()}
    report = verify.verify(str(tmp_path), manifest, md5s, ["CC-NEWS-2016-08"], list(Kind), list(verify.Check), 2)
    assert report.ok, report.to_dict()
    part = facts["cc_news_2016_08"]["part-00000-of-00001"]
    table = pq.ParquetFile(tmp_path / VERSION / "crawl=CC-NEWS-2016-08" / "part-00000-of-00001.parquet").read()
    ids = table.column("record_id").to_pylist()
    assert hashlib.sha256("\n".join(ids).encode()).hexdigest() == part["record_ids_sha256"]
    contaminated = pq.ParquetFile(tmp_path / VERSION / "crawl=CC-NEWS-2016-08" / "contaminated-00000-of-00001.parquet")
    matches = {r["record_id"]: r["decon_matches"] for r in contaminated.read().to_pylist()}
    assert matches == {
        r["record_id"]: r["decon_matches"] for r in facts["cc_news_2016_08"]["contaminated-00000-of-00001"]["records"]
    }
