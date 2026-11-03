# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""U-REL: the checksums file built from upload done-markers."""

import hashlib
import json

import pytest

from atlantic_cc.release import checksums
from atlantic_cc.release.checksums import UploadRecord

MD5_A = "6d2a3b0f3d71576e33f384507ad107c9"
MD5_B = "70ef1582aed844fa159ff3452adfdfc0"
PART = "projects/atlantic-cc/v1.0/crawl=CC-NEWS-2016-08/contaminated-00000-of-00001.parquet"
MANIFEST = "projects/atlantic-cc/v1.0.manifest.json"


def test_single_part_etag_is_the_md5():
    assert checksums.etag_md5(f'"{MD5_A}"') == MD5_A


@pytest.mark.parametrize("etag", [f'"{MD5_A}-2"', '"not-hex"', f'"{MD5_A.upper()}"'])
def test_multipart_or_malformed_etags_are_rejected(etag):
    with pytest.raises(ValueError, match="not the MD5 of a single-part upload"):
        checksums.etag_md5(etag)


def test_checksums_file_is_sorted_md5sum_lines_relative_to_the_release_root():
    md5s = checksums.checksums_from_uploads(
        [UploadRecord(PART, 145_767, f'"{MD5_A}"'), UploadRecord(MANIFEST, 37_690, f'"{MD5_B}"')]
    )
    text = checksums.format_checksums(md5s)
    assert text == (
        f"{MD5_B}  v1.0.manifest.json\n{MD5_A}  v1.0/crawl=CC-NEWS-2016-08/contaminated-00000-of-00001.parquet\n"
    )
    assert checksums.parse_checksums(text) == md5s


def test_parse_accepts_binary_mode_lines_and_rejects_garbage():
    assert checksums.parse_checksums(f"{MD5_A} *v1.0.manifest.json\n") == {"v1.0.manifest.json": MD5_A}
    with pytest.raises(ValueError, match="not an md5sum line"):
        checksums.parse_checksums(f"{MD5_A} v1.0.manifest.json\n")


def _marker(root, key, etag):
    path = root / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"key": key, "bytes": 10, "etag": etag}))


def test_done_markers_are_read_from_their_key_paths(tmp_path):
    _marker(tmp_path, PART, f'"{MD5_A}"')
    _marker(tmp_path, MANIFEST, f'"{MD5_B}"')
    records = checksums.read_done_markers(str(tmp_path), concurrency=4)
    assert sorted(r.key for r in records) == [MANIFEST, PART]
    assert checksums.checksums_from_uploads(records)["v1.0.manifest.json"] == MD5_B


def test_file_md5_streams_the_file(tmp_path):
    data = bytes(range(256)) * 50_000
    (tmp_path / "f").write_bytes(data)
    assert checksums.file_md5(str(tmp_path / "f")) == hashlib.md5(data).hexdigest()
