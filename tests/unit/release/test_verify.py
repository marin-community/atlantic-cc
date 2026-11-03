# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""U-REL: checking a local copy of a release against its manifest and checksums."""

import json
import os
import shutil

import pytest

from atlantic_cc.release import checksums
from atlantic_cc.release.params import VERSION, Kind
from atlantic_cc.release.verify import Check, verify

ALL_CHECKS = list(Check)


@pytest.fixture
def copy(built, tmp_path) -> tuple[str, dict, dict]:
    """A private copy of the synthetic release, its manifest and its checksums."""
    root = tmp_path / "release"
    shutil.copytree(built.out, root)
    manifest = json.loads((root / f"{VERSION}.manifest.json").read_text())
    md5s = {}
    for dirpath, _, names in os.walk(root / VERSION):
        for name in names:
            path = os.path.join(dirpath, name)
            md5s[os.path.relpath(path, root)] = checksums.file_md5(path)
    return str(root), manifest, md5s


def test_an_intact_copy_passes_every_check(copy):
    root, manifest, md5s = copy
    report = verify(root, manifest, md5s, None, list(Kind), ALL_CHECKS, workers=2)
    assert report.ok
    assert report.files_expected == report.md5_checked == 5


def test_missing_corrupt_and_stray_files_are_reported(copy):
    root, manifest, md5s = copy
    os.remove(f"{root}/{VERSION}/crawl=CC-MAIN-2026-17/part-00000-of-00001.parquet")
    with open(f"{root}/{VERSION}/crawl=CC-NEWS-2016-08/part-00000-of-00001.parquet", "r+b") as f:
        f.seek(100)
        f.write(b"\x00\x01")
    shutil.copy(
        f"{root}/{VERSION}/crawl=CC-NEWS-2016-08/part-00000-of-00001.parquet",
        f"{root}/{VERSION}/crawl=CC-NEWS-2016-08/part-00001-of-00002.parquet",
    )
    report = verify(root, manifest, md5s, None, list(Kind), [Check.FILES, Check.MD5], workers=2)
    assert report.missing == [f"{VERSION}/crawl=CC-MAIN-2026-17/part-00000-of-00001.parquet"]
    assert report.md5_mismatch == [f"{VERSION}/crawl=CC-NEWS-2016-08/part-00000-of-00001.parquet"]
    assert report.unexpected == [f"{VERSION}/crawl=CC-NEWS-2016-08/part-00001-of-00002.parquet"]
    assert not report.ok


def test_row_counts_are_checked_against_the_manifest(copy):
    root, manifest, md5s = copy
    manifest["crawls"]["CC-MAIN-2013-20"]["contaminated"]["records"] = 5
    report = verify(root, manifest, md5s, ["CC-MAIN-2013-20"], [Kind.CONTAMINATED], [Check.ROWS], workers=2)
    assert report.rows_mismatch == [{"crawl": "CC-MAIN-2013-20", "kind": "contaminated", "manifest": 5, "found": 1}]
    assert report.files_expected == 1


def test_unknown_crawls_are_an_error(copy):
    root, manifest, md5s = copy
    with pytest.raises(ValueError, match="no such crawls"):
        verify(root, manifest, md5s, ["CC-MAIN-1999-01"], list(Kind), ALL_CHECKS, workers=2)
