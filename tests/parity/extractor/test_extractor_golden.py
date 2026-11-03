# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The installed ``resiliparse._extract_rs`` reproduces the as-run extractor on every HTML page of F1.

``golden_f1_asrun.tsv.gz`` holds, for each of F1's 13,516 HTML pages (as ``pages.py`` decodes them), the
page's HTML sha256 prefix and the as-run binaries' result: ``ok`` with the text's sha256 prefix, or
``crash``. It was written by ``run_parity.py --write-golden`` on Linux x86_64 (docs/extractor.md). The test
downloads F1 (222 MB) from data.commoncrawl.org and needs about 1.2 GB of scratch space.
"""

import gzip
import os
import sys
from pathlib import Path

import pytest

from tests.parity.extractor.pages import FIXTURES, download, write_pages
from tests.parity.extractor.run_parity import DEFAULT_PAGE_TIMEOUT, GOLDEN_HASH_CHARS, Variant, run_variants

GOLDEN = Path(__file__).with_name("golden_f1_asrun.tsv.gz")

pytestmark = [pytest.mark.network, pytest.mark.extractor, pytest.mark.slow]


def read_golden() -> list[tuple[str, str, str]]:
    with gzip.open(GOLDEN, "rt") as f:
        return [tuple(line.rstrip("\n").split("\t")[1:]) for line in f]


def test_f1_matches_asrun_extractor(tmp_path: Path) -> None:
    extension = pytest.importorskip("resiliparse._extract_rs")
    # Workers run with ``python -I``, so hand them the directory this process imported the extension from
    # (a site-packages directory, or production's unpacked layout found through PYTHONPATH).
    package_dir = str(Path(extension.__file__).parents[1])
    fixture = FIXTURES["F1"]
    warc = download(fixture, tmp_path / f"{fixture.warc_hash}.warc.gz")
    index = write_pages(warc, tmp_path / "pages.bin", tmp_path / "index.json")
    golden = read_golden()
    assert [entry.html_sha256[:GOLDEN_HASH_CHARS] for entry in index] == [row[0] for row in golden]

    variant = Variant("installed", sys.executable, package_dir)
    records = run_variants(
        [variant],
        "F1",
        tmp_path / "pages.bin",
        tmp_path / "index.json",
        len(index),
        tmp_path,
        os.cpu_count() or 1,
        DEFAULT_PAGE_TIMEOUT,
    )["installed"].records
    got = [
        (
            entry.html_sha256[:GOLDEN_HASH_CHARS],
            records[i]["status"],
            records[i].get("sha256", "")[:GOLDEN_HASH_CHARS] or records[i].get("error", ""),
        )
        for i, entry in enumerate(index)
    ]
    mismatched = [i for i, (have, want) in enumerate(zip(got, golden, strict=True)) if have != want]
    assert not mismatched, f"{len(mismatched)}/{len(index)} pages differ from the as-run extractor: {mismatched[:20]}"
