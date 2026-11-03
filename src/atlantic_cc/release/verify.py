# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check a copy of the release (local, ``gs://``, ``s3://`` or ``https://``) against its manifest and checksums.

Three checks, each optional:

* ``files``: every file the manifest names is present, and no other part files are;
* ``rows``: per crawl and kind, the row counts in the Parquet footers add up to the manifest's;
* ``md5``: every file's MD5 equals the checksums file. Local files are hashed; for object stores the
  MD5 comes from the listing or a HEAD request (GCS ``md5Hash``, S3/HTTPS ``ETag``), so no data is
  downloaded.
"""

import base64
import logging
from collections import defaultdict
from collections.abc import Collection, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from enum import StrEnum

import fsspec

from atlantic_cc.release import storage
from atlantic_cc.release.checksums import etag_md5, file_md5
from atlantic_cc.release.layout import ReleaseFile, release_files
from atlantic_cc.release.params import Kind

logger = logging.getLogger(__name__)


class Check(StrEnum):
    FILES = "files"
    ROWS = "rows"
    MD5 = "md5"


@dataclass
class VerifyReport:
    root: str
    files_expected: int
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    md5_checked: int = 0
    md5_mismatch: list[str] = field(default_factory=list)
    rows_mismatch: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.missing or self.unexpected or self.md5_mismatch or self.rows_mismatch)

    def to_dict(self) -> dict:
        return {**asdict(self), "ok": self.ok}


def _is_http(root: str) -> bool:
    return root.startswith(("http://", "https://"))


def _metadata_md5(info: dict) -> str | None:
    if "md5Hash" in info:  # GCS, base64 of the raw digest; absent for composite objects
        return base64.b64decode(info["md5Hash"]).hex()
    etag = info.get("ETag") or info.get("etag")
    return etag_md5(etag) if etag else None


def _object_infos(root: str, files: Sequence[ReleaseFile], workers: int) -> dict[str, dict]:
    """Release-relative path -> fsspec info for the present files (a listing per crawl, or HEADs over HTTPS)."""
    fs, base = fsspec.core.url_to_fs(root)
    if _is_http(root):

        def head(f: ReleaseFile) -> tuple[str, dict | None]:
            try:
                return f.path, fs.info(f"{root}/{f.path}")
            except FileNotFoundError:
                return f.path, None

        with ThreadPoolExecutor(workers) as ex:
            return {path: info for path, info in ex.map(head, files) if info is not None}
    crawl_dirs = sorted({f.path.rsplit("/", 1)[0] for f in files})
    infos = {}
    for crawl_dir in crawl_dirs:
        for path, info in fs.find(f"{base}/{crawl_dir}", detail=True).items():
            infos[path.removeprefix(f"{base}/")] = info
    return infos


def verify(
    root: str,
    manifest: dict,
    checksums: dict[str, str] | None,
    crawls: Collection[str] | None,
    kinds: Sequence[Kind],
    checks: Collection[Check],
    workers: int,
) -> VerifyReport:
    files = release_files(manifest, crawls, kinds)
    report = VerifyReport(root=root, files_expected=len(files))
    infos = _object_infos(root, files, workers)
    expected = {f.path for f in files}
    report.missing = sorted(expected - infos.keys())
    present = [f for f in files if f.path in infos]
    if Check.FILES in checks and not _is_http(root):
        selected_dirs = {f.path.rsplit("/", 1)[0] for f in files}
        prefixes = tuple(str(k) for k in kinds)
        report.unexpected = sorted(
            p
            for p in infos
            if p.endswith(".parquet")
            and p.rsplit("/", 1)[0] in selected_dirs
            and p.rsplit("/", 1)[1].startswith(prefixes)
            and p not in expected
        )
    if Check.MD5 in checks:
        if checksums is None:
            raise ValueError("the md5 check needs a checksums file")
        local = not storage.has_protocol(root)

        def md5_of(f: ReleaseFile) -> tuple[str, str | None]:
            return f.path, file_md5(f"{root}/{f.path}") if local else _metadata_md5(infos[f.path])

        with ThreadPoolExecutor(workers) as ex:
            got = dict(ex.map(md5_of, present))
        unknown = sorted(p for p, md5 in got.items() if md5 is None)
        if unknown:
            raise ValueError(f"{len(unknown)} objects expose no MD5, e.g. {unknown[:3]}")
        report.md5_checked = len(got)
        report.md5_mismatch = sorted(p for p, md5 in got.items() if checksums.get(p) != md5)
    if Check.ROWS in checks:
        with ThreadPoolExecutor(workers) as ex:
            rows = list(ex.map(lambda f: storage.parquet_rows(f"{root}/{f.path}"), present))
        found: dict[tuple[str, str], int] = defaultdict(int)
        for f, n in zip(present, rows, strict=True):
            found[(f.crawl, str(f.kind))] += n
        for crawl in sorted({f.crawl for f in files}):
            for kind in kinds:
                want = manifest["crawls"][crawl][kind]["records"]
                if found[(crawl, str(kind))] != want:
                    report.rows_mismatch.append(
                        {"crawl": crawl, "kind": str(kind), "manifest": want, "found": found[(crawl, str(kind))]}
                    )
    logger.info(
        "verify %s: %s",
        root,
        {k: v if isinstance(v, int) else len(v) for k, v in asdict(report).items() if k != "root"},
    )
    return report
