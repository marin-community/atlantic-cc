# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn a Common Crawl WARC into a pages file that extractor parity runs read.

Record selection follows production: ``response`` records with HTTP headers whose ``Content-Type``
contains ``text/html``. Decoding only has to be deterministic, because every extractor under test gets
the same string: the declared HTTP charset when Python knows it as a text codec, else strict UTF-8, else
cp1252 with replacement characters. It is not production's charset cascade (``atlantic_cc.warc``).

A pages file is the concatenation of each page's UTF-8 bytes (``surrogatepass``, so any ``str`` round
trips); the index is a JSON list with one ``{"record_id", "offset", "length", "html_sha256"}`` per page,
in WARC order.
"""

import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import requests
from warcio.archiveiterator import ArchiveIterator

CC_HTTPS_PREFIX = "https://data.commoncrawl.org/"
CHARSET_RE = re.compile(r"charset\s*=\s*[\"']?([a-zA-Z0-9_\-:.]+)", re.IGNORECASE)
DOWNLOAD_CHUNK = 1 << 20
DOWNLOAD_TIMEOUT = 600
FALLBACK_ENCODING = "cp1252"
PAGE_ENCODING = "utf-8"
PAGE_ERRORS = "surrogatepass"


@dataclass(frozen=True)
class Fixture:
    name: str
    warc_hash: str
    path: str  # relative to the Common Crawl bucket root
    size: int

    @property
    def url(self) -> str:
        return CC_HTTPS_PREFIX + self.path


# Golden fixtures F1 and F3 (an early 2014 crawl and a 2026 crawl).
FIXTURES = {
    "F1": Fixture(
        "F1",
        "3302ab887ad0",
        "crawl-data/CC-MAIN-2014-23/segments/1405997884573.18/warc/"
        "CC-MAIN-20140722025804-00181-ip-10-33-131-23.ec2.internal.warc.gz",
        221_920_839,
    ),
    "F3": Fixture(
        "F3",
        "4ff1f9108b5a",
        "crawl-data/CC-MAIN-2026-12/segments/1772687277590.81/warc/CC-MAIN-20260308022436-20260308052436-00349.warc.gz",
        845_424_740,
    ),
}


@dataclass(frozen=True)
class PageEntry:
    record_id: str
    offset: int
    length: int
    html_sha256: str


def download(fixture: Fixture, dest: Path) -> Path:
    """Download a fixture WARC once; a complete file of the expected size is reused."""
    if dest.exists() and dest.stat().st_size == fixture.size:
        return dest
    partial = dest.with_suffix(dest.suffix + ".partial")
    with requests.get(fixture.url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(partial, "wb") as f:
            for block in response.iter_content(DOWNLOAD_CHUNK):
                f.write(block)
    if partial.stat().st_size != fixture.size:
        raise RuntimeError(f"{fixture.url}: got {partial.stat().st_size} bytes, expected {fixture.size}")
    partial.rename(dest)
    return dest


def decode_html(payload: bytes, content_type: str) -> str:
    """Declared charset (strict), else UTF-8 (strict), else cp1252 with replacement characters."""
    match = CHARSET_RE.search(content_type)
    for codec in ([match.group(1)] if match else []) + ["utf-8"]:
        try:
            return payload.decode(codec)
        except (LookupError, UnicodeDecodeError, ValueError):  # unknown or bytes-to-bytes codec, bad bytes
            continue
    return payload.decode(FALLBACK_ENCODING, errors="replace")


def iter_html_pages(warc: Path) -> Iterator[tuple[str, str]]:
    """``(record_id, html)`` for every HTML response record, in WARC order."""
    with open(warc, "rb") as f:
        for record in ArchiveIterator(f):
            if record.rec_type != "response" or record.http_headers is None:
                continue
            content_type = record.http_headers.get_header("Content-Type") or ""
            if "text/html" not in content_type.lower():
                continue
            payload = record.content_stream().read()
            yield record.rec_headers.get_header("WARC-Record-ID"), decode_html(payload, content_type)


def write_pages(warc: Path, pages_path: Path, index_path: Path) -> list[PageEntry]:
    entries: list[PageEntry] = []
    offset = 0
    with open(pages_path, "wb") as out:
        for record_id, html in iter_html_pages(warc):
            data = html.encode(PAGE_ENCODING, PAGE_ERRORS)
            out.write(data)
            entries.append(PageEntry(record_id, offset, len(data), hashlib.sha256(data).hexdigest()))
            offset += len(data)
    index_path.write_text(json.dumps([asdict(entry) for entry in entries]))
    return entries


def read_index(index_path: Path) -> list[PageEntry]:
    return [PageEntry(**entry) for entry in json.loads(index_path.read_text())]
