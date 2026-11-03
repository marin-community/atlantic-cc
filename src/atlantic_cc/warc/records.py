# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select the HTML responses of a WARC and decode them (as-run ``_iter_decoded_records``).

Selection, with warcio 1.8.1's ``ArchiveIterator`` at default arguments:

* ``response`` records only, with parsed HTTP headers;
* the HTTP ``Content-Type`` contains ``text/html`` (case-insensitive substring), so
  ``application/xhtml+xml`` is excluded and there is no HTTP status filter;
* the payload is ``record.content_stream().read()``: de-chunked and decompressed by warcio for the
  encodings it supports (gzip, deflate, and br only if the ``brotli`` module is importable).

An exception while handling one record skips that record (it is not counted in ``n_in``). An
exception raised by the iterator itself (a corrupt gzip member) propagates.
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import BinaryIO

import warcio

from atlantic_cc.warc.decode import REPLACEMENT_CHAR, decode_payload
from atlantic_cc.warc.identity import normalize_record_id

logger = logging.getLogger(__name__)

HTML_CONTENT_TYPE = "text/html"
_MAX_LOGGED_ERRORS = 3


@dataclass(frozen=True)
class HtmlRecord:
    doc_id: str  # WARC-Record-ID without "<urn:uuid:" and ">"
    url: str  # WARC-Target-URI, "" when absent
    html: str


@dataclass
class RecordStats:
    """Records dropped inside the loop. Diagnostics only: production logged them and counted nothing."""

    n_errors: int = 0  # the record raised while being handled (e.g. a declared ``charset=hex``)
    n_fffd: int = 0  # decoded text still held U+FFFD; unreachable given the cp1252/latin-1 tail
    n_brotli: int = 0  # selected records with ``Content-Encoding: br`` (left compressed without brotli)


def iter_html_records(stream: BinaryIO, warc_path: str, stats: RecordStats) -> Iterator[HtmlRecord]:
    """Yield the decoded ``text/html`` responses of an uncompressed-or-gzipped WARC stream, in file order."""
    for record in warcio.ArchiveIterator(stream):
        try:
            if record.rec_type != "response":
                continue
            http_headers = record.http_headers
            if http_headers is None:
                continue
            content_type = http_headers.get_header("Content-Type") or ""
            if HTML_CONTENT_TYPE not in content_type.lower():
                continue
            if (http_headers.get_header("Content-Encoding") or "").lower() == "br":
                stats.n_brotli += 1

            payload = record.content_stream().read()
            html = decode_payload(payload, content_type)
            if REPLACEMENT_CHAR in html:
                stats.n_fffd += 1
                continue
            yield HtmlRecord(
                doc_id=normalize_record_id(record.rec_headers.get_header("WARC-Record-ID") or ""),
                url=record.rec_headers.get_header("WARC-Target-URI") or "",
                html=html,
            )
        except Exception as e:  # as-run: one bad record must not cost the WARC
            stats.n_errors += 1
            if stats.n_errors <= _MAX_LOGGED_ERRORS:
                logger.warning("Skipping corrupt record in %s: %s", warc_path, e)
    if stats.n_fffd:
        logger.warning("%s: skipped %d records that still held U+FFFD after decode", warc_path, stats.n_fffd)
    if stats.n_errors:
        logger.warning("Skipped %d corrupt records in %s", stats.n_errors, warc_path)
