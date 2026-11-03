# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""U-SEL: which WARC records reach extraction, and the identity of WARCs and records."""

import gzip
import io

import pytest

from atlantic_cc.warc.identity import normalize_record_id, s3_to_https, snapshot_of, warc_hash
from atlantic_cc.warc.records import RecordStats, iter_html_records
from tests.unit.warc.warc_builder import http_block, warc_record

# brotli.compress(b"<html><body>br</body></html>")
BROTLI_PAGE = b"\x1b\x1b\x00\xf8\x1d\xa9Q\x9f=.y\xc4\xd0J\x8e.}\xd8\xda\x8a\x16i\n\x13`\xf0\x93\x13\x01"
F1_WARC = (
    "s3://commoncrawl/crawl-data/CC-MAIN-2014-23/segments/1405997884573.18/warc/"
    "CC-MAIN-20140722025804-00181-ip-10-33-131-23.ec2.internal.warc.gz"
)


def _records(*blobs: bytes) -> tuple[list, RecordStats]:
    stats = RecordStats()
    return list(iter_html_records(io.BytesIO(b"".join(blobs)), "test.warc.gz", stats)), stats


def test_only_html_responses_with_http_headers_are_selected():
    html = b"<html><body>x</body></html>"
    records, stats = _records(
        warc_record("warcinfo", None, b"software: test\r\n", "<urn:uuid:info>"),
        warc_record("request", "http://a/", b"GET / HTTP/1.1\r\nHost: a\r\n\r\n", "<urn:uuid:req>"),
        warc_record("response", "http://a/", http_block(html), "<urn:uuid:html>"),
        warc_record(
            "response",
            "http://b/",
            http_block(b"<x/>", (("Content-Type", "application/xhtml+xml"),)),
            "<urn:uuid:xhtml>",
        ),
        warc_record("response", "http://c/", http_block(b"<p>no type</p>", ()), "<urn:uuid:notype>"),
        warc_record(
            "response", "http://d/", http_block(b"{}", (("Content-Type", "application/json"),)), "<urn:uuid:json>"
        ),
        warc_record("response", "dns:a", b"20260101 a. 300 IN A 1.2.3.4", "<urn:uuid:dns>"),
        warc_record("response", "http://e/", b"", "<urn:uuid:empty>"),
        warc_record("resource", "http://f/", html, "<urn:uuid:resource>", extra=[("Content-Type", "text/html")]),
        warc_record("revisit", "http://g/", http_block(b""), "<urn:uuid:revisit>"),
        warc_record(
            "response",
            "http://h/",
            http_block(b"<p>gone</p>", (("content-type", "TEXT/HTML; Charset=UTF-8"),), "404 Not Found"),
            "<urn:uuid:404>",
        ),
        warc_record(
            "response",
            "http://i/",
            http_block(b"<p>x</p>", (("Content-Type", "application/x-text/html-ish"),)),
            "<urn:uuid:substr>",
        ),
    )
    # No status filter (404 kept), case-insensitive substring match (so the odd last type matches),
    # xhtml/json/no Content-Type excluded, non-HTTP and zero-length responses have no HTTP headers.
    assert [r.doc_id for r in records] == ["html", "404", "substr"]
    assert stats == RecordStats()


def test_payload_decoding_follows_warcio_1_8_1():
    gz = gzip.compress(b"<html><body>gz</body></html>")
    records, stats = _records(
        warc_record(
            "response",
            "http://a/",
            http_block(gz, (("Content-Type", "text/html"), ("Content-Encoding", "GZIP"))),
            "<urn:uuid:gzip>",
        ),
        warc_record(
            "response",
            "http://b/",
            http_block(gz, (("Content-Type", "text/html"), ("Content-Encoding", "x-gzip"))),
            "<urn:uuid:xgzip>",
        ),
        warc_record(
            "response",
            "http://c/",
            http_block(b"5\r\nhello\r\n0\r\n\r\n", (("Content-Type", "text/html"), ("Transfer-Encoding", "chunked"))),
            "<urn:uuid:chunked>",
        ),
        warc_record(
            "response",
            "http://d/",
            http_block(BROTLI_PAGE, (("Content-Type", "text/html"), ("Content-Encoding", "br"))),
            "<urn:uuid:br>",
        ),
    )
    by_id = {r.doc_id: r.html for r in records}
    assert by_id["gzip"] == "<html><body>gz</body></html>"
    # warcio 1.8.1 does not know the x-gzip alias (1.8.2 does): the payload stays compressed and
    # decodes as cp1252 mojibake.
    assert by_id["xgzip"] == gz.decode("cp1252")
    assert by_id["chunked"] == "hello"
    # br is decompressed only when the brotli module is importable; the pinned environment has none.
    assert "<body>br</body>" not in by_id["br"]
    assert stats.n_brotli == 1


def test_a_failing_record_is_skipped_and_iteration_continues():
    records, stats = _records(
        warc_record(
            "response", "http://a/", http_block(b"abc", (("Content-Type", "text/html; charset=hex"),)), "<urn:uuid:hex>"
        ),
        warc_record("response", "http://b/", http_block(b"<p>ok</p>"), "<urn:uuid:ok>"),
    )
    assert [r.doc_id for r in records] == ["ok"]
    assert stats.n_errors == 1


def test_literal_replacement_character_page_is_kept_without_it():
    """The record is not skipped: the cascade never returns U+FFFD (see test_u_dec_charset)."""
    records, stats = _records(
        warc_record("response", "http://a/", http_block("x \ufffd y".encode()), "<urn:uuid:fffd>")
    )
    assert [r.html for r in records] == ["x 魹ｽ y"]
    assert stats.n_fffd == 0


def test_iteration_errors_propagate():
    """A record warcio cannot parse aborts the WARC (production retried and finally abandoned it)."""
    good = warc_record("response", "http://a/", http_block(b"<p>a</p>"), "<urn:uuid:a>")
    no_uri = warc_record("response", None, http_block(b"<p>b</p>"), "<urn:uuid:b>")
    with pytest.raises(AttributeError):
        _records(good, no_uri)


def test_record_fields():
    records, _ = _records(
        warc_record(
            "response",
            "http://example.com/ü?q=1",
            http_block(b"<p>a</p>"),
            "<urn:uuid:0e6b7a52-3f43-4a3e-9e0b-0d8c3f1c2a11>",
        )
    )
    (r,) = records
    assert (r.doc_id, r.url) == ("0e6b7a52-3f43-4a3e-9e0b-0d8c3f1c2a11", "http://example.com/ü?q=1")


@pytest.mark.parametrize(
    ("record_id", "doc_id"),
    [
        ("<urn:uuid:3f1c2a11-0000-4000-8000-000000000001>", "3f1c2a11-0000-4000-8000-000000000001"),
        ("  <urn:uuid:abc>\t", "abc"),
        ("urn:uuid:abc", "abc"),
        ("<abc>", "abc"),
        ("abc", "abc"),
        ("<urn:uuid:abc", "<urn:uuid:abc"),
        ("", ""),
    ],
)
def test_normalize_record_id(record_id, doc_id):
    assert normalize_record_id(record_id) == doc_id


@pytest.mark.parametrize(
    ("warc_path", "snapshot"),
    [
        (F1_WARC, "CC-MAIN-2014-23"),
        (
            "s3://commoncrawl/projects/cc-open-athena-test/CC-SUPPLEMENTAL-2026-22/segments/20260526134233/warc/"
            "CC-SUPPLEMENTAL-2026-22-20260526134319-20260526174319-00066.warc.gz",
            "CC-SUPPLEMENTAL-2026-22",
        ),
        ("s3://commoncrawl/crawl-data/CC-NEWS/2017/07/CC-NEWS-20170728033529-00050.warc.gz", "CC-NEWS-2017-07"),
        ("s3://commoncrawl/crawl-data/CC-NEWS/2025/12/CC-MAIN-2025-51-lookalike.warc.gz", "CC-NEWS-2025-12"),
        ("s3://commoncrawl/crawl-data/CC-WET-2020-05/x.warc.gz", ""),
    ],
)
def test_snapshot_names_every_crawl_family(warc_path, snapshot):
    assert snapshot_of(warc_path) == snapshot


def test_warc_hash_is_sha256_of_the_full_s3_path():
    assert warc_hash(F1_WARC) == "3302ab887ad0"
    assert (
        warc_hash("s3://commoncrawl/crawl-data/CC-NEWS/2017/07/CC-NEWS-20170728033529-00050.warc.gz") == "4e28437dae88"
    )


@pytest.mark.parametrize(
    ("path", "url"),
    [
        ("s3://commoncrawl/crawl-data/a.warc.gz", "https://data.commoncrawl.org/crawl-data/a.warc.gz"),
        ("https://data.commoncrawl.org/crawl-data/a.warc.gz", "https://data.commoncrawl.org/crawl-data/a.warc.gz"),
        ("crawl-data/a.warc.gz", "https://data.commoncrawl.org/crawl-data/a.warc.gz"),
    ],
)
def test_s3_to_https(path, url):
    assert s3_to_https(path) == url
