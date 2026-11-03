# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hand-built WARC records (one gzip member each, like Common Crawl) for offline tests."""

import gzip


def http_block(
    payload: bytes, headers: tuple[tuple[str, str], ...] = (("Content-Type", "text/html"),), status="200 OK"
):
    head = f"HTTP/1.1 {status}\r\n" + "".join(f"{k}: {v}\r\n" for k, v in headers) + "\r\n"
    return head.encode() + payload


def warc_record(rec_type: str, uri: str | None, block: bytes, record_id: str, extra=()) -> bytes:
    headers = [("WARC-Type", rec_type), ("WARC-Record-ID", record_id)]
    if uri is not None:
        headers.append(("WARC-Target-URI", uri))
    headers += [*extra, ("Content-Length", str(len(block)))]
    head = "WARC/1.0\r\n" + "".join(f"{k}: {v}\r\n" for k, v in headers) + "\r\n"
    return gzip.compress(head.encode() + block + b"\r\n\r\n")


def html_response(doc_id: str, html: bytes, url: str | None = None, content_type: str = "text/html") -> bytes:
    """A ``response`` record whose WARC-Record-ID normalizes to ``doc_id``."""
    return warc_record(
        "response",
        url or f"http://example.com/{doc_id}",
        http_block(html, (("Content-Type", content_type),)),
        f"<urn:uuid:{doc_id}>",
    )
