# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""WARC download: retries, the Content-Length guard, the deadline, and the explicit failure."""

import pytest
import requests

from atlantic_cc.warc import download
from atlantic_cc.warc.download import DownloadError, check_content_length, download_warc

WARC_PATH = "s3://commoncrawl/crawl-data/CC-MAIN-2020-05/segments/x/warc/CC-MAIN-x.warc.gz"
BODY = b"0123456789" * 1000


class _Response:
    def __init__(self, body: bytes = BODY, status: int = 200, headers: dict | None = None):
        self.status_code = status
        self.headers = {"Content-Length": str(len(body))} if headers is None else headers
        self._body = body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} error")

    def iter_content(self, chunk_size):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    calls: list[float] = []
    monkeypatch.setattr(download.time, "sleep", calls.append)
    return calls


def _serve(monkeypatch, *responses: _Response) -> list[str]:
    """Serve ``responses`` in order (the last one repeats) and record the requested URLs."""
    urls: list[str] = []

    def get(url, **kwargs):
        assert kwargs == {"stream": True, "timeout": download.HTTP_TIMEOUT}
        urls.append(url)
        return responses[min(len(urls), len(responses)) - 1]

    monkeypatch.setattr(download.requests, "get", get)
    return urls


def test_complete_body_is_returned(monkeypatch, sleeps):
    urls = _serve(monkeypatch, _Response())
    assert download_warc(WARC_PATH) == BODY
    assert urls == ["https://data.commoncrawl.org/crawl-data/CC-MAIN-2020-05/segments/x/warc/CC-MAIN-x.warc.gz"]
    assert sleeps == []


def test_truncated_body_is_retried_then_succeeds(monkeypatch, sleeps):
    truncated = _Response(BODY[:-10], headers={"Content-Length": str(len(BODY))})
    urls = _serve(monkeypatch, truncated, _Response())
    assert download_warc(WARC_PATH) == BODY
    assert len(urls) == 2
    assert sleeps == [5.0]


def test_persistent_truncation_is_an_explicit_failure(monkeypatch, sleeps):
    urls = _serve(monkeypatch, _Response(BODY[:-10], headers={"Content-Length": str(len(BODY))}))
    with pytest.raises(DownloadError, match=r"^Failed to download .* truncated"):
        download_warc(WARC_PATH)
    assert len(urls) == download.MAX_ATTEMPTS
    assert sleeps == [5.0, 10.0, 20.0, 40.0]


def test_rate_limits_back_off_exponentially(monkeypatch, sleeps):
    urls = _serve(monkeypatch, _Response(status=503))
    with pytest.raises(DownloadError, match=r"^Failed to download .* after 5 attempts$"):
        download_warc(WARC_PATH)
    assert len(urls) == 5
    assert sleeps == [5.0, 10.0, 20.0, 40.0, 80.0]


def test_http_errors_are_retried(monkeypatch, sleeps):
    _serve(monkeypatch, _Response(status=403), _Response(status=429), _Response())
    assert download_warc(WARC_PATH) == BODY
    assert sleeps == [5.0, 10.0]


def test_deadline_turns_a_dribbling_body_into_a_retry(monkeypatch, sleeps):
    clock = iter(range(0, 100_000, 700))  # every clock read is 700 s after the previous one
    monkeypatch.setattr(download.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(download, "READ_CHUNK_BYTES", 4000)  # the 10,000-byte body arrives in 3 pieces
    urls = _serve(monkeypatch, _Response(BODY))
    with pytest.raises(DownloadError, match="exceeded 1200s"):
        download_warc(WARC_PATH)
    assert len(urls) == download.MAX_ATTEMPTS


def test_content_length_guard_is_skipped_for_encoded_transfers():
    encoded = _Response(b"abc", headers={"Content-Length": "999", "Content-Encoding": "gzip"})
    check_content_length(encoded, 3, WARC_PATH)
    check_content_length(_Response(b"abc", headers={}), 3, WARC_PATH)  # no Content-Length: nothing to check
    with pytest.raises(requests.exceptions.RequestException, match="3 of 999 bytes"):
        check_content_length(_Response(b"abc", headers={"Content-Length": "999"}), 3, WARC_PATH)
