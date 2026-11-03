# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""U-REL: resumable, MD5-checked downloads (against a local HTTP server with range support)."""

import hashlib
import http.server
import json
import os
import threading
from pathlib import Path

import pytest

from atlantic_cc.release import download
from atlantic_cc.release.params import VERSION, Kind


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """Serves files under ``directory`` with S3-like ETags (the MD5) and ``Range: bytes=N-`` support."""

    directory: Path

    def log_message(self, *args) -> None:
        pass

    def _send(self, body: bool) -> None:
        path = self.directory / self.path.lstrip("/")
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        start = int(self.headers["Range"].removeprefix("bytes=").rstrip("-")) if "Range" in self.headers else 0
        if start >= len(data) and start:
            self.send_response(416)
            self.end_headers()
            return
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(data) - start))
        self.send_header("ETag", f'"{hashlib.md5(data).hexdigest()}"')
        self.end_headers()
        if body:
            self.wfile.write(data[start:])

    def do_GET(self) -> None:
        self._send(body=True)

    def do_HEAD(self) -> None:
        self._send(body=False)


@pytest.fixture
def server(built):
    handler = type("Handler", (_RangeHandler,), {"directory": Path(built.out)})
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


PART = f"{VERSION}/crawl=CC-MAIN-2013-20/part-00000-of-00001.parquet"


def test_download_resumes_a_partial_file(built, server, tmp_path):
    data = Path(built.out, PART).read_bytes()
    dest = tmp_path / "part.parquet"
    Path(f"{dest}.part").write_bytes(data[:100])
    fetched = download.download_file(f"{server}/{PART}", str(dest), expected_md5=None)
    assert dest.read_bytes() == data
    assert fetched == len(data)
    assert not os.path.exists(f"{dest}.part")


def test_a_wrong_md5_discards_the_download(server, tmp_path):
    dest = tmp_path / "part.parquet"
    with pytest.raises(ValueError, match="md5"):
        download.download_file(f"{server}/{PART}", str(dest), expected_md5="0" * 32)
    assert not dest.exists()
    assert not os.path.exists(f"{dest}.part")


def test_download_fetches_selected_crawls_once(built, server, tmp_path):
    summary = download.download(server, str(tmp_path), ["CC-MAIN-2013-20"], [Kind.CLEAN], 2, None, VERSION)
    assert summary["files"] == 1
    assert (tmp_path / PART).read_bytes() == Path(built.out, PART).read_bytes()
    assert json.loads((tmp_path / f"{VERSION}.manifest.json").read_text())["records"] == 12
    again = download.download(server, str(tmp_path), ["CC-MAIN-2013-20"], [Kind.CLEAN], 2, None, VERSION)
    assert again["bytes_fetched"] == 0
