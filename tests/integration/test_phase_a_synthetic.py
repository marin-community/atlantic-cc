# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase A end to end on a synthetic WARC: every record fate, the counters, chunking, failures.

Offline: the real spawn pool runs the production task over a fake ``resiliparse._extract_rs``
(``tests/unit/extract/fake_engine.py``), and the real ``FastTextGate`` wraps a stub model whose
score depends on the text. Exactness on real WARCs is ``test_phase_a_golden.py`` and R1.
"""

import numpy as np
import pytest

from atlantic_cc import spec
from atlantic_cc.classify.fasttext_gate import FastTextGate
from atlantic_cc.extract.pool import ExtractorPool
from atlantic_cc.pipeline import phase_a
from atlantic_cc.pipeline.phase_a import PhaseACounters, run_phase_a
from atlantic_cc.schemas import PRESURVIVOR_SCHEMA
from atlantic_cc.warc.download import DownloadError
from atlantic_cc.warc.records import RecordStats
from tests.unit.extract.fake_engine import write_fake_engine
from tests.unit.warc.warc_builder import html_response, http_block, warc_record

WARC_PATH = "s3://commoncrawl/crawl-data/CC-NEWS/2024/06/CC-NEWS-20240601001620-04355.warc.gz"
AT_THRESHOLD = float(spec.FASTTEXT_KEEP_AT)
BELOW_THRESHOLD = float(np.nextafter(spec.FASTTEXT_KEEP_AT, np.float32(0)))


class _StubModel:
    """fastText-shaped model: P(useful) from markers in the normalized text."""

    class f:  # noqa: N801 - mirrors fasttext's ``model.f``
        @staticmethod
        def predict(text, k, threshold, on_unicode_error):
            assert "\n" not in text and text == text.lower()
            p = 0.9 if "useful" in text else AT_THRESHOLD if "boundary" in text else BELOW_THRESHOLD
            return [(p, "__label__useful"), (1.0 - p, "__label__no_useful")]


def _warc() -> bytes:
    return b"".join(
        [
            warc_record("request", "http://a/", b"GET / HTTP/1.1\r\nHost: a\r\n\r\n", "<urn:uuid:req>"),
            html_response("keep", b"<html><body>A USEFUL article</body></html>"),
            html_response("outside", b"<html><body></body>useful but outside the body</html>"),
            html_response("crash", b"<body>KILL useful</body>"),
            html_response("nothing", b"<body>NOTHING useful</body>"),
            html_response("frameset", b"<body>useful</body><frameset cols='*'></frameset>"),
            html_response("xhtml", b"<body>useful</body>", content_type="application/xhtml+xml"),
            html_response("spaces", b"<body>SPACES useful</body>"),
            html_response("below", b"<body>below the gate</body>"),
            warc_record(
                "response",
                "http://h/",
                http_block(b"x", (("Content-Type", "text/html; charset=hex"),)),
                "<urn:uuid:hex>",
            ),
            html_response("boundary", b"<body>exactly at the boundary</body>"),
            html_response("keep2", "<body>useful ü</body>".encode(), url="http://example.com/ü"),
        ]
    )


@pytest.fixture(scope="module")
def pool(tmp_path_factory):
    with ExtractorPool(2, write_fake_engine(tmp_path_factory.mktemp("engine"))) as pool:
        yield pool


def _run(pool, fetch=lambda path: _warc()):
    return run_phase_a(WARC_PATH, FastTextGate(_StubModel()), pool, fetch=fetch)


def test_every_fate_and_counter(pool):
    table, counters = _run(pool)
    assert table.schema == PRESURVIVOR_SCHEMA
    assert table.to_pylist() == [
        {
            "doc_id": "keep",
            "url": "http://example.com/keep",
            "warc_hash": "d8ed02580f8e",
            "snapshot": "CC-NEWS-2024-06",
            "fasttext_score": 0.8999999761581421,  # float32(0.9)
            "text": "md:<html><body>A USEFUL article</body></html>",  # raw extractor output
        },
        {
            "doc_id": "boundary",
            "url": "http://example.com/boundary",
            "warc_hash": "d8ed02580f8e",
            "snapshot": "CC-NEWS-2024-06",
            "fasttext_score": AT_THRESHOLD,
            "text": "md:<body>exactly at the boundary</body>",
        },
        {
            "doc_id": "keep2",
            "url": "http://example.com/ü",
            "warc_hash": "d8ed02580f8e",
            "snapshot": "CC-NEWS-2024-06",
            "fasttext_score": 0.8999999761581421,
            "text": "md:<body>useful ü</body>",
        },
    ]
    # n_in: the 10 text/html responses. Prefilter: outside + crash. Empty: nothing + frameset.
    # spaces and below are fastText drops and are not counted anywhere.
    assert counters == PhaseACounters(
        n_in=9, n_prefilter_dropped=2, n_extract_empty=2, n_extract_crashed=1, records=RecordStats(n_errors=1)
    )


def test_chunking_does_not_change_outputs(pool, monkeypatch):
    whole = _run(pool)
    monkeypatch.setattr(phase_a, "CHUNK_RECORDS", 2)
    chunked = _run(pool)
    assert chunked[0].equals(whole[0])
    assert chunked[1] == whole[1]


def test_unfetchable_warc_is_an_explicit_failure(pool):
    def fetch(path):
        raise DownloadError(f"Failed to download {path} after 5 attempts: 403 Forbidden" + "!" * 400)

    table, counters = _run(pool, fetch)
    assert table.num_rows == 0 and table.schema == PRESURVIVOR_SCHEMA
    assert counters.download_failed and counters.n_in == 0
    assert counters.download_error.startswith(f"Failed to download {WARC_PATH}")
    assert len(counters.download_error) == phase_a.DOWNLOAD_ERROR_MAX_CHARS


def test_other_fetch_errors_propagate(pool):
    def fetch(path):
        raise OSError("disk on fire")

    with pytest.raises(OSError, match="disk on fire"):
        _run(pool, fetch)
