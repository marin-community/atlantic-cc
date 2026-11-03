# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase A of the per-WARC worker: download -> decode -> prefilter -> extract -> fastText gate.

A port of production's ``fused_phase.a_extract_warc``. The output is the WARC's fastText survivors
("presurvivors") in WARC record order with the raw extracted text; phase B tokenizes and scores
them. Records are processed in chunks of 2,000 (bounds memory; no effect on outputs).

Per record, in order:

1. not in the population (empty ``body_strip`` text) or crashed in the extractor:
   ``n_prefilter_dropped`` (crashes also count in ``n_extract_crashed``);
2. extracted text empty after ``_safe_utf8``: ``n_extract_empty``;
3. ``float32(P(useful)) < float32(0.0048)`` on the normalized text: dropped, not counted;
4. otherwise a presurvivor row ``{doc_id, url, warc_hash, snapshot, fasttext_score, text}``.

Whitespace-only text passes step 2 and scores 0.0 at step 3. A WARC that cannot be downloaded gives
zero rows and ``download_failed``; any other error propagates (production then retried the WARC).
"""

import io
import itertools
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import pyarrow as pa

from atlantic_cc import spec
from atlantic_cc.classify.fasttext_gate import FastTextGate, passes_gate
from atlantic_cc.classify.text import normalize_text
from atlantic_cc.extract.pool import ExtractorPool
from atlantic_cc.schemas import PRESURVIVOR_SCHEMA
from atlantic_cc.warc.download import DownloadError, download_warc
from atlantic_cc.warc.identity import snapshot_of, warc_hash
from atlantic_cc.warc.records import RecordStats, iter_html_records

logger = logging.getLogger(__name__)

CHUNK_RECORDS = 2000
DOWNLOAD_ERROR_MAX_CHARS = 300


@dataclass
class PhaseACounters:
    n_in: int = 0  # decoded text/html responses
    n_prefilter_dropped: int = 0  # out of the population, plus extractor crashes
    n_extract_empty: int = 0
    n_extract_crashed: int = 0
    download_failed: bool = False
    download_error: str = ""  # production's ``a_download_failed`` reason, truncated to 300 chars
    records: RecordStats = field(default_factory=RecordStats)  # records skipped before ``n_in`` (diagnostic)


def safe_utf8(s: str) -> str:
    """Drop code points that cannot be UTF-8 encoded (lone surrogates)."""
    return s.encode("utf-8", "ignore").decode("utf-8")


def run_phase_a(
    warc_path: str,
    gate: FastTextGate,
    extractor_pool: ExtractorPool,
    *,
    fetch: Callable[[str], bytes] = download_warc,
) -> tuple[pa.Table, PhaseACounters]:
    """Presurvivors of one WARC (``schemas.PRESURVIVOR_SCHEMA``) and its phase-A counters.

    ``warc_path`` is the full ``s3://commoncrawl/...`` path: it determines ``warc_hash`` and
    ``snapshot``. ``fetch`` returns the WARC's bytes and raises ``DownloadError`` when it cannot.
    """
    h = warc_hash(warc_path)
    snapshot = snapshot_of(warc_path)
    counters = PhaseACounters()
    try:
        raw = fetch(warc_path)
    except DownloadError as e:
        logger.error("WARC %s unfetchable, recording explicit failure: %s", h, e)
        counters.download_failed = True
        counters.download_error = str(e)[:DOWNLOAD_ERROR_MAX_CHARS]
        return PRESURVIVOR_SCHEMA.empty_table(), counters

    rows: list[dict] = []
    html_records = iter_html_records(io.BytesIO(raw), warc_path, counters.records)
    for chunk in itertools.batched(html_records, CHUNK_RECORDS):
        counters.n_in += len(chunk)
        results, crashed = extractor_pool.run([r.html for r in chunk], [r.doc_id for r in chunk], spec.MAX_HTML_CHARS)
        counters.n_extract_crashed += len(crashed)
        for record, (text, in_population) in zip(chunk, results, strict=True):
            if not in_population:
                counters.n_prefilter_dropped += 1
                continue
            text = safe_utf8(text)
            if not text:
                counters.n_extract_empty += 1
                continue
            prob = gate.p_useful(normalize_text(text))
            if not passes_gate(prob):
                continue
            rows.append(
                {
                    "doc_id": record.doc_id,
                    "url": safe_utf8(record.url),
                    "warc_hash": h,
                    "snapshot": snapshot,
                    "fasttext_score": prob,
                    "text": text,
                }
            )
    if counters.records.n_brotli:
        logger.info("%s: %d text/html records were brotli-encoded", h, counters.records.n_brotli)
    table = pa.Table.from_pylist(rows, schema=PRESURVIVOR_SCHEMA) if rows else PRESURVIVOR_SCHEMA.empty_table()
    return table, counters
