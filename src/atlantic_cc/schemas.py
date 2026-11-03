# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Arrow schemas shared between pipeline stages.

``kept/data-{warc_hash}.parquet`` is the per-WARC output of extraction + classification and the
input to dedup. Its schema matches production's ``KEPT_V3_SCHEMA`` exactly so that the standalone
dedup can run on the as-run ``kept/`` tree and vice versa.
"""

import pyarrow as pa

# One row per fastText survivor, in WARC record order; held in memory between phase A and phase B.
PRESURVIVOR_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),  # WARC-Record-ID without "<urn:uuid:" ">", a bare UUID
        ("url", pa.string()),  # WARC-Target-URI
        ("warc_hash", pa.string()),  # sha256("s3://commoncrawl/<path>")[:12]
        ("snapshot", pa.string()),  # CC-MAIN-YYYY-WW, CC-NEWS-YYYY-MM or CC-SUPPLEMENTAL-YYYY-WW
        ("fasttext_score", pa.float32()),
        ("text", pa.large_string()),  # raw resiliparse-rs output (not normalized)
    ]
)

# Kept documents of one WARC, in WARC record order. ``modernbert_prob`` is NaN (not null) for
# documents the pooled model accepted outright. Written with zstd level 12.
KEPT_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("url", pa.string()),
        ("warc_hash", pa.string()),
        ("snapshot", pa.string()),
        ("fasttext_score", pa.float32()),
        ("text", pa.large_string()),
        ("n_tokens", pa.int32()),  # ModernBERT tokens incl. [CLS]/[SEP], <= 8192
        ("pooled_prob", pa.float32()),
        ("modernbert_prob", pa.float32()),
    ]
)
KEPT_ZSTD_LEVEL = 12

# Tombstones: ``tombstones/data-{warc_hash}.jsonl.gz``, one JSON line per fastText survivor that was
# not kept, in WARC record order: {"doc_id", "pooled_prob": round(p, 6),
# "modernbert_prob": round(p, 6) or null}. null means pooled dropped the doc (< POOLED_DROP_BELOW).
TOMBSTONE_PROB_DECIMALS = 6
