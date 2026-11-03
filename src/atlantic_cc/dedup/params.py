# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Every constant of the Atlantic-CC v1.0 dedup, with the as-run source it was taken from.

Paths are in the Marin monorepo at tag ``lpv11-8m-as-run`` (commit d3ebceb6bc) unless noted. The solver and
its limits come from the fork's ``cluster_dedup.py`` / ``cluster_verify.py`` at f47919338b, which is what that
tag carries. Changing any value here changes which documents survive.
"""

import dataclasses

# --- Normalization and exact dedup (experiments/fast_curation/rdedup/docs.py:38-46) ---------------------------
MAX_WHITESPACE_RUN_CHARS = 128  # longer whitespace runs are cut to their first 128 characters
ROW_BITS = 24  # doc key = file_index << 24 | row
MAX_ROWS_PER_FILE = 1 << ROW_BITS
# fuzzy_dups tags every node with its co-partitioned source; a single-source run is source_000. Only the
# canonical-member order (``attrs.canon``) uses it.
CC_SOURCE_TAG = "source_000"
# Kept columns that never reach the dedup output (experiments/fast_curation/project_text_only.py:33).
DROP_COLUMNS = ("input_ids", "n_tokens")

# --- MinHash / LSH (rdedup/docs.py:38-41, 49-56; dupekit minhash_ops.rs) ----------------------------------------
NUM_PERMS = 286
NUM_BANDS = 26  # 26 bands x 11 rows; the band index is not mixed into the bucket hash
NGRAM_SIZE = 5  # character 5-grams of the CleanText output
SEED = 42  # rand_pcg Pcg64 seed of the 286 odd u128 multipliers
APPROX_JACCARD_THRESHOLD = 0.75  # reported in the stats only; the S-curve midpoint is (1/26)**(1/11) = 0.744

# --- Connected components (rdedup/graph.py:88) -------------------------------------------------------------------
# Labels compare in the order key * LABEL_MIX mod 2**64; the root of a component is its member that is smallest
# in that order.
LABEL_MIX = 0x9E3779B97F4A7C15


# --- c075 containment rule (experiments/baseline_collection/verified_dedup.py:65-74) -----------------------------
@dataclasses.dataclass(frozen=True)
class ClusterDedupParams:
    """Thresholds and work bounds of the containment solver (as-run ``ClusterDedupParams``, a pydantic model)."""

    rule_version: str
    ngram_size: int
    minimum_containment: float
    exact_scan_maximum: int  # batches with at most this many members compare every ordered pair
    probe_ngrams: int  # a member's rarest n-grams that probe the inverted index
    maximum_posting_length: int  # n-grams held by more documents are skipped as boilerplate
    maximum_candidates: int


C075_PARAMS = ClusterDedupParams(
    rule_version="containment_ngram3_v2",
    ngram_size=3,
    minimum_containment=0.75,
    exact_scan_maximum=256,
    probe_ngrams=32,
    maximum_posting_length=512,
    maximum_candidates=32,
)
MAX_CLUSTER_SIZE = 100_000  # components with more members (strictly) are split before solving

# --- Oversized split (experiments/datakit/scripts/fuzzy_cluster_text.py:65-77 @ tag) ----------------------------
SPLIT_NGRAM_SIZE = 5  # word 5-grams of the split MinHash
SPLIT_SUBDIVISIONS = 16  # uniform sub-split of each split by content id
TEXT_MAX_DOCUMENT_CHARS = 64 * 1024 * 1024  # member text is cut to this many code points before grouping

# --- Solve limits (lib/marin/.../deduplication/cluster_verify.py:69, 75 @ f47919338b) ----------------------------
SOLVE_MAX_CLUSTER_CHARS = 256 * 1024 * 1024  # characters of one solve batch
SOLVE_MAX_DOCUMENT_CHARS = 8 * 1024 * 1024  # each document is cut to this many code points before solving

# --- Scheduling of the c075 stages (rdedup/stages.py:835, 981); no effect on outputs ------------------------------
C075_FLUSH_CHARS = 512 * 1024 * 1024  # map-side text is flushed to partition frames in batches of this many chars
SOLVE_PASS_COMPRESSED_BYTES = 400 * 1024 * 1024  # compressed partition bytes one solve pass loads

# --- Output layout (rdedup/stages.py:83-85; launcher.py:286-288) ------------------------------------------------
GZIP_LEVEL = 6  # one gzip member per input kept file, mtime 0
UNIT_BYTES = int(8.0 * 1e9)  # compressed kept bytes per unit (--unit-gb 8)
FILES_PER_OUTPUT = 32  # kept files per output file: data-{unit:06d}-{position // 32:03d}.jsonl.gz
OUTPUT_DIR = "deduped_c075"
RULE = "c075"
EXACT_COPIES_FIELD = "exact_copies"
CLUSTER_SIZE_FIELD = "fuzzy_cluster_size"
