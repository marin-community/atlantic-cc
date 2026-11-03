# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen decontamination parameters of Atlantic-CC v1.0 (decon v4, eval set ``full-val``).

Values are the as-run ones from Marin ``experiments/fast_curation/decon_8m.py`` and
``experiments/datakit/decontam/config.py`` at tag ``lpv11-8m-as-run``.
"""

from atlantic_cc.decon.matching import NGramConfig

EVAL_SET = "full-val"
MIN_MATCHED_FEATURES = 2
# The filter rule: word 13-grams, paragraph overlap >= 0.5 with >= 2 distinct eval phrases.
FILTER_NGRAM = NGramConfig(ngram_length=13, stride=0, overlap_threshold=0.5, min_matched_features=MIN_MATCHED_FEATURES)
# The mark (and every eval-side feature extraction: blooms, index, drop-set counts) uses one phrase.
MARK_NGRAM = NGramConfig(ngram_length=13, stride=0, overlap_threshold=0.5, min_matched_features=1)

# Bloom filter sizing (dupekit.Bloom): capacity = int(index rows * BLOOM_HEADROOM), p = 1e-9.
FALSE_POSITIVE_RATE = 1e-9
BLOOM_HEADROOM = 1.25
ESTIMATED_DOC_COUNT = 50_000_000

# Source-local boilerplate (per crawl): phrases in >= max(5, int(0.005 * n)) of a source's first
# <= 5,000 sampled docs.
SOURCE_DF_SAMPLE_DOCS = 5_000
SOURCE_DF_COMMON_FRAC = 0.005
SOURCE_DF_COMMON_MIN_ABS = 5
# Cross-source boilerplate: summed document frequency over each source's first <= 1,000,000 sampled
# docs >= 50, in >= 3 sources.
GLOBAL_DF_SAMPLE_DOCS = 1_000_000
GLOBAL_DF_COMMON_MIN_ABS = 50
GLOBAL_DF_COMMON_MIN_SOURCES = 3

# Drop-set sample (how the as-run per-crawl sample was drawn from the deduped_c075 file layout).
SAMPLE_SEED = 20261019
SAMPLE_FILES = 60_000
SAMPLE_PER_FILE = 2_100

# Pool and code boilerplate lists.
POOL_DROP_MIN_DF = 1_000
POOL_DROP_MIN_CRAWLS = 3
CODE_DROP_MIN_DF = 100
TRACK_MIN_CANDIDATE_DF = 20
COUNT_PARTITIONS = 64
POOL_COUNT_FILES_PER_TASK = 25
COUNT_FILES_PER_TASK = 200
CODE_FILES_PER_LANGUAGE = 10
CODE_LANGUAGES = (
    "c",
    "cpp",
    "c-sharp",
    "java",
    "javascript",
    "typescript",
    "python",
    "php",
    "go",
    "rust",
    "ruby",
    "kotlin",
    "scala",
    "shell",
    "perl",
    "lua",
    "r",
    "dart",
    "haskell",
    "css",
    "html",
    "sql",
    "julia",
)

# Sizes of the shipped artifacts (exact, used as integrity checks).
DROP_HASH_COUNT = 66_558
GLOBAL_DROP_COUNT = 31_287
EVAL_HIT_COUNT = 10_944_079
