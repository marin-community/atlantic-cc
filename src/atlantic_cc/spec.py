# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen definition of the Atlantic-CC v1.0 extraction and classification cascade.

These constants reproduce the production spec ``lpv11_fastpipe_v2_1_fused`` (version hash
``2a076ddfcb``) that built the release. Changing any of them changes which documents are kept, so
they are not configuration: a different cascade is a different dataset version.

Stage-specific constants for dedup, decontamination and release packaging live next to that code
(``atlantic_cc.dedup.params``, ``atlantic_cc.decon.params``, ``atlantic_cc.release.params``).
"""

import numpy as np

DATASET_VERSION = "v1.0"
PRODUCTION_SPEC_ID = "lpv11_fastpipe_v2_1_fused"
PRODUCTION_SPEC_HASH = "2a076ddfcb"

# --- Download -----------------------------------------------------------------------------------
CC_S3_PREFIX = "s3://commoncrawl/"
CC_HTTPS_PREFIX = "https://data.commoncrawl.org/"

# --- Extraction (resiliparse-rs, a fork of chatnoir-resiliparse) ---------------------------------
EXTRACTOR_REPO = "https://github.com/marin-community/atlantic-cc-resiliparse"
EXTRACTOR_TAG = "atlantic-cc-v1.0"  # annotated tag on EXTRACTOR_COMMIT
EXTRACTOR_COMMIT = "850891b277e919f671ec19f12aafdce451b327f4"
# Pages whose decoded HTML is longer than this many code points are dropped before extraction.
MAX_HTML_CHARS = 50_000_000

# --- Classifier thresholds -----------------------------------------------------------------------
# Production compared float32 scores against these values under NumPy 2 (NEP 50) rules, so the
# effective thresholds are the float32 roundings, e.g. float32(0.4378) = 0.43779999 < 0.4378.
FASTTEXT_KEEP_AT = np.float32(0.0048)  # keep iff P(useful) >= this
POOLED_DROP_BELOW = np.float32(0.079)  # drop iff pooled P(useful) < this
POOLED_ACCEPT_AT = np.float32(0.8883)  # keep without the terminal model iff pooled P(useful) >= this
TERMINAL_KEEP_AT = np.float32(0.4378)  # band docs: keep iff Ettin P(useful) >= this

# --- Tokenization (ModernBERT-base BPE) ----------------------------------------------------------
TOKENIZER_REPO = "answerdotai/ModernBERT-base"
TOKENIZER_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
CLS_TOKEN_ID = 50281
SEP_TOKEN_ID = 50282
PAD_TOKEN_ID = 50283
# Only the first 65,536 code points of the normalized text are tokenized (8 chars/token * 8192).
TOKENIZE_CHAR_CAP = 65_536
POOLED_MAX_TOKENS = 8192  # [CLS] + up to 8,190 body tokens + [SEP]
TERMINAL_MAX_TOKENS = 2048  # derived from the 8192 ids: ids[:2047] + [SEP]

# --- Batched scoring -----------------------------------------------------------------------------
SCORE_BATCH_SIZE = 32
LENGTH_BUCKETS = (128, 256, 512, 1024, 2048, 4096, 8192)
USEFUL_CLASS_INDEX = 1  # softmax over 2 classes; index 1 = useful

# --- Released code and classifiers ---------------------------------------------------------------
CODE_REPO_URL = "https://github.com/marin-community/atlantic-cc"
# Hugging Face repos (marin-community). The as-run artifacts' sizes and MD5s are recorded so the
# Hub copies can be checked against what production actually loaded.
FASTTEXT_HF_REPO = "marin-community/atlantic-cc-fasttext"
POOLED_HF_REPO = "marin-community/atlantic-cc-pooled-transformer"
# The terminal classifier: a fine-tuned Ettin-68M encoder (ModernBERT architecture).
TERMINAL_HF_REPO = "marin-community/atlantic-cc-modernbert"

ASRUN_FASTTEXT_MODEL = {
    "uri": "gs://marin-us-east5/classifiers/useful_fasttext_lpv11/resiliparse_scale_w640_sub0p22_strat_prep_mc500_TEXT/model.bin",
    "bytes": 918_710_997,
    "md5_b64": "AHpq5KR/JN7qQyprSoAyWw==",
}
ASRUN_POOLED_MODEL = {
    "uri": "gs://marin-us-east5/checkpoints/modernbert-useful/mb-clf-lpv11-text-pooled-90M/hf/model.eqx",
    "bytes": 103_794_816,
    "md5_b64": "5fiflzlFT9VZL1+nt8fwUg==",
}
ASRUN_TERMINAL_MODEL = {
    "uri": "gs://marin-us-east5/checkpoints/modernbert-useful/mb-clf-lpv11-text-ettin68-10M/hf/model.safetensors",
    "bytes": 273_645_504,
    "md5_b64": "m+Bu9NgBJanvpCKqPlchFg==",
}
# Production's two extractor binaries (identical copies in every region's bucket); the as-run wheel
# in docs/extractor.md packages exactly these files.
ASRUN_EXTRACTOR_ARTIFACT = {
    "uri": "gs://marin-us-east5/artifacts/resiliparse_rs/latest/",
    "_extract_rs.so": {
        "bytes": 3_979_816,
        "md5_b64": "mjHiDXINNTxVCUttKER1Iw==",
        "sha256": "9e2bdfe91dc14fbe17846c0d1c0806eb12b19f64bba6db9cf8669fd1bd19a364",
    },
    "liblexbor.so.2": {
        "bytes": 12_431_649,
        "md5_b64": "hX51w84LHTU6X/dTPVUyfw==",
        "sha256": "cf016d48cf67a8b6d859cb25e71c82c3c7ff414cfcaedf595e8b3c158f5aecd2",
    },
}
