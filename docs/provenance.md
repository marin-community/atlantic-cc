# Provenance

What produced Atlantic-CC v1.0, for audit. [reproducibility.md](reproducibility.md) says how far each part
can be reproduced.

## Production run

| | |
|---|---|
| Cascade spec | `lpv11_fastpipe_v2_1_fused`, version hash `2a076ddfcb`, frozen in `src/atlantic_cc/spec.py`. Shards 0–403 (99,638 WARCs) ran an earlier two-phase build of the same cascade, spec `lpv11_fastpipe_v2_1` (`944ca6bc38`): phase A on CPU hosts, phase B on TPUs. |
| Thresholds (float32) | fastText ≥ 0.0048; pooled transformer < 0.079 drops, ≥ 0.8883 keeps; ModernBERT ≥ 0.4378 ([classifiers.md](classifiers.md)) |
| Code | The Marin repository at tag `lpv11-8m-as-run` (commit `d3ebceb6bc`, a snapshot of 2026-10-29), which this repository ports. Workers ran working-tree snapshots of it; the spec hash, the model files, the thresholds and the extractor commit fix the cascade. Changes during the run: the download Content-Length guard (absent for shards 0–8079, processed until 2026-09-07), the three-attempt cap (from about 2026-10-08), and crawl names for CC-SUPPLEMENTAL and CC-NEWS (added with their shards). |
| Accelerators | TPU v5e (2,329,442 WARCs), v6e (2,154,124), v5p (2,083,848) and v4 (2,014,701), per the catalog's `device` column |
| Regions | Google Cloud us-central1, us-central2, us-east1, us-east5, us-west4 and europe-west4 |

| Stage | Dates (UTC) | Output |
|---|---|---|
| Worklist | base crawls as listed on 2026-04-10; built 2026-08-29; extended 2026-10-13 and 2026-10-15 | 8,582,115 WARCs |
| Extraction and classification | 2026-08-29 to 2026-10-17 | 61,211,321,760 kept documents, one `kept/` file per WARC |
| Dedup | 2026-10-17 05:06 to 2026-10-18 11:41 | 9,951,767,085 documents |
| Decontamination | 2026-10-19 to 2026-10-26 | 1,349,408 documents flagged |
| Packaging | 2026-10-28 to 2026-10-29 | 27,985 Parquet files, 22.77 TB |
| Upload to Common Crawl | 2026-10-29 to 2026-10-30 | `s3://commoncrawl/projects/atlantic-cc/` |

## Extractor

| | |
|---|---|
| Source | [`marin-community/atlantic-cc-resiliparse`](https://github.com/marin-community/atlantic-cc-resiliparse) @ `850891b277e919f671ec19f12aafdce451b327f4` (tag `atlantic-cc-v1.0`); `extract_plain_text(html, main_content=True, preserve_formatting="markdown")` |
| `_extract_rs` extension | 3,979,816 B, sha256 `9e2bdfe91dc14fbe17846c0d1c0806eb12b19f64bba6db9cf8669fd1bd19a364`; built 2026-08-26, rustc 1.98.0 |
| `liblexbor.so.2` | 12,431,649 B, sha256 `cf016d48cf67a8b6d859cb25e71c82c3c7ff414cfcaedf595e8b3c158f5aecd2`; built 2026-08-11, gcc 14.2.0, lexbor `647a0b5` |
| As-run wheel | `resiliparse_rs_atlantic_cc-0+850891b-cp312-cp312-manylinux_2_34_x86_64.whl`, sha256 `7747e6e4dce15057df3082275fcdd722a4a81a571aeda33dc919365baf1439f8`: the two binaries above, unchanged ([extractor.md](extractor.md)) |

## Classifiers

| Model | Hugging Face repo | File production loaded | Bytes | md5 (base64) | On the Hub |
|---|---|---|---:|---|---|
| fastText | [`marin-community/atlantic-cc-fasttext`](https://huggingface.co/marin-community/atlantic-cc-fasttext) | `model.bin` | 918,710,997 | `AHpq5KR/JN7qQyprSoAyWw==` | `model.bin`: the same bytes |
| pooled transformer | [`marin-community/atlantic-cc-pooled-transformer`](https://huggingface.co/marin-community/atlantic-cc-pooled-transformer) | Levanter `model.eqx` | 103,794,816 | `5fiflzlFT9VZL1+nt8fwUg==` | `levanter/model.eqx`: the same bytes. `model.safetensors` is converted from it (PyTorch layout, two unused query vectors dropped); M2 checks its scores |
| ModernBERT (Ettin-68M) | [`marin-community/atlantic-cc-modernbert`](https://huggingface.co/marin-community/atlantic-cc-modernbert) | `model.safetensors` | 273,645,504 | `m+Bu9NgBJanvpCKqPlchFg==` | `model.safetensors`: a lossless re-save, every tensor bitwise equal (checked when staging), with label names and an explicit zero `classifier.bias` added; M4 loads it in fresh environments |

`atlantic_cc.spec` records these repos, sizes and md5s (`*_HF_REPO`, `ASRUN_*`). Each repo's `manifest.json`
gives the sha256 of every file and, for copied files, production's source and md5.

**Tokenizer:** `answerdotai/ModernBERT-base` at revision `8949b909ec900327062f0ebf497f51aef5e6f0c8`, run
through gigatoken 0.10.0 and checked at startup against Hugging Face `tokenizers` 0.22.2
([extraction.md](extraction.md)).

**Dependencies:** `pyproject.toml` pins, and `uv.lock` locks, production's versions of every package that
affects outputs (list in [reproducibility.md](reproducibility.md#the-contract)); production's lock has the
same versions in the run's first committed snapshot (2026-09-05) and at the tag. The extractor's toolchain is
in [extractor.md](extractor.md#pinned-build-inputs).

## Worklist

| | |
|---|---|
| WARCs | 8,582,115: every WARC of 249 crawls ([worklist.md](worklist.md)) |
| sha256 of the uncompressed list | `101505dd737241a05afb03338b6e2db6175a40651d57538a7c4db0fd4576961a` |
| Shards | 34,628, production's work units; they change no document |
| WARCs with no output | 5,978: 3,905 empty at the source, 101 abandoned (three attempts that hung the extractor or killed the worker), 1,422 refused (HTTP 403) and 550 throttled (HTTP 503) at audit time |

## Dedup, decontamination, packaging

| Stage | Parameters |
|---|---|
| Dedup ([dedup.md](dedup.md)) | exact (`xxh3_128` of the text, whitespace runs cut to 128), then MinHash (286 permutations, character 5-grams, seed 42), LSH 26 × 11, connected components, and c075: a member is removed when a longer surviving member contains ≥ 75% of its word 3-grams. Input: 8,582,115 kept files, `inputs_sha256` `b403438eeb676f417fe6aff69efe28fe185bded93563a92b1399f5c33a9bd526` |
| Decontamination ([decon.md](decon.md)) | Marin decontamination v4: 13-token phrases; a paragraph matches when ≥ 50% of its phrases, and ≥ 2 distinct ones, are eval phrases; eval set `full-val` (2,106,579 records); drop set D of 66,558 boilerplate hashes; flagged documents kept in `contaminated-*` files |
| Packaging ([release_format.md](release_format.md)) | 249 crawl partitions; Parquet, zstd level 9, 10,000-row groups, pyarrow 23.0.1; 27,985 files |

## Not shipped

Common Crawl hosts the data ([release_format.md](release_format.md)). Not part of this release:

| Item | Status |
|---|---|
| Decontamination bloom filter (`full-val`, 901 MB, sha256 `8fe585247c881e882e601bab0230c280954380a3f22f9369805e8fc5f5a9ff67`) | Not shipped: it also covers evaluation phrases that never occur on the web, including gated sets, so shipping it needs a licensing decision. D (in this package) and S′ (the eval-phrase hashes found in the pool) replay the release exactly; new text needs the bloom. |
| Per-WARC `kept/` and tombstone files (61.2B documents before dedup) | Not published |
| The Marin code that ran (tag `lpv11-8m-as-run`) | Not in a public repository as of 2026-11-03; this repository ports it |

Release assets not yet published; these digests pin them wherever they are published:

| File | Bytes | sha256 |
|---|---:|---|
| `v1.0.checksums.md5` | 2,492,924 | `a44350759eaecbd543b7fcc420f9b247dc405f4d84090df36f0e55f1419118f8` |
| `v1.0.input.warc.paths.gz` (the worklist) | 23,757,911 | `75fb0936859373ab2ac29aa548ad35fe067a3dbc706b4b71ae13195ea4b71d7a` |
| `v1.0.input.empty_warcs.tsv.gz` | 96,299 | `13aa3a363e56bb2ce4b5f73fe3fcc1e37b09f79b377536d55026c288a6dc5841` |
| `v1.0.input.warc_catalog.parquet` | 178,471,321 | `79c6f1b1855ce2c81c6ed87a04d37aaffad0f096e12caf36cbc44602593b39fd` |
| `eval_hits.npy` (S′) | 87,552,760 | `243edfa7306c95cd9ba8bf45f8eae54a7f3f89640afdb7b18de96f72214e7205` |
| `eval_hit_names.parquet` | 94,902,663 | `b6f2653e3e3dc94f9321b253026752247d7897592833b58da7ab3b9d8ede3681` |
| `eval_record_digests.parquet` | 68,468,737 | `c2db5681dc0b63d2dab5d113b0548ff66c17caadb1e5e6892c501e5b406ec651` |
| `resiliparse_rs_atlantic_cc-0+850891b-cp312-cp312-manylinux_2_34_x86_64.whl` (as-run) | 3,247,529 | `7747e6e4dce15057df3082275fcdd722a4a81a571aeda33dc919365baf1439f8` |
| `resiliparse_rs_atlantic_cc-0+850891b-cp312-cp312-manylinux_2_28_x86_64.whl` (portable source build) | 3,248,878 | `a3373f065fdbdd44c3cd09111b09c83016536bca09984f1b1ffe618e3f7fa979` |
