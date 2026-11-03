# Decontamination

Atlantic-CC v1.0 keeps every deduplicated document and marks the ones that overlap evaluation data.
Clean documents are in `crawl=<id>/part-*.parquet` with `decon_matches == []`; the 1,349,408 flagged
documents (0.0136% of 9,951,767,085) are in `crawl=<id>/contaminated-*.parquet`, each with the sorted
names of the evals it matched, for example `["paloma_4chan"]` or
`["hendrycks_math_precalc", "minerva_math_precalc"]`.

This page defines the rule, the eval set, the boilerplate drop sets and the shipped hash lists, and
says what can be reproduced exactly. Code: `src/atlantic_cc/decon/`. Command: `atlantic-cc decon`.

## The rule

A document is flagged if

```python
matched = _document_overlap_and_matches(
    text, B, NGramConfig(ngram_length=13, stride=0, overlap_threshold=0.5, min_matched_features=2), D
)[1]
decon_matches = sorted({eval_name(eval_id) for h in matched for eval_id in INDEX[h]})
```

`_document_overlap_and_matches` is a verbatim port of the as-run rule (Marin decontamination v4,
`atlantic_cc.decon.matching`). In words:

- **Tokens** are `str.split()` tokens (Python 3.12 whitespace, which includes `\x1c`-`\x1f`, NBSP and
  U+2028 but not the zero-width space). There is no lower-casing, Unicode normalization or punctuation
  stripping. A **phrase** is 13 consecutive tokens joined by one space; phrases without any alphabetic
  character are ignored. A phrase is identified by `blake2b(phrase.encode(), digest_size=8)` read as a
  big-endian unsigned 64-bit integer. Keep hashes as uint64: an int64 view probes different bloom bits.
- **Paragraphs** are the non-empty pieces of `text.split("\n\n")`.
- A paragraph **qualifies** when at least half of its phrases are eval phrases (`>= 0.5`, repeats
  counted) and it contains at least 2 distinct eval phrases. Phrases in the drop set D are removed from
  the paragraph first (from both the numerator and the denominator).
- Single-phrase exceptions: a single-paragraph document whose only phrase is an eval phrase; a document
  with exactly one remaining phrase overall that is an eval phrase; and, when no paragraph has 13 tokens,
  the whole text scored as one paragraph (this catches multiple-choice items written as short lines).
  A paragraph of 3-12 tokens is one "short exact" feature; a short eval record can only flag a document
  whose whole text is that record (on one line or split over short lines).
- `matched` is the sorted set of eval-phrase hashes from qualifying paragraphs.

Eval-side features (`_extract_features`) differ slightly: a whole eval record of 3-12 tokens is one
exact feature, otherwise the 13-grams of each paragraph, and when that yields nothing, the 13-grams of
the whole record across paragraph breaks.

## The eval set (`full-val`)

`full-val` holds 2,106,579 records:

| Component | Records | Source |
|---|---|---|
| Artificial Analysis Intelligence Index v4.1.1, 9 benchmarks | 4,165 | GDPval-AA v2, tau3-Banking, Terminal-Bench v2.1, SciCode, AA-LCR, AA-Omniscience (public), HLE, GPQA Diamond, CritPt, at pinned revisions (`atlantic_cc.decon.eval_corpus.AA_EVALS`) |
| lm-eval-harness, 339 configured tasks, 1,114 leaf files | 1,939,089 | `stanford-crfm/lm-evaluation-harness@d5e3391f`; test split, else validation, else training |
| Validation sets | 163,325 | 16 Paloma subsets (`val`), 7 Uncheatable Eval subsets (2025-09-01..14), LIMA |

lm-eval documents are indexed as prompt + target + every string field; for reading-comprehension
documents with a passage field (`passage`, `context`, `article`, `premise`, ...), the passage and the
rendered prompt are dropped so that quoting a public passage is not a match. Record ids are
`<task>-<split>-<i>` (lm-eval), `<benchmark>-<id>` (AA) and `<set>-<i>` (validation). `eval_name`
(`atlantic_cc.decon.names`) strips the split and record suffixes; ids whose last segment is neither all
digits nor >= 16 hex digits stay whole (`scicode-77.1`, `terminal_bench_2_1-crack-7z-hash`). The release
contains 484 distinct names.

The 133,669,639 index rows (one per distinct feature per record) went into the `full-val` bloom filter:
capacity `int((93,734,507 + 39,935,132) * 1.25) = 167,087,048`, false-positive rate 1e-9, 29 hash
functions, 7,206,925,992 bits, file sha256 `8fe585247c881e882e601bab0230c280954380a3f22f9369805e8fc5f5a9ff67`
(`dupekit.Bloom`; `atlantic_cc.decon.bloom` reads the same file format without native code).

## Boilerplate drop sets (D)

Some eval phrases are ordinary web or code boilerplate (license headers, legal clauses, navigation
text). They are removed before scoring. D is the union of four lists, 66,558 hashes in total:

| List | Rule | Hashes |
|---|---|---|
| per-crawl | per source, bloom-hit phrases in >= `max(5, int(0.005 n))` of the source's first n <= 5,000 sampled docs | 94 (in 24 of 127 sources) |
| global | summed document frequency over every source's first <= 1,000,000 sampled docs >= 50, in >= 3 sources | 31,287 |
| pooldf1000 | phrases in >= 1,000 pool documents across >= 3 crawls (exact counts over all 9.95B documents) | 60,158 |
| codedf100 | phrases in >= 100 files of a 167-file StarCoderData sample | 9,901 |

A source is a crawl, with all CC-NEWS months merged. Every document is scored with the union of all
crawls' lists. The per-crawl and global lists were counted on a seeded sample of the deduplicated pool
(2,100 documents from each of 60,000 shards, split by crawl); the sample depends on the internal shard
layout and is not reproducible from the regrouped release, so D is shipped.

## Shipped artifacts

| Artifact | Where | Size | Digest |
|---|---|---|---|
| D, `drop_hashes.npy` (sorted uint64) | this package, `atlantic_cc/decon/data/` | 66,558 hashes, 532,592 B | sha256 of the LE uint64 bytes `5d71024672f39cd7e52fe4ae97c6d846481cbcf290aacf5d9cbd4ed787795fb8` |
| S', `eval_hits.npy` (sorted uint64) | [v1.0 release][release] | 10,944,079 hashes, 87,552,760 B | array sha256 `1890774187c40b516dd1243ec69f9022e7c5854e9ae64e3ce19fbca955a828a0`; file sha256 `243edfa7306c95cd9ba8bf45f8eae54a7f3f89640afdb7b18de96f72214e7205` |
| `eval_hit_names.parquet` (`hash uint64`, `eval_names list<string>`) | [v1.0 release][release] | 10,944,079 rows, 94,902,663 B | file sha256 `b6f2653e3e3dc94f9321b253026752247d7897592833b58da7ab3b9d8ede3681` |
| eval corpus digests, `eval_file_digests.json` (per file: records, sha256 of `id\tsha256(text)` lines, sha256 of sorted distinct feature hashes) | this package, `data/as_run/` | 1,147 files, 366 KB | |
| eval record digests, `eval_record_digests.parquet` (`file`, `id`, `text_sha256`) | [v1.0 release][release] | 2,106,579 rows, 68,468,737 B | file sha256 `c2db5681dc0b63d2dab5d113b0548ff66c17caadb1e5e6892c501e5b406ec651` |
| as-run manifests and provenance (`aa_manifest.json`, `lmh_manifest.json`, `val_manifest.json`, `provenance.json`) | this package, `data/as_run/` | 220 KB | task names, record counts, summaries, object md5s |

S' is every eval phrase that the as-run mark pass (one required phrase, no drop set) matched anywhere in
the 9.95B-document pool. It contains no eval text, only hashes of 13-word phrases (and of whole short
records) that already occur in public Common Crawl documents. The full bloom (901 MB) also covers phrases
that never occur on the web (HLE, GPQA, Paloma, LIMA) and is not shipped by default.

[release]: https://github.com/marin-community/atlantic-cc/releases/tag/v1.0

## Replay: exact on the released text

```bash
for f in eval_hits.npy eval_hit_names.parquet; do
    curl -LO "https://github.com/marin-community/atlantic-cc/releases/download/v1.0/$f"
done
atlantic-cc decon filter --kind release \
    --eval-hits eval_hits.npy --names eval_hit_names.parquet --workers 8 \
    --out decon_out/ 'crawl=CC-NEWS-2016-08/*.parquet'
```

For release files the command also compares every record with its stored `decon_matches` and reports
mismatches. `--kind dedup` reads deduplicated `*.jsonl.gz` shards (`doc_id`, `text`). `--bloom filter.bin`
replaces S' when you have the bloom; it also works on text that is not in the release.

Why one pass over S' equals the as-run two passes over the bloom: every drop hash is a bloom hit, so
dropping hashes can only lower a paragraph's score and its distinct hits, and the single-phrase
exceptions require every remaining phrase to be a hit. Every document the filter flags is therefore a
mark candidate, and every hash in a qualifying paragraph is a mark-matched hash, i.e. in S'. Replacing the
bloom by S' changes only paragraphs that do not qualify either way. This holds for any text that the mark
saw, which includes all released text (the release copies `text` unchanged). For new text, S' is
incomplete; use the bloom.

Cost: about 3-4 ms of CPU per document in Python (13-gram hashing dominates), i.e. ~10,000 CPU hours for
the whole release. Memory: ~1 GB per worker process for S' (shared copy-on-write by the fork-based pool).

## Rebuild: what can and cannot be reproduced

| Step | Command | Reproducibility |
|---|---|---|
| eval corpus | `decon eval-corpus aa\|lmh\|download-val\|stage-val` | best effort: lm-eval datasets were not revision-pinned (Hub `main` on 2026-10-19/20), GPQA, HLE, Paloma and LIMA are gated, three AA pins moved upstream, the as-run corpus was staged in two environments (809 leaves under `datasets` 4.8.5, 305 under the fork's `datasets` 3.6). Compare a rebuild with the as-run per-record digests (`decon eval-digests --compare`). |
| index and bloom | `decon eval-index` | exact given the as-run corpus: rebuilding from the staged as-run corpus reproduces both indexes as multisets and the bloom byte for byte (see Verification). |
| drop-set sample | `decon sample` | needs the internal `deduped_c075` shard layout (274,767 files); not reproducible from the release |
| per-crawl and global lists | `decon dropsets` | exact given the sample (verified on all 127 sources, see Verification) |
| mark, candidate counts (S') | `decon mark`, `decon candidate-counts` | exact; one full pass over the 9.95B documents, then the attribute tables |
| pool and code counts | `decon pool-counts`, `decon code-counts` | exact; pool counts are a second full pass (every document with its crawl); code counts need gated StarCoderData `9fc30b5` |
| D | `decon drop-hashes` | exact from the component tables |
| two-pass filter, manifest | `decon filter-pool`, `decon manifest` | exact (equal to the single pass) |

The decision rule depends on Python's Unicode tables (`str.split`, `str.isalpha`); use CPython 3.12.

## Verification

| Check | Inputs | Result |
|---|---|---|
| Public release, whole crawl | `crawl=CC-NEWS-2016-08` (43,264 docs) | exactly 2 flagged: `f263c271-...` -> `["paloma_4chan"]`, `3cdc9411-...` -> `["paloma_c4_100_domains"]`; 0 mismatches |
| Public release, contaminated files | `crawl=CC-NEWS-2024-05` (52), `crawl=CC-SUPPLEMENTAL-2026-22` (797) | all 849 flagged with equal `decon_matches` |
| Public release, CC-MAIN | `crawl=CC-MAIN-2014-23`: the contaminated file and parts 0-1 of 311 (799,314 docs) | 15,072 flagged, equal to the stored flags and `decon_matches`; 0 mismatches |
| Internal shards (mark) | 4 `deduped_c075` shards, 176,304 docs | `contaminated`, `max_overlap` and `matched_hashes` equal to the as-run attributes for every row (bloom); with S', equal `contaminated` and `matched_hashes` |
| Internal shards (filter) | same | removed ids and matched hashes equal to the as-run sidecars (25 docs) with the bloom and with S'; `decon_matches` equal |
| Drop sets | as-run sample, 127 sources | per-crawl and global lists recomputed from the sample equal the as-run ones for all 127 sources |
| Eval index and bloom | staged as-run corpus (1,147 files, 2,106,579 records) | both rebuilt indexes equal the stored ones as multisets (93,734,507 + 39,935,132 rows); the rebuilt `full-val` bloom is byte-identical (sha256 above) |
| Global | stored tables | D = 66,558 (digest above); global list from the stored counts = stored global list (31,287); every S' hash, every drop hash and all 133,669,639 index rows are bloom members (pure-Python reader and dupekit) |

Reports: `tests/parity/decon/reports/`. Unit tests (`tests/unit/decon/`, offline) pin hashes,
tokenization, the rule's boundaries, eval names, the bloom reader and the drop-set counting; a golden
generated by the as-run Marin code on deterministic synthetic inputs
(`tests/golden/decon/asrun_synthetic.json`) checks decisions, matched hashes and drop-set tables row for
row.
