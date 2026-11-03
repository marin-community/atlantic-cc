# Release format

Atlantic-CC v1.0 is published by Common Crawl. On this page `$ROOT` is the release root, currently
`https://data.commoncrawl.org/projects/atlantic-cc` (no credentials needed; also
`s3://commoncrawl/projects/atlantic-cc`). It holds every document
that survived global deduplication: 9,951,767,085 records, of which 9,950,417,677 are clean and 1,349,408
matched an evaluation set during decontamination and are kept in separate `contaminated-*` files.

Common Crawl hosts and documents the data release, including its terms of use. This page describes the
files as this code writes, verifies and reads them. The documents were selected by the classifiers
[`marin-community/atlantic-cc-fasttext`](https://huggingface.co/marin-community/atlantic-cc-fasttext),
[`marin-community/atlantic-cc-pooled-transformer`](https://huggingface.co/marin-community/atlantic-cc-pooled-transformer)
and [`marin-community/atlantic-cc-modernbert`](https://huggingface.co/marin-community/atlantic-cc-modernbert)
(see [classifiers.md](classifiers.md)).

## Layout

```
$ROOT/
  v1.0.manifest.json                 37,690 B   per-crawl record and file counts
  v1.0.parquet.paths.gz              27,985 keys: every data file
  v1.0.clean.parquet.paths.gz        27,736 keys: part-* files only
  v1.0.checksums.md5                 md5sum of every file above and below (not yet published)
  v1.0/crawl=<crawl>/part-NNNNN-of-MMMMM.parquet          clean records, decon_matches == []
  v1.0/crawl=<crawl>/contaminated-00000-of-00001.parquet  flagged records, decon_matches != []
```

- **249 crawl partitions**: 125 CC-MAIN crawls (`CC-MAIN-2013-20` … `CC-MAIN-2026-39`), 123 CC-NEWS months
  (`CC-NEWS-2016-08` … `CC-NEWS-2026-10`) and `CC-SUPPLEMENTAL-2026-22`. Every crawl has exactly one
  contaminated file. `NNNNN` counts from 0 and `MMMMM` is the crawl's number of part files, so a crawl's
  files can be listed from the manifest alone.
- The manifest and paths files sit **next to** `v1.0/`, not inside it, so a hive reader of `v1.0/` sees
  only `crawl=` partitions. Paths files use Common Crawl's format (gzipped, one bucket key per line); they
  list crawls in sorted order and each crawl's contaminated file before its parts.

| Family | Crawls | Clean records | Clean files | Contaminated records | Bytes (all files) |
|---|---:|---:|---:|---:|---:|
| CC-MAIN | 125 | 9,741,311,283 | 27,345 | 1,325,408 | 22.529 TB |
| CC-NEWS | 123 | 205,347,765 | 380 | 23,203 | 232.4 GB |
| CC-SUPPLEMENTAL | 1 | 3,758,629 | 11 | 797 | 8.39 GB |
| **Total** | **249** | **9,950,417,677** | **27,736** | **1,349,408** | **22,769,659,891,232 B** |

Part files are 5.9 MB to 1,017 MB (median 822 MB). Contaminated files are 24 KB to 196 MB, 15.2 GB in
total. The largest crawl is `CC-MAIN-2013-20` (168.2M clean records, 410 parts).

## Schema

Every file has these eleven columns (Parquet written by `parquet-cpp-arrow 23.0.1`, zstd level 9,
10,000-row row groups):

| Column | Arrow type | Content |
|---|---|---|
| `record_id` | string | The WARC-Record-ID as a bare UUID (`<urn:uuid:…>` stripped) |
| `url` | string | WARC-Target-URI |
| `crawl` | string | `CC-MAIN-YYYY-WW`, `CC-NEWS-YYYY-MM` or `CC-SUPPLEMENTAL-2026-22`; equals the partition |
| `warc_path` | string | Bucket-relative WARC path, e.g. `crawl-data/CC-NEWS/2016/08/CC-NEWS-20160827145159-00003.warc.gz` |
| `text` | string | resiliparse-rs light Markdown; runs of more than 128 whitespace characters are cut to 128 |
| `fasttext_score` | double | fastText P(useful) (a float32 score widened to double) |
| `pooled_prob` | double | Pooled-transformer P(useful) |
| `modernbert_prob` | double | P(useful) from the ModernBERT terminal classifier (fine-tuned Ettin-68M); null when `pooled_prob ≥ 0.8883` (the document skipped the terminal model) |
| `exact_copies` | int64 | Number of exact copies (after whitespace compaction) in the pre-dedup pool, ≥ 1 |
| `fuzzy_cluster_size` | int64 | Size of the document's MinHash connected component, ≥ 1 (4,557,202,931 marks the giant chained component) |
| `decon_matches` | list\<string\> | Evaluation sets the document matched; empty in `part-*` files |

Readers see `decon_matches` as `list<element: string>` (Parquet's list naming). Records carry no WARC
byte offsets: finding a record's HTML means streaming its WARC and matching `record_id`.

## Sort order

Within a crawl and kind, records are grouped by source WARC, WARCs appear in ascending `warc_hash` order
(`sha256("s3://commoncrawl/" + warc_path)[:12]`), and each WARC keeps its records in WARC order. A 10,000-row
row group holds about 10 WARCs (4.5 to 20, median 9.5, over the 39 part files sampled by R7), which makes
WARC-level filtering cheap (see `atlantic-cc release subset`).
Records are not sorted by URL or score.

## Reading

```python
import fsspec
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT = "https://data.commoncrawl.org/projects/atlantic-cc"  # the release root

# One file over HTTPS (no credentials).
url = f"{ROOT}/v1.0/crawl=CC-NEWS-2016-08/part-00000-of-00001.parquet"
with fsspec.open(url, "rb") as f:
    table = pq.read_table(f, columns=["record_id", "url", "text"])

# A local copy (atlantic-cc release download ...) as one hive-partitioned dataset.
dataset = ds.dataset("atlantic-cc/v1.0", format="parquet", partitioning="hive")
news = dataset.to_table(filter=ds.field("crawl") == "CC-NEWS-2016-08")
```

These examples were run against the published files.

**Do not call `pyarrow.parquet.read_table(path)` on a release path without `partitioning=None`.** Its
default hive discovery turns `crawl=…` into a dictionary-typed `crawl` field that cannot be merged with
the string `crawl` column, and it raises `ArrowTypeError: Unable to merge: Field crawl has incompatible
types`. `pyarrow.dataset.dataset(..., partitioning="hive")`, `pq.ParquetFile(path).read()` and
`pq.read_table(path, partitioning=None)` all work. DuckDB reads the tree with
`read_parquet('atlantic-cc/v1.0/*/*.parquet', hive_partitioning = true)`.

To skip contaminated documents, read only `part-*` files (e.g. the keys in `v1.0.clean.parquet.paths.gz`)
or filter `decon_matches` to empty lists.

## Manifest

`v1.0.manifest.json` (written with `json.dump(..., indent=1)`):

```json
{
 "version": "v1.0",
 "records": 9951767085,
 "by_kind": {"part": 9950417677, "contaminated": 1349408},
 "crawls": {"CC-MAIN-2013-20": {"part": {"records": 168238886, "files": 410},
                                "contaminated": {"records": 20534, "files": 1}}, "...": {}},
 "decontamination": {"method": "Marin decontamination v4 (hero run rule): ...", "run": "gs://..."}
}
```

`decontamination.run` names the internal decontamination run the totals were checked against.

## Checksums

`v1.0.checksums.md5` is `md5sum` output for all 27,988 published objects (the data files, the manifest
and both paths files), with paths relative to `$ROOT/`, sorted:

```
85731ddfa63d77d7d2a1c222f495c7ab  v1.0.clean.parquet.paths.gz
70ef1582aed844fa159ff3452adfdfc0  v1.0.manifest.json
1fb4e3bb6f2973a599016ee4bfd4b8a0  v1.0.parquet.paths.gz
...
```

It was built from the upload records without re-reading data: each object was uploaded in a single PUT,
so its S3 ETag is the MD5 of its bytes (the HTTPS `ETag` header shows the same value). All 27,988 MD5s
and sizes were cross-checked against the build's object-store metadata. sha256 of the file:
`a44350759eaecbd543b7fcc420f9b247dc405f4d84090df36f0e55f1419118f8`. It is not yet published.

From a downloaded copy: `cd atlantic-cc && md5sum -c v1.0.checksums.md5` (or `atlantic-cc release verify`).
Without the file, `atlantic-cc release download` still checks every file against its `ETag`.

## Tools

```bash
# Download crawls (resumable; each file is checked against its MD5 before it is kept).
atlantic-cc release download --dest atlantic-cc --crawls CC-NEWS-2016-08 CC-MAIN-2024-10
atlantic-cc release download --dest atlantic-cc --all          # 22.8 TB

# Check a copy: files present, footer row counts vs the manifest, MD5s vs the checksums file.
# Over HTTPS or s3:// the MD5 comes from the ETag, so nothing is downloaded.
atlantic-cc release verify --root atlantic-cc --checksums v1.0.checksums.md5
atlantic-cc release verify --root $ROOT \
    --checksums v1.0.checksums.md5 --crawls CC-NEWS-2016-08

# The documents of a list of WARCs (bucket-relative paths or s3:// URLs), straight from HTTPS:
# per-WARC counts, optional token counts, optional Parquet export of the clean documents.
atlantic-cc release subset --warcs my_warcs.txt --output subset_out --workers 16 \
    [--tokenizer meta-llama/Meta-Llama-3.1-8B] [--export subset_out/docs]
atlantic-cc release subset --warcs my_warcs.txt --output subset_out --merge
```

## Building the release

The release is regrouped from the deduplication output (`deduped_c075/*.jsonl.gz`) and the
decontamination run (`removed_docs.parquet`, `export.json`) in four resumable steps
(`atlantic_cc.release.regroup`, a port of production's `release_regroup.py`):

```bash
atlantic-cc release warc-paths --worklist v1.0.input.warc.paths.gz --stage STAGE
atlantic-cc release pieces --source DEDUP/deduped_c075 --decon-run DECON_RUN --stage STAGE --workers 8
atlantic-cc release parts --stage STAGE --out OUT --workers 8          # --part-mb 1000 --zstd-level 9
atlantic-cc release manifest --stage STAGE --out OUT --decon-run DECON_RUN [--gzip-mtime 1793254392]
atlantic-cc release checksums --done-markers UPLOAD_MARKERS --out OUT/v1.0.checksums.md5
```

1. `warc-paths` maps `<crawl>/<warc_hash>` to each WARC's path (dedup records carry only the hash).
2. `pieces` reads the dedup files in sorted order, 50 per batch, and streams each batch into one piece per
   `(crawl, kind)`, written 10,000 rows at a time with pyarrow's default zstd level. Records the
   decontamination filter removed get `decon_matches` = their matched evals and go to `kind=contaminated`;
   the step fails if a batch tags a different number of documents than the filter removed. A
   `_done/batch-NNNNNN.json` marker, written last, makes the step resumable. `--shard I/N` splits the
   batches across machines. Each worker holds the 8.6M-entry path map (about 3 GB) plus up to 10,000
   buffered rows per crawl, so budget up to 32 GB per worker.
3. `parts` lists the pieces with their sizes once, then packs each crawl's pieces of one kind, in batch
   order, closing a part as soon as its pieces reach `--part-mb` × 10⁶ bytes. Because pieces are
   compressed with a faster zstd level than the parts, parts end up at about 0.8 GB. The plan is written
   to `parts_plan.json` before any part, so every `-of-MMMMM` is known up front; existing parts are skipped.
4. `manifest` reads every part footer, checks the clean and contaminated totals against the
   decontamination run's `export.json`, and writes the manifest and paths files. Production stamped the
   paths files' gzip header with the time of the run; `--gzip-mtime 1793254392` reproduces v1.0's bytes.

Part boundaries depend on piece byte sizes, which depend on the pyarrow version (pin 23.0.1); record
content and order do not.

## Versioning

Releases are append-only: a published `v1.0/` file never changes. A corrected or extended release is a new
version (`v1.1/`, …) with its own manifest, paths files and checksums next to the existing ones; readers
pin a version by its prefix.

## Reproduction results

Run on Marin's cluster (us-central2) against production's intermediate and published files with
`tests/parity/release/r6_regroup.py` and `r7_consistency.py` (maintainers only: they read Marin's
buckets). The JSON reports are in `tests/parity/release/reports/`.

| Test | Result |
|---|---|
| R6 plan | Listing production's 1,292,989 pieces (28.3 TB) and re-planning them gives a `parts_plan.json` byte-identical to production's: 27,985 parts over 249 crawls. |
| R6 manifest | The manifest rebuilt from the 27,985 part footers (9,951,767,085 records: 9,950,417,677 clean, 1,349,408 contaminated, equal to the decontamination run's totals) and both paths files (with v1.0's gzip time) are byte-identical to the published files. |
| R6 parts | Four files rebuilt from production's pieces are byte-identical to the published ones: CC-NEWS-2016-08 part 0 (43,262 records from 8 pieces) and its contaminated file (2 records), CC-SUPPLEMENTAL-2026-22 part 10 (136,429 records, 165 pieces) and CC-MAIN-2013-20 part 0 (419,226 records, 9 pieces). |
| R6 pieces | `warc_paths.parquet` rebuilt from the frozen worklist (8,582,115 rows), `decon_lookup.parquet` rebuilt from `removed_docs.parquet` (1,349,408 rows), and all 222 pieces of batch 3,140 rebuilt from its 50 dedup files (1,473,489 records, 208 contaminated, 132 clean and 90 contaminated pieces) are byte-identical to production's; the done marker lists the same files, totals and pieces. |
| R7 | 820,897 published records (every row of CC-NEWS-2016-08 and of the CC-SUPPLEMENTAL-2026-22 contaminated file, plus 2 random row groups of 40 random files; 43 files, 35 crawls, 1,633 WARCs) all exist in their WARC's cascade output with the same `url`, crawl and scores (`modernbert_prob` NaN → null) and the same text after dedup's whitespace compaction; 238 of them (0.029%) were changed by that compaction. `xxh3_128(text)` is unique across the sample, every clean record has empty and every contaminated record non-empty `decon_matches`, and in every row group each WARC's records are contiguous with WARCs in ascending `warc_hash` order. The manifest totals are covered by R6 manifest. |
| Checksums | The 27,988 upload done-markers cover exactly the published objects; every ETag equals the object's MD5 and every size its size in the build bucket (22,769,659,891,232 data bytes). |
| Public copy | `atlantic-cc release verify --root https://data.commoncrawl.org/projects/atlantic-cc` on CC-NEWS-2016-08, CC-NEWS-2026-10 and CC-SUPPLEMENTAL-2026-22: all 16 files present, ETags equal to `v1.0.checksums.md5`, footer row counts equal to the manifest. |
