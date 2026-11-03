# The input worklist

Atlantic-CC v1.0 was built from **8,582,115 Common Crawl WARC files**: every WARC of an explicit list of
249 crawls. No WARC was sampled or skipped. This page describes the list, how to rebuild it from Common
Crawl's public listings, and the files that pin it.

## What is in it

| Part | Crawls | WARCs |
|---|---|---:|
| CC-MAIN, base | the 119 crawls with a WARC listing on 2026-04-10: `CC-MAIN-2013-20` … `CC-MAIN-2026-12` | 7,925,398 |
| CC-MAIN, late | `CC-MAIN-2026-17`, `-21`, `-25`, `-30`, `-34`, `-39` (100,000 each) | 600,000 |
| CC-SUPPLEMENTAL | `CC-SUPPLEMENTAL-2026-22`, Common Crawl's science-focused crawl | 4,573 |
| CC-NEWS | 123 months, `CC-NEWS-2016-08` … `CC-NEWS-2026-10` | 52,144 |
| **Total** | **249 crawls** | **8,582,115** |

The crawl list is `atlantic_cc.worklist.crawls.V1_0_CRAWLS`. Two details matter when rebuilding it:

- **"All crawls" is not reproducible.** `CC-MAIN-2026-43` appeared after the list was frozen and is not
  included, nor are the pre-WARC crawls `CC-MAIN-2008-2009`, `CC-MAIN-2009-2010` and `CC-MAIN-2012`.
  Always use the explicit list.
- **The October 2026 CC-NEWS month is cut.** CC-NEWS listings grow during their month. v1.0 includes
  October 2026 up to and including `crawl-data/CC-NEWS/2026/10/CC-NEWS-20261015151252-00272.warc.gz`,
  the first 179 lines of the listing (which had 369 lines on 2026-11-02). Every other month is whole.

### Why CC-SUPPLEMENTAL comes from the frozen list

Common Crawl publishes no `warc.paths.gz` for `CC-SUPPLEMENTAL-2026-22`. The WARCs themselves are public
under `projects/cc-open-athena-test/CC-SUPPLEMENTAL-2026-22/segments/<segment>/warc/`, but none of
`projects/cc-open-athena-test/CC-SUPPLEMENTAL-2026-22/warc.paths.gz`, `…/warc.paths`,
`projects/cc-open-athena-test/warc.paths.gz` or `crawl-data/CC-SUPPLEMENTAL-2026-22/warc.paths.gz` exists
(all return HTTP 404). Its 4,573 paths therefore ship with this package
(`src/atlantic_cc/worklist/data/v1.0.supplemental.warc.paths.gz`, 13 KB) as part of the frozen list.

## Files

| File | Where | Content |
|---|---|---|
| `v1.0.input.warc.paths.gz` | release asset (23.8 MB) | The frozen list in Common Crawl's paths-file format: one bucket-relative path per line, sorted. |
| `data/v1.0.crawls.tsv` | this package | Per crawl: WARC count and sha256 of its sorted paths. |
| `data/v1.0.supplemental.warc.paths.gz` | this package | The CC-SUPPLEMENTAL-2026-22 paths. |
| `v1.0.input.empty_warcs.tsv.gz` | release asset (96 KB) | The 5,978 WARCs that produced no documents (see below). |
| `v1.0.input.warc_catalog.parquet` | release asset (178 MB) | Production's per-WARC counters (see below). |

The release assets are not published yet.

**Digests.** The list's identity is the sha256 of its *uncompressed* text (`"\n"`-terminated lines), because
gzip bytes vary with the zlib build:

| | Value |
|---|---|
| WARCs | 8,582,115 |
| sha256 (uncompressed) | `101505dd737241a05afb03338b6e2db6175a40651d57538a7c4db0fd4576961a` |
| sha256 of the published `.gz` | `75fb0936859373ab2ac29aa548ad35fe067a3dbc706b4b71ae13195ea4b71d7a` |
| production shard layout, sha256 of `"<shard>\t<path>\n"` | `7dfd37b8722b7f398c1e061d8368e33bbaee927112b825b402a9a73cb54726b9` |

## Commands

```bash
# Rebuild the v1.0 list from Common Crawl's public listings (about 30 MB of downloads) and check it
# crawl by crawl and as a whole against the pinned digests.
atlantic-cc worklist rebuild --out v1.0.input.warc.paths.gz

# Check a downloaded copy of the list (digest, then crawl by crawl against the public listings).
atlantic-cc worklist verify --worklist v1.0.input.warc.paths.gz

# Any other explicit crawl list, e.g. a crawl released after v1.0 (no digests to check).
atlantic-cc worklist build --crawls CC-MAIN-2026-43 --out cc-main-2026-43.paths.gz
atlantic-cc worklist build --crawls CC-NEWS-2026-11 \
    --news-last-warc crawl-data/CC-NEWS/2026/11/CC-NEWS-20261115000000-00123.warc.gz --out news.paths.gz

# Production's shard layout as Parquet (shard, warc_path, warc_hash).
atlantic-cc worklist layout --worklist v1.0.input.warc.paths.gz --out v1.0.layout.parquet
```

In Python, `atlantic_cc.worklist.frozen.load_frozen(path_or_url)` returns the 8,582,115 bucket-relative
paths and raises unless the file is exactly the frozen list.

## Identity: `warc_hash` and crawl names

- **`warc_hash = sha256("s3://commoncrawl/" + path)[:12]`** (12 hex characters,
  `atlantic_cc.warc.identity.warc_hash`). The hash covers the full `s3://commoncrawl/…` URL, not the
  bucket-relative path stored in the worklist and the release, and the function hashes whatever string
  it gets: call it as `warc_hash(to_s3_url(path))` (`atlantic_cc.worklist.identity.to_s3_url`).
  Per-WARC outputs are named `kept/data-<warc_hash>.parquet`.
- **Crawl name** (`atlantic_cc.warc.identity.snapshot_of`): `CC-NEWS-YYYY-MM` for
  `crawl-data/CC-NEWS/YYYY/MM/…`, else the first `CC-MAIN-YYYY-WW` or `CC-SUPPLEMENTAL-YYYY-WW` in the
  path. This is the release's `crawl` column.

## Production's shard layout

Shards were the unit of work during extraction (34,628 shards). They change no document; the release
build uses them only for the row order of an internal lookup file. The layout
(`atlantic_cc.worklist.layout.v1_0_layout`) is:

- shards 0–31,999: the 119 base crawls, `shard = int(sha256(url)[:8], 16) % 32000`;
- then three extensions, each sorted by URL and dealt round-robin into `ceil(n / 250)` new shards:
  CC-SUPPLEMENTAL-2026-22 → shards 32,000–32,018, the six late CC-MAIN crawls → 32,019–34,418,
  CC-NEWS → 34,419–34,627.

Each shard lists its WARCs sorted. Rebuilding the layout from the frozen list reproduces all 34,628
production shards exactly.

## WARCs without output

5,978 WARCs (0.07%) produced no documents, in 28 crawls concentrated in 2014–2015 (CC-MAIN-2015-27:
1,535; 2015-14: 1,156; 2015-11: 963; 2015-18: 617; 2014-41: 492). `v1.0.input.empty_warcs.tsv.gz` lists
them with columns `warc_path`, `warc_hash`, `shard`, `http_status`, `bytes` from a final HEAD audit:

| `http_status` | WARCs | Meaning |
|---:|---:|---|
| 200 | 4,006 | Reachable. 3,905 are header-only files (427–459 bytes); the other 101 (0.76 GB and up) hung the extractor or killed its worker on every attempt and were abandoned after 3 tries. |
| 403 | 1,422 | Refused by data.commoncrawl.org (`bytes` = -1). |
| 503 | 550 | Throttled at audit time (`bytes` = -1). |

A reproduction will meet the same empty files.

## Per-WARC catalog

`v1.0.input.warc_catalog.parquet` has one row per WARC (8,582,115 rows, sorted by `warc_path`) with
production's per-WARC counters, to compare a rerun WARC by WARC:

| Column | Meaning |
|---|---|
| `shard`, `warc_path`, `warc_hash` | identity |
| `a_n_in` | HTML response records read |
| `a_n_prefilter_dropped`, `a_n_extract_empty`, `a_n_extract_crashed` | records dropped before or during extraction |
| `n_presurvivors` | fastText survivors |
| `n_hi_accept`, `n_band` | pooled-model accepts (≥ 0.8883) and band documents sent to the terminal ModernBERT classifier |
| `n_kept` | documents kept by the cascade (before deduplication) |
| `device`, `n_devices` | the accelerator that scored the WARC (e.g. `TPU v6 lite`) |

The `a_*` counters are null for shards 0–403 (99,638 WARCs), which ran an earlier two-phase build whose
catalog did not record them. Because classifier scores came from several TPU generations, a rerun
matches `n_presurvivors` exactly but `n_kept` only statistically ([reproducibility.md](reproducibility.md)).

Totals over the catalog:

| Quantity | Value |
|---|---:|
| HTML responses read, `a_n_in` (8,482,477 WARCs with phase-A counters) | 322,798,446,328 |
| dropped by the prefilter / extracted empty / extractor crashes | 1,538,604,933 / 5,955,413,543 / 216,793 |
| fastText survivors, `n_presurvivors` (all WARCs) | 184,628,377,721 (56.5% of `a_n_in` where both are known) |
| pooled accepts / band documents | 33,804,891,712 / 63,789,101,516 |
| kept, `n_kept` | 61,211,321,797 (18.7% of `a_n_in` where both are known) |
| WARCs with `n_kept = 0` | 6,057 |
| `device` | TPU v5 lite 2,329,442; TPU v6 lite 2,154,124; TPU v5 2,083,848; TPU v4 2,014,701 WARCs |

The `n_kept` total is 37 higher than the 61,211,321,760 rows deduplication read from `kept/`. The
catalog holds one row per WARC, but 76 WARCs were processed twice (in different regions) and only one
kept file survived consolidation; for 29 of them the catalog row comes from the other run (differences
of -8 to +8 rows, net -3). The remaining +40 is not located; finding it
would mean reading all 8,582,115 `kept/` footers.

## Order effects downstream

Deduplication keeps, for each group of exact duplicates, the copy with the smallest
`(file index, row)` key, where files are the `kept/data-<warc_hash>.parquet` outputs sorted by name.
Which crawl an exact duplicate is filed under is therefore decided by `warc_hash` order, not by crawl date.

## How the frozen list was produced

`atlantic-cc worklist export-asrun --what worklist` read production's 34,628 worklist shards
(`warc_path` as `s3://commoncrawl/…`, `warc_hash`), checked every hash and the index's per-shard
counts, checked that `v1_0_layout` reproduces every shard, and wrote the sorted list. The result was
then compared with a rebuild from the public listings (`worklist verify`): all 249 crawls and the whole
list are identical.
