# Running extraction and classification

`atlantic-cc extract-classify` turns Common Crawl WARCs into the per-WARC files that the rest of the
pipeline (dedup, decontamination, release) reads. It is a port of the worker that built Atlantic-CC
v1.0. For each WARC it:

1. downloads the WARC from `https://data.commoncrawl.org/` and decodes its `text/html` responses;
2. extracts the main content with resiliparse-rs and keeps documents whose fastText P(useful) is at
   least 0.0048 (phase A, on CPUs);
3. scores those survivors with the pooled transformer on up to 8,192 tokens, keeps those at or above
   0.8883, drops those below 0.079, and sends the band in between to the ModernBERT terminal model
   (a fine-tuned Ettin-68M) on the first 2,048 tokens, which keeps those at or above 0.4378 (phase B,
   on an accelerator).

See [extraction.md](extraction.md) (phase A) and [classifiers.md](classifiers.md) (phase B) for the
definition of every step and [reproducibility.md](reproducibility.md) for what "the same output" means for
each of them.

## Outputs

Under `--output` (a local directory or a `gs://` or `s3://` prefix):

| path | contents |
|---|---|
| `kept/data-{warc_hash}.parquet` | kept documents in WARC record order: `doc_id`, `url`, `warc_hash`, `snapshot`, `fasttext_score`, `text`, `n_tokens`, `pooled_prob`, `modernbert_prob` (NaN for documents the pooled model accepted outright). zstd level 12, one row group. Written for every WARC, with zero rows when nothing was kept. |
| `tombstones/data-{warc_hash}.jsonl.gz` | one line per fastText survivor that was not kept: `{"doc_id", "pooled_prob", "modernbert_prob"}`, probabilities rounded to six decimals, `modernbert_prob` null when the pooled model dropped the document. Absent when nothing was dropped. |
| `markers/data-{warc_hash}.json` | written last; the WARC is done when it exists. Status (`ok`, `download_failed`, `abandoned`), phase A and phase B counts, stage times, backend and device. |
| `attempts/data-{warc_hash}.json` | how often the WARC was started. |

`warc_hash` is the first 12 hex digits of the SHA-256 of the WARC's full `s3://commoncrawl/...` URL.
These files have exactly the format production wrote: re-serializing a production kept table with
this code reproduces the production file byte for byte, and tombstones decompress to identical bytes.

## Requirements

- Linux x86_64 and CPython 3.12. Phase A is exact only there ([reproducibility.md](reproducibility.md)); the
  resiliparse-rs extension is built for it.
- The resiliparse-rs extension: either the prebuilt artifact production used, passed as
  `--extractor <prefix>` (unpacked under `--cache-dir`, which must allow executable mappings, so not a
  `noexec` `/tmp`), or a from-source install of the pinned fork commit. See
  [extraction.md](extraction.md).
- The environment: `uv sync --extra tpu` on a TPU VM, `uv sync --extra torch` for GPUs or CPUs,
  `uv sync --extra jax` for JAX on GPU or CPU; add `--extra gcs` or `--extra s3` to read or write
  buckets.
- The three classifiers. By default they come from the Hugging Face Hub
  (`marin-community/atlantic-cc-fasttext`, `-pooled-transformer`, `-modernbert`). `--fasttext-model`,
  `--pooled-model` and `--terminal-model` also take local repo folders and `gs://`/`s3://` folders,
  which are copied once to `--cache-dir`.
- Memory: every phase-A process loads the 0.9 GB fastText model and holds one WARC (about 1 GB
  compressed) plus its decoded HTML; budget about 6 GB per phase-A process plus 4 GB for phase B.

## One machine

```bash
uv sync --extra torch
atlantic-cc extract-classify \
  --warc-list my.warc.paths --output out/ \
  --backend torch --phase-a-procs 4 --extractor /path/to/resiliparse_rs
```

`--warc-list` is a file in Common Crawl's `warc.paths` format (one path per line, optionally
gzipped): bucket-relative paths such as `crawl-data/CC-MAIN-2026-12/segments/.../x.warc.gz` or
`s3://commoncrawl/` URLs. The v1.0 worklist, `v1.0.input.warc.paths.gz`, uses this format. It is not
published yet; `atlantic-cc worklist rebuild` regenerates it from Common Crawl's listings
([worklist.md](worklist.md); digests in [provenance.md](provenance.md)).

Each phase-A process uses `1 + --extract-procs` cores (default 6 extraction processes), so set
`--phase-a-procs` to about `cores / 7`. `--queue-depth` (default 6) caps the WARCs in flight. With
`--backend torch` the models run on `cuda` when a GPU is visible, else on the CPU; a CPU is fine for a
few WARCs but phase B then dominates (minutes per WARC). `scripts/launch/local.sh` loops over shards
and restarts a stalled worker.

## TPUs

Production ran this worker on TPU v4-8, v5p-8, v5e (v5litepod-8/16/32) and v6e hosts with the JAX
backend:

```bash
uv sync --extra tpu --extra gcs
atlantic-cc extract-classify \
  --warc-list gs://bucket/v1.0.input.warc.paths.gz --output gs://bucket/atlantic-cc/ \
  --shard-index 0 --num-shards 32000 \
  --backend jax --phase-a-procs 14 --extract-procs 6 --queue-depth 6 \
  --extractor gs://bucket/artifacts/resiliparse_rs/latest
```

The JAX scorer shards every batch of 32 over the host's chips (8 rows per chip on 4-chip hosts,
4 per chip on 8-chip v5e-8/v6e-8 hosts), as production did. XLA compiles one program per padded
shape, at most seven per model; set `JAX_COMPILATION_CACHE_DIR` to a shared directory to compile
them once per fleet. On a multihost slice, run one independent worker per host (production set
`TPU_PROCESS_BOUNDS=1,1,1` and `TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1` so each process sees only its
own chips) and give each host its own shards. `scripts/launch/iris_tpu.sh` submits blocks of shards
to Marin's Iris cluster.

## GPUs

```bash
uv sync --extra torch --extra s3
atlantic-cc extract-classify --warc-list ... --output s3://bucket/atlantic-cc/ \
  --shard-index $SLURM_ARRAY_TASK_ID --num-shards 1000 \
  --backend torch --torch-device cuda --phase-a-procs 4 --extractor ...
```

`scripts/launch/slurm_gpu.sbatch` runs one Slurm array task per shard and requeues a task whose
phase A stalled. `--backend jax` also runs on GPUs (`uv sync --extra jax` with a CUDA jaxlib).
Neither reproduces production's TPU bf16 numerics bit for bit; expect the decision on roughly
0.05-0.1% of kept documents to flip, each one next to a threshold ([reproducibility.md](reproducibility.md)).

## Sharding

`--shard-index i --num-shards n` processes the WARCs with
`int(sha256(url)[:8], 16) % n == i`, sorted by URL. With `--num-shards 32000`, shard `i` of the
v1.0 CC-MAIN base crawls is exactly production's shard `i`. Production appended the later crawls
(shards 32,000-34,627) round-robin instead; `atlantic_cc.worklist.layout.v1_0_layout` rebuilds those
shard lists if you need production's exact grouping. Sharding never changes outputs: every WARC is
processed on its own.

## Resuming and failures

Rerunning the same command resumes: a WARC with a marker is skipped. Changing `--num-shards` between
runs is safe, because markers are per WARC. Writes are atomic per file, and the marker is written
after the kept and tombstone files, so an interrupted WARC is simply redone.

| event | result | exit status |
|---|---|---|
| the WARC cannot be downloaded after five attempts | marker `download_failed`, zero-row kept file | 0 |
| phase A raises on a WARC (for example, a corrupt gzip member) | no marker; the other WARCs continue | 1 at the end; rerun to retry |
| no WARC finishes phase A for `--stall-minutes` (default 25) | the worker stops | 70; restart it |
| a WARC was started three times without finishing | marker `abandoned`, zero-row kept file | 0 |

The last two rows are production's guard against the few WARCs that hang the extractor or kill
their worker: production abandoned 101 WARCs this way. Restart the worker after exit 70 (Iris and
Slurm do it with retries or `--requeue`; `scripts/launch/local.sh` loops).

## Throughput

Production's catalog for 2,223 WARCs (medians per WARC; a WARC has about 25,000-40,000 HTML
records, of which 55% survive fastText and a third of those reach the terminal model):

| TPU host (4 chips) | phase A, one process | phase B wall | tokenize | pooled | terminal |
|---|---|---|---|---|---|
| v4-8 | 130 s | 36 s | 16 s | 3.8 s | 13 s |
| v5p-8 | 120 s | 18 s | 9 s | 2.2 s | 5.6 s |
| v5e (4 chips per host) | 131 s | 37 s | 17 s | 3.8 s | 14 s |
| v6e (4 chips per host) | 105 s | 59 s | 23 s | 6.4 s | 24 s |

(The v6e row is 14 WARCs that happened to be larger than average.) Per host that is about 5,000-6,800
pooled and 480-700 terminal documents per second; per chip on v6e the pooled model scores about
3,900 documents per second at 8,192 tokens and Ettin about 140 at 2,048 tokens. Phase A runs about
240 HTML records per second per phase-A process. On a v4-8 host phase B takes about a third of the
time phase A takes in one process, so a handful of phase-A processes already keeps the TPU busy;
production ran 14. All of v1.0 (8,582,115 WARCs) took about 45 days on a mixed fleet of these hosts.
GPU throughput is not yet measured; see R3 in [reproducibility.md](reproducibility.md#results).

## Checking a run against production

`tests/parity/r2_phase_b.py` runs the worker on fixture WARCs and compares every fastText survivor's
scores and decision with production's outputs (it needs read access to Marin's buckets):

```bash
python -m tests.parity.r2_phase_b run --fixtures F3 --output OUT --report OUT/r2_report.json -- \
  --backend jax --phase-a-procs 1 --extractor ... --fasttext-model ... --pooled-model ... --terminal-model ...
```

The report gives survivor-set agreement (phase A exactness), |delta p| distributions for both
models, the route confusion matrix, decision agreement, and every flipped document with its
distance to the threshold that decided it, split by whether the device type matches the one
production used for that WARC. `tests/parity/r3_phase_b.py` is the same harness for GPUs.
