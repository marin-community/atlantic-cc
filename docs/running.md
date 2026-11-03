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
- The environment: `uv sync --extra torch` for GPUs or CPUs, `uv sync --extra tpu` on a TPU VM, or
  `uv sync --extra jax` for the JAX backend on a CPU; add `--extra gcs` or `--extra s3` to read or write
  buckets. GPUs have further requirements ([GPUs](#gpus)).
- The resiliparse-rs extension: the as-run wheel, installed with `uv pip install` after the last
  `uv sync` ([extractor.md](extractor.md)). The wheel is not in the lockfile, so `uv sync` removes it;
  `uv run` and `uv sync --inexact` keep it. Alternatively, pass a production-layout artifact directory
  as `--extractor <prefix>`; it is unpacked under `--cache-dir`, which must allow executable mappings
  (not a `noexec` `/tmp`).
- The three classifiers. By default they come from the Hugging Face Hub
  (`marin-community/atlantic-cc-fasttext`, `-pooled-transformer`, `-modernbert`). `--fasttext-model`,
  `--pooled-model` and `--terminal-model` also take local repo folders and `gs://`/`s3://` folders,
  which are copied once to `--cache-dir`.
- Memory: every phase-A process loads the 0.9 GB fastText model and holds one WARC (about 1 GB
  compressed) plus its decoded HTML; budget about 6 GB per phase-A process plus 4 GB for phase B.

## Caches

Besides `--output`, a run writes to four caches under the home directory. Where home is small or a
network volume, as on many clusters, move them to local or scratch space:

| cache | default | move with |
|---|---|---|
| the classifiers and the tokenizer from the Hub (about 1.4 GB) | `~/.cache/huggingface` | `HF_HOME` |
| the unpacked `--extractor` and copies of `gs://`/`s3://` model folders | `~/.cache/atlantic_cc` | `--cache-dir` (must allow executable mappings) |
| a Triton kernel that PyTorch compiles for the first ModernBERT batch on a GPU | `~/.triton/cache` | `TRITON_CACHE_DIR` |
| uv's package cache | `~/.cache/uv` | `UV_CACHE_DIR` |

Triton also builds a small helper module with the node's C compiler on first use, so GPU nodes need
`cc`.

## One machine

```bash
uv run atlantic-cc extract-classify \
  --warc-list my.warc.paths --output out/ --backend torch --phase-a-procs 4
```

`--warc-list` is a file in Common Crawl's `warc.paths` format (one path per line, optionally
gzipped): bucket-relative paths such as `crawl-data/CC-MAIN-2026-12/segments/.../x.warc.gz` or
`s3://commoncrawl/` URLs. The v1.0 worklist, `v1.0.input.warc.paths.gz`, uses this format. It is not
published yet; `atlantic-cc worklist rebuild` regenerates it from Common Crawl's listings
([worklist.md](worklist.md); digests in [provenance.md](provenance.md)).

Each phase-A process uses `1 + --extract-procs` cores (default 6 extraction processes), so set
`--phase-a-procs` to about `cores / 7`. `--queue-depth` (default 6) caps the WARCs in flight. With
`--backend torch` the models run on `cuda` when PyTorch sees a GPU, which must be one the installed
PyTorch supports ([GPUs](#gpus)), and otherwise on the CPU; `--torch-device cpu` forces the CPU. A CPU
is fine for a few WARCs, but phase B then dominates (minutes per WARC). `scripts/launch/local.sh`
loops over shards and restarts a stalled worker.

## TPUs

Production ran this worker on TPU v4-8, v5p-8, v5e (v5litepod-8/16/32) and v6e hosts with the JAX
backend:

```bash
uv sync --extra tpu --extra gcs
uv run atlantic-cc extract-classify \
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

The PyTorch backend runs on one GPU per worker:

```bash
uv sync --extra torch --extra s3
uv pip install resiliparse_rs_atlantic_cc-0+850891b-cp312-cp312-manylinux_2_34_x86_64.whl   # after every uv sync
uv run atlantic-cc extract-classify --warc-list ... --output s3://bucket/atlantic-cc/ \
  --shard-index $SLURM_ARRAY_TASK_ID --num-shards 1000 \
  --backend torch --torch-device cuda --phase-a-procs 4
```

- **Compute capability 7.5 or newer** (NVIDIA Turing and later), with a driver for CUDA 13 (580.82 and
  610.43 worked). The locked PyTorch, 2.15.0+cu132, has kernels only for these GPUs. On older ones
  `torch.cuda.is_available()` is still true, but every batch fails with "no kernel image is available".
- **Turing** (compute capability 7.5) has no bf16 hardware, so PyTorch emulates the default
  `--precision bf16`: the ModernBERT model runs about 10 times slower than on an A100, and half as fast
  as in fp32. On 11 GB cards such as the RTX 2080 Ti, bf16 runs out of memory; use `--precision fp32`,
  which agrees with production slightly less often (99.941% of decisions against 99.949%).
- **Pascal and Volta** (6.x and 7.0) need a PyTorch build with their kernels, installed by hand, for
  example 2.14.1 for CUDA 12.6 (2.15.0 has no CUDA 12 build). Run it with `uv run --no-sync` or
  `UV_NO_SYNC=1`, or uv restores the locked build:

  ```bash
  uv pip install --reinstall-package torch 'torch==2.14.1+cu126' --index-url https://download.pytorch.org/whl/cu126
  UV_NO_SYNC=1 uv run atlantic-cc extract-classify ...
  ```

No GPU reproduces production's TPU bf16 numerics bit for bit. In R3 ([reproducibility.md](reproducibility.md#results)),
bf16 runs on five GPU models flipped about 0.1% of kept documents (0.05% of fastText survivors), all
of them ModernBERT decisions within 0.015 of its threshold.

`scripts/launch/slurm_gpu.sbatch` runs one Slurm array task per shard and requeues a task whose
phase A stalled. Submit it from the repository root:

```bash
mkdir -p logs                    # older Slurm versions do not create the log directory
export OUTPUT=/scratch/atlantic-cc WARC_LIST=v1.0.input.warc.paths.gz NUM_SHARDS=1000
export CACHE_DIR=/scratch/cache/atlantic_cc HF_HOME=/scratch/cache/hf TRITON_CACHE_DIR=/scratch/cache/triton
sbatch --array=0-999%64 -A ACCOUNT -p PARTITION scripts/launch/slurm_gpu.sbatch
```

The script passes `CACHE_DIR` as `--cache-dir`, and `EXTRACTOR` as `--extractor` when it is set;
without it, the worker uses the installed wheel. To keep older GPUs off the job, add a constraint such
as `--constraint="ampere|ada|hopper"` (feature names vary by cluster).

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

Every start of a WARC is counted in `attempts/` before its work begins, so that a WARC that hangs the
extractor or kills its worker is eventually given up (production abandoned 101 WARCs this way). A
broken setup must not use up those starts: before counting any, the worker checks that a phase-A
process can load the fastText model and extract a test page.

| event | result | exit status |
|---|---|---|
| phase A cannot start (for example, resiliparse-rs is not installed) | the worker stops before starting any WARC | 1; fix the setup and rerun |
| phase B raises (for example, CUDA out of memory), or an output cannot be written | the worker stops; unfinished WARCs stay unmarked, and this run's starts of them are not counted | 1; fix the setup and rerun |
| phase A raises on a WARC (for example, a corrupt gzip member) | no marker; the other WARCs continue | 1 at the end; rerun to retry |
| the WARC cannot be downloaded after five attempts | marker `download_failed`, zero-row kept file | 1 at the end |
| no WARC finishes phase A for `--stall-minutes` (default 25) | the worker stops | 70; restart it |
| a WARC was started three times without finishing | marker `abandoned`, zero-row kept file | 1 at the end |

The last lines of the log of a run that exits 1 say why: the setup error, or how many WARCs the run
abandoned, could not download or saw raise. Restart the worker after exit 70 (Iris and Slurm do it
with retries or `--requeue`; `scripts/launch/local.sh` loops). A start also counts when a phase-A
process dies (for example, killed for lack of memory) or the worker is killed or interrupted; if such
starts came from a problem you have since fixed, delete `attempts/` before rerunning. To retry WARCs
marked `download_failed` or `abandoned`, delete their files under `markers/` and `attempts/`.

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

GPUs, measured on R3's five fixture WARCs (8,010-20,550 HTML records each; F1-F4 on the RTX A5000).
Medians per WARC, in bf16 unless noted; documents per second leave out each run's first WARC, which
includes CUDA start-up and the Triton compile:

| GPU (CPUs, phase-A processes) | phase A | tokenize | pooled | terminal | pooled docs/s | terminal docs/s |
|---|---|---|---|---|---|---|
| A100-SXM4-40GB (16, 2) | 49 s | 2.7 s | 1.9 s | 7.6 s | 5,220 | 429 |
| RTX 6000 Ada (12, 2) | 43 s | 8.3 s | 1.9 s | 10.0 s | 5,950 | 326 |
| RTX A5000 (16, 2) | 48 s | 2.6 s | 3.4 s | 19.6 s | 4,320 | 228 |
| Titan RTX (8, 2) | 68 s | 3.4 s | 2.9 s | 90.7 s | 4,270 | 35 |
| Titan RTX, fp32 (8, 2) | 64 s | 3.5 s | 1.8 s | 46.9 s | 5,550 | 70 |
| RTX 2080 Ti, fp32 (4, 1) | 74 s | 4.0 s | 2.0 s | 51.2 s | 4,210 | 64 |
| V100-SXM2-32GB, torch 2.14.1+cu126 (12, 2) | 78 s | 5.2 s | 3.3 s | 69.3 s | 2,920 | 46 |

An A100 or an RTX 6000 Ada does about a fifth of a TPU v4-8 host's phase-B work, and still scores a
WARC's survivors in 10-25 s, so on these GPUs phase A is the bottleneck (160-290 HTML records per
second per phase-A process); give a node about `cores / 7` phase-A processes. On Turing and Volta the
terminal model is 10-12 times slower than on an A100, and phase B becomes the bottleneck.

## Checking a run against production

`tests/parity/r2_phase_b.py` runs the worker on fixture WARCs and compares every fastText survivor's
scores and decision with production's outputs (it needs read access to Marin's buckets):

```bash
uv run python -m tests.parity.r2_phase_b run --fixtures F3 --output OUT --report OUT/r2_report.json -- \
  --backend jax --phase-a-procs 1 --extractor ... --fasttext-model ... --pooled-model ... --terminal-model ...
```

The report gives survivor-set agreement (phase A exactness), |delta p| distributions for both
models, the route confusion matrix, decision agreement, and every flipped document with its
distance to the threshold that decided it, split by whether the device type matches the one
production used for that WARC. `tests/parity/r3_phase_b.py` is the same harness for GPUs. On a GPU
host without access to Marin's buckets, run `atlantic-cc extract-classify` on the fixtures there,
copy the output tree to a bucket in us-central2, where production's outputs are, and run
`r3_phase_b compare` on it.
