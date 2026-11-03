# Deduplication

Atlantic-CC v1.0 is deduplicated globally: exact duplicates first, then near-duplicates found with
MinHash/LSH and resolved by a containment rule ("c075"). Of 61,211,321,760 kept documents from
8,582,115 WARCs, 9,951,767,085 survived (72.7T characters). Every survivor records how many exact copies
it had and how large its near-duplicate component was.

This page defines the algorithm and its parameters, states what can be reproduced exactly, and shows how
to run and verify a dedup. Code: `src/atlantic_cc/dedup/` (a port of Marin's `rdedup`, the code that
built the release, with only imports, storage and orchestration replaced). Command: `atlantic-cc dedup`.

## The algorithm

The input is a directory of `kept/data-{warc_hash}.parquet` files, one per WARC, as written by
[`atlantic-cc extract-classify`](running.md).

1. **Document keys.** The kept files are sorted by full path. A document's key is
   `file_index << 24 | row`, where `row` is its parquet row (WARC record order). Keys decide which copy
   of an exact duplicate survives, so the sorted path list is part of the input.
2. **Normalization.** A document is skipped if its text is missing or blank after Python's `str.strip()`
   (so a text of only U+3000 or `\x1c` is skipped). Every run of more than 128 whitespace characters
   (Python's `re` `\s`) is cut to its first 128 characters, keeping the original characters. The output
   carries this compacted text.
3. **Exact dedup.** The content id is `xxh3_128` of the compacted UTF-8 text. Among documents with the
   same id, the one with the smallest key survives: the copy in the file that sorts first, then the
   lowest row. Its provenance fields (`doc_id`, `url`, `warc_hash`, `snapshot`, scores) come from that
   copy. `exact_copies` is the number of input documents with that id (at least 1).
4. **MinHash and LSH.** Each unique document goes through dupekit's `CleanText` (Rust lowercase, ASCII
   punctuation removed, whitespace collapsed), then MinHash over character 5-grams with 286 permutations
   (seed 42), then 26 LSH bands of 11 rows. A bucket is `xxh3_64` of the band's 11 values; the band
   index is not mixed in, so equal slices in different bands collide.
5. **Connected components.** Two documents are linked when they share any bucket. Components are exact
   (FastSV label propagation run to convergence; any exact algorithm gives the same components). A
   component's root is its member whose key is smallest in the order `key * 0x9E3779B97F4A7C15 mod 2^64`.
   `fuzzy_cluster_size` is the number of unique documents in the component (1 outside any component).
6. **Containment rule (c075).** Each component is solved on its own. A member is removed when a longer
   surviving member of the same group contains at least 75% of its distinct word 3-grams:
   - Word 3-grams come from `text.casefold().split()` (Python 3.12), joined by single spaces and hashed
     with `xxh3_64`. A text with 1-2 tokens is one shingle; a text without tokens is never removed and
     never removes anything.
   - Members are processed from longest to shortest (code points; on equal length the smaller content id
     goes first). A member's candidates are the earlier, not yet removed members; it is removed by the
     first candidate with `|N(m) ∩ N(c)| / |N(m)| >= 0.75`. A removed member never removes another.
   - Solve batches of at most 256 members compare every candidate. Larger batches use an inverted index
     over the batch's 3-grams: the member's 32 rarest 3-grams (among those held by at most 512 documents,
     when any are) probe it, and the 32 earlier members sharing the most probes become its candidates;
     the cap applies before removed members are skipped. See [Exactness](#exactness) for why this path
     depends on the CPU.
   - Components of more than 100,000 members are split before solving: into
     `ceil(size / 100,000)` splits by a word-5-gram MinHash of the text, each cut into 16 sub-groups by
     content id. Documents in different sub-groups are never compared.
   - Limits: text is cut to 64 MiB code points before grouping and to 8 MiB before solving; a group is
     solved in batches of at most 256 MiB of text (documents in different batches are never compared).
7. **Output.** The survivors are written as JSON lines, one gzip member per input kept file (level 6,
   mtime 0), 32 kept files per output file.

The 8M run: 61,211,321,760 documents in, 39,375,554,171 exact duplicates, 21,835,767,589 unique
documents, of which 18,498,947,789 were in 2,394,062,201 components (the largest has 4,557,202,931
members). c075 removed 11,884,000,504 and kept 9,951,767,085 documents.

## Parameters

Every constant is in `atlantic_cc.dedup.params`, with the as-run file and line it came from. Changing any
of them changes which documents survive.

| Parameter | Value | As-run source |
|---|---|---|
| whitespace run cap | 128 characters | `rdedup/docs.py` |
| document key | `file_index << 24 \| row` (at most 2^24 rows per file) | `rdedup/docs.py` |
| content id | `xxh3_128(utf8(compacted text))` | `rdedup/docs.py` |
| MinHash | 286 permutations, character 5-grams, seed 42 | `rdedup/docs.py`, dupekit `minhash_ops.rs` |
| LSH | 26 bands x 11 rows, no band index (S-curve midpoint (1/26)^(1/11) = 0.744; reported as 0.75) | dupekit `minhash_ops.rs` |
| label order | `key * 0x9E3779B97F4A7C15 mod 2^64` | `rdedup/graph.py` |
| c075 rule | `containment_ngram3_v2`: word 3-grams, containment >= 0.75 | `verified_dedup.C075_PARAMS` |
| exact scan / probes / posting cap / candidates | 256 / 32 / 512 / 32 | `verified_dedup.C075_PARAMS` |
| oversized component | more than 100,000 members | `verified_dedup.MAX_CLUSTER_SIZE` |
| split | word 5-gram MinHash x 16 content-id sub-groups; key `{root}:{split:04d}:{sub:02d}` | `fuzzy_cluster_text.py` |
| text caps | 64 MiB (grouping), 8 MiB (solving), 256 MiB per solve batch | `fuzzy_cluster_text.py`, `cluster_verify.py` |
| units | 8e9 compressed kept bytes per unit | `rdedup/launcher.py` |
| files per output | 32 | `rdedup/launcher.py` |
| gzip | level 6, mtime 0, one member per kept file | `rdedup/stages.py` |

The dupekit functions come from `marin-dupekit-native==0.1.6.dev31131847556` on PyPI, the build the
release used (its Rust sources are byte-identical to the as-run tag). `atlantic_cc.dedup.oracle` is a
pure-Python port of the CleanText, MinHash and LSH steps; the tests hold the native wheel to it.

## Outputs

Under `--output-root`:

| Path | Contents |
|---|---|
| `deduped_c075/data-{unit:06d}-{chunk:03d}.jsonl.gz` | Survivors. `unit` numbers the 8e9-byte units of the sorted kept list; `chunk` is the position of the kept file in its unit divided by 32. Inside a file, kept files appear in sorted order and rows in ascending row order. A chunk without survivors is not written. |
| `stats/rdedup_stats.json` | Counters of every stage (records in, exact duplicates, components, removals, outputs, round-by-round CC changes) |
| `.rdedup_run.json` | The run root and `inputs_sha256 = sha256("\n".join(sorted paths))` |

A record is the kept row without `n_tokens`, in kept column order, with the compacted text, then
`exact_copies` and `fuzzy_cluster_size`:

```json
{"doc_id": "...", "url": "...", "warc_hash": "...", "snapshot": "CC-MAIN-2025-08", "fasttext_score": 0.9802150130271912,
 "text": "...", "pooled_prob": 0.97, "modernbert_prob": null, "exact_copies": 14, "fuzzy_cluster_size": 4557202931}
```

Records are encoded with msgspec: float32 scores are widened to double and written in shortest
round-trip form (`1e-05`, `0.9802150130271912`), `modernbert_prob` NaN becomes `null`. The public release
regroups these files into crawl-partitioned parquet ([release_format.md](release_format.md)).

## Exactness

**The contract.** Given the same kept files (same paths, so the same sort order; same bytes, so the same
unit packing), this code writes the same documents with the same bytes as the release's dedup, when run
on Linux x86_64 with CPython 3.12, numpy 2.3.5, pyarrow 23.0.1, msgspec 0.20.0 and the pinned
dupekit wheel, on an AVX2 CPU or with AVX-512 masked (the default, below). That covers the survivor and
removal sets, `exact_copies` and `fuzzy_cluster_size`, the decompressed JSON lines of every output file,
and the stats. The `.gz` bytes are also equal when the zlib build is the same (Debian's, as in the
`python:3.12-slim` image); the gzip header's OS byte alone differs between Linux (`03`) and macOS
(`13`), so compare decompressed bytes across platforms.

Nothing else matters: worker counts, partition counts, solve pass counts, task order, retries,
preemptions and the storage backend change no output byte. Every task is a deterministic function of
the plan and of the sealed outputs of the previous stage, exact dedup is a global minimum, components are
exact, and every c075 group is solved whole in content-id order. The tests check this for different
partition and pass counts and for injected task failures.

**Reproducing v1.0.** The release's dedup input was the 8,582,115 files of one `kept/` directory
(`inputs_sha256 = b403438eeb676f417fe6aff69efe28fe185bded93563a92b1399f5c33a9bd526` over their full `gs://`
paths), packed into 13,647 units. The kept files themselves are not published. Regenerating them with
`extract-classify` is exact for extraction and fastText but only statistical for the neural classifier
scores ([running.md](running.md)), so a from-scratch reproduction keeps a slightly different set of
documents and its global dedup differs accordingly; output file names also depend on the kept files'
byte sizes, through the unit packing. Within one directory, the sort order of the kept paths is the
order of their WARC hashes, whatever the prefix.

**The CPU caveat (c075 batches over 256 members).** The inverted-index path picks the 32 rarest probe
3-grams and the 32 best-sharing candidates with `np.argpartition`. Among tied counts, which items make
the cut depends on the SIMD kernel NumPy dispatches at runtime:

| CPU | NumPy 2.3.5 selection kernel | Tied items kept |
|---|---|---|
| x86_64 with AVX-512 (Intel Skylake-SP and later; AMD Zen 4, e.g. EPYC 9B14) | `ArgQSelect_AVX512_SKX` | differ from the release's |
| x86_64 with AVX2 and no AVX-512 (e.g. AMD EPYC 7B12, Zen 2) | `ArgQSelect_AVX2` | **the release's** |
| ARM (Apple silicon, Graviton); x86_64 without AVX2 (Intel before Haswell) or with AVX2 masked | generic introselect | differ from the release's (the same on all) |

The release was built on TPU v4 hosts in us-central2 (every worker was a TPU VM; their CPUs are AMD EPYC
7B12, with AVX2 and no AVX-512). Only removals inside solve batches of more than 256 members can change,
and only when tied counts straddle a cut. On a synthetic 300-member cluster, the generic kernel changes 5
of 151 removals and the AVX-512 kernel none; on synthetic clusters of 300 to 2,500 members, random tie
orders change 0.3% to 4.7% of the removals. In the 8M run, 30% to 35% of the 18.5B component members were
in components of more than 256 members (the largest component alone splits into groups of about 6,000).
On real data, re-solving the whole T2 pool (below) with the generic kernel changes 18,456 of its 7,268,475
removals (0.25%; 17,049 other documents are removed instead), spread over all 56 partitions
([report](../tests/parity/dedup/reports/t2_tie_scalar.json)).

To get the release's results on an AVX-512 host, keep the default `--simd mask-avx512` of
`atlantic-cc dedup run` and `worker`: when NumPy would dispatch AVX-512 kernels, the command re-executes
itself with `NPY_DISABLE_CPU_FEATURES` listing them (the AVX2 kernels stay enabled) and refuses to run if
the mask did not take. Child processes inherit the mask. On an AMD EPYC 9B14 the masked process keeps
exactly the AVX2 kernel's tied items and reproduces the release's solver goldens. `--simd native` lets
NumPy use whatever the CPU offers. Every `run`/`worker` process records its CPU model, flags and enabled
NumPy features in `{run_root}/runtime/`, and `atlantic-cc dedup verify` reports them. ARM hosts and x86_64
hosts without AVX2 cannot reproduce the release's tie order; their results differ only in solve batches of
more than 256 members.

The rest of the contract is pinned by the Python and library versions: CPython 3.12's Unicode tables
decide `casefold`, `split`, `strip` and `\s`; dupekit's Rust tables decide CleanText's lowercasing
(newer than Python's: U+1C89 is lowercased); msgspec decides float and escape formatting.

## Running on one machine

```bash
uv sync --extra gcs            # --extra gcs or --extra s3 only to read or write buckets
atlantic-cc dedup init --run-root scratch/run --output-root out/ \
    --kept gs://bucket/.../kept/ --expect-files 10364
atlantic-cc dedup run --run-root scratch/run --cpus 32 --memory-gb 120
atlantic-cc dedup status --run-root scratch/run
```

- `init` lists `--kept` (or reads `--listing`, a parquet of `path`, `size`), sorts the paths, packs them
  into units, and freezes `plan.json` and `inputs.parquet` under `--run-root`. `--warc-hashes` narrows the
  list to a manifest of WARC hashes; `--expect-files` fails unless exactly that many files remain.
  Running `init` again with the same arguments is a no-op; different arguments for the same run root are
  refused.
- `run` executes the stages in order on a local process pool: `pass1`, `exact`, `lshmap`, `bucket`,
  rounds of `ccNNN{a,b,c}` until the components converge, `canon_a`, `canon_b`, `attrs`, `c075_text`,
  `c075_solve`, `c075_marks`, `apply_c075`, `report`. It is resumable: finished tasks leave done marks in
  `{run_root}/stages/`, and a rerun skips them.
- `--run-root` and `--output-root` take local paths, `gs://` or other fsspec URLs (`s3://`). Seals delete
  intermediates once no later stage needs them; `data/reps`, `data/attrs`, `data/c075_marks` (the
  survivor and removal sets) remain. `--keep-intermediates` keeps everything (for audits).
- Memory: tasks are scheduled against `--memory-gb`. The largest per-task estimates are 15 GB
  (`c075_solve`) and 14 GB (`c075_text`); `pass1`, `c075_text` and `apply_c075` use all `--cpus` through
  their own process pools. Give the host at least 16 GB.
- Partition counts follow the as-run formulas from the input size. `--exact-partitions`,
  `--lsh-partitions`, `--c075-partitions` and `--node-partitions` override them; this changes scheduling
  only. `--unit-gb` (8) and `--files-per-output` (32) change the output file names and grouping, not the
  documents.

## Running on a cluster

Any number of workers can share one run root:

```bash
atlantic-cc dedup init --run-root gs://bucket/scratch/run --output-root gs://bucket/out --kept ... --expect-files N
# on each of N hosts, started in any order and restarted at will:
atlantic-cc dedup worker --run-root gs://bucket/scratch/run --cpus 16 --memory-gb 60
```

Workers coordinate only through the store: a task is claimed by creating its claim object (exactly one
creator wins), the claim is refreshed every 60 s by a compare-and-swap on its generation, and a claim not
refreshed for 480 s (`--stale-seconds`) is taken over. A task whose attempts keep dying (for example from
running out of memory) stops after 6 attempts and stalls its stage visibly (`status` shows it); its
traceback is in `stages/<stage>/failures/`. A worker exits 0 once every stage is sealed. Preempted or
killed workers lose only their in-flight tasks.

Claims need conditional writes: use a `gs://` run root (generation preconditions) or a POSIX filesystem
shared by all workers (hard-link create, rename, and `flock`, with clocks kept in sync, since claim ages
compare one host's write time with another's clock). Other fsspec stores (`s3://`, ...) have no
atomic conditional writes here, so use them with one `run` process only. The output root can be any store.

On the Marin Iris cluster, the T2 check below ran four workers like this, in the region of the data:

```bash
iris --cluster=marin job run --region us-central2 --cpu 16 --memory 64GB --enable-extra-resources \
  --extra gcs --preemptible --max-retries 3 --job-name dedup-w0 --no-wait -- \
  python -m atlantic_cc.dedup.cli dedup worker --run-root $RUN --cpus 16 --memory-gb 60
```

On Slurm, submit an array of the same command; on plain VMs, start it on each.

## Scale

| Run | Kept files | Compressed | Documents in | Units | Partitions (exact / LSH / node / c075) | Output |
|---|---:|---:|---:|---:|---|---|
| DCLM 400m-1x pool (T2 below) | 10,364 | 172.8 GB | 95,466,571 | 22 | 3 / 20 / 5 / 56 | 334 files, 82,354,319 docs |
| Atlantic-CC v1.0 | 8,582,115 | ~109 TB | 61,211,321,760 | 13,647 | 1,393 / 12,073 / 2,730 / 34,824 | 274,767 files, ~26.5 TB |

- **v1.0 run.** Launched 2026-10-17 05:07 UTC, sealed 2026-10-18 11:41 UTC (about 30.5 hours), on up to
  60 whole TPU v4 hosts (workers of 200 CPUs / 360 GB) plus CPU workers of 8-32 cores, in the region of
  the kept files. Components converged after 14 rounds. Scratch was estimated beforehand at about 14 TB
  of pass-1 stash (the 400m-1x pool's 225 bytes per document scale to 13.8 TB) and about 60 TB of c075
  member text at its peak. Most of the storage cost is shuffle reads (each reduce task reads one range
  from every map object; about a billion GETs at this size, growing with the square of the input) and
  scratch storage: budget several hundred dollars at full scale, and keep the run root in the region of
  the kept files.
- **Per-unit costs.** `pass1` reads one 8 GB unit in about 10 minutes on 16 cores, and `apply_c075`
  re-reads it. `c075_solve` took 8.4 CPU-hours for the 400m-1x pool (one core per task).
- The 400m-1x pool's intermediates (as-run): pass-1 stash 21.5 GB, reps 1.92 GB, attrs 97 MB, c075
  marks 8.2 MB.

## Verifying a run

`atlantic-cc dedup verify` compares a run with a reference run (for example the release's, or a second
run with different partition counts) and exits 1 on any difference:

```bash
atlantic-cc dedup verify --candidate RUN_ROOT --reference REFERENCE_RUN_ROOT --report report.json --procs 16
```

It checks, all exactly:

1. the survivor and removal sets: per unit, the `reps` (unique documents with their copy counts),
   `attrs` (component members with root, canonical flag and size) and `c075_marks` (removed keys)
   arrays, byte for byte, and `oversized.json`;
2. `(doc_id, exact_copies, fuzzy_cluster_size)` of every survivor (implied by 3; checked record by record
   in any file that differs);
3. the set of output file names and the decompressed bytes of every output file;
4. `stats/rdedup_stats.json` without timing, worker and path fields (and without partition-dependent
   fields when the runs used different partition counts);
5. the `.gz` bytes (reported; they also depend on the zlib build).

The report also lists the CPU and NumPy features of every process of both runs. Without a reference
run, these invariants hold for any correct output and are cheap to check: every record's
`(warc_hash, doc_id)` exists in its kept file, with `text` equal to the kept text after whitespace
compaction; rows within each gzip member are in ascending kept-row order; `xxh3_128(text)` is unique
across the output; and the stats satisfy `records_in - exact_duplicates = unique_docs` and
`unique_docs - c075.removed = c075.docs = documents written`.

### Parity results

Standalone runs against Marin's as-run `rdedup` (us-central2, AMD EPYC 7B12 hosts, AVX2):

| Tier | Input | Reference | Result |
|---|---|---|---|
| T0 | first 32 files of the 400m-1x list (553.7 MB, 303,479 docs) | as-run code run by us on the same inputs | all five criteria equal: 299,906 unique, 2,040 components, 4,103 removals, 295,803 docs in 6 files; `.gz` bytes identical |
| T0, other partitions | same, partitions 3 / 5 / 3 / 7 instead of 1 / 1 / 1 / 1 | same | all equal (partition-dependent stats excluded) |
| T1 | 400m-1x unit 0: 493 files, 7.99 GB, 4,410,306 docs | as-run code run by us | all five criteria equal: 89,937 exact duplicates, 55,393 components, 121,765 removals, 4,198,604 docs in 16 files (24.9 GB); `.gz` bytes identical |
| T1, other partitions | same, partitions 4 / 6 / 1 / 11 instead of 1 / 1 / 1 / 3 | same | all equal |
| T2 | the whole 400m-1x pool: 10,364 files, 172.8 GB, 95,466,571 docs | the as-run run's outputs and intermediates | all five criteria equal (160 stats fields, 0 differences): 5,843,777 exact duplicates, 89,622,794 unique, 3,728,645 components, 7,268,475 removals, 82,354,319 docs in 334 files; `.gz` bytes identical ([report](../tests/parity/dedup/reports/t2.json)) |

Reports: `tests/parity/dedup/reports/{t0,t1,t2}.json`.
