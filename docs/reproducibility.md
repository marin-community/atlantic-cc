# Reproducibility

What "the same output" means for each stage of Atlantic-CC v1.0, and the checks that measured it.
[provenance.md](provenance.md) lists what produced the release.

## The contract

On Linux x86_64 with CPython 3.12 and the pins below, the CPU stages reproduce v1.0 exactly: download,
decoding, extraction, the fastText gate, tokenization, deduplication, decontamination and packaging. The
neural classifiers reproduce production's scores bit for bit on the accelerator type and chip count that
scored a WARC in production (the catalog's `device` and `n_devices` columns; measured on TPU v4), with
production's per-WARC batching.
Elsewhere they are statistical: a few decisions per WARC flip, each next to a threshold. Dedup's tie order
needs an x86_64 CPU with AVX2, or AVX-512 masked (`--simd mask-avx512`, the default). A classifier flip
propagates through global dedup, so each stage is checked against production's own inputs to that stage,
not end to end.

Pins that change outputs (`pyproject.toml`; `uv.lock` locks the rest): CPython 3.12; warcio 1.8.1 and
charset-normalizer 3.5.0, without the `brotli` module; resiliparse-rs `850891b`, as-run binaries
([extractor.md](extractor.md)); fasttext-wheel 0.9.2; gigatoken 0.10.0 and tokenizers 0.22.2; numpy 2.3.5;
marin-dupekit-native 0.1.6.dev31131847556 and xxhash 3.6.0; msgspec 0.20.0; pyarrow 23.0.1; jax 0.11.0
with libtpu 0.0.44.1.

## Results

Checks against production's outputs, run on 2026-11-02 and 2026-11-03 on Marin's cluster (Linux x86_64,
CPython 3.12, us-central2 unless noted). "Maintainers" marks checks that read production's intermediate
files in Marin's buckets; the others need only public data. Separately, the default test suite
(`uv run pytest`: offline, under a minute) pins every stage's rules on synthetic inputs and committed goldens.

F1–F6 are six fixture WARCs (paths in `tests/parity/r2_asrun.py`), with the TPU type that scored each in
production: F1 CC-MAIN-2014-23 (v5e; processed before the Content-Length guard, a download check production
added during the run), F2 CC-MAIN-2013-48 (v6e), F3 CC-MAIN-2026-12 (v4; the README's quickstart WARC),
F4 CC-NEWS-2017-07 (v5p), F5 CC-SUPPLEMENTAL-2026-22 (v4; one extractor crash) and F6 CC-MAIN-2013-48
(v6e; from shards 0–403, which ran an earlier two-phase build of the same cascade).

| ID | Compares | Scale | Result | Rerun |
|---|---|---|---|---|
| R1 | Phase A and tokenization vs production's `kept/` and tombstones: survivor set and order, kept text bytes, `fasttext_score`, `n_tokens`, phase-A counters | 206 WARCs (F1–F6, 3 from each of 19 chosen shards, 1 from each of 143 random shards): 7,385,216 HTML responses, 4,128,481 survivors, 1,391,125 kept; 103 crawls, all six production regions, 10 two-phase and 57 pre-guard WARCs | **206/206 exact**, including production's 7 extractor crashes | `python -m tests.parity.r1_phase_a` (`plan`, `refs`, `run`, `summarize`; maintainers); per-WARC report `tests/parity/r1_report.json`; [extraction.md](extraction.md#r1-results) |
| I-WARC | Phase A on F1–F6, downloaded from data.commoncrawl.org, vs committed goldens built from production's outputs (`tests/golden/per_warc/`) | 6 WARCs | **6/6 exact** | `uv run pytest -m "network and extractor" tests/integration/test_phase_a_golden.py`, with the extractor installed |
| EXT | Extractor builds and install layouts vs production's binaries, page by page: status (text or crash) and text sha256 | all 32,628 HTML pages of F1 and F3; the as-run wheel (reference) and five Linux variants: production's artifact layout, source builds in production's environment and for manylinux_2_28, the as-run wheel beside upstream resiliparse 1.0.9 or 2.0.0; a macOS arm64 source build | **identical for every variant on every page** (F1: 13,477 text, 39 crash; F3: 19,109 text, 3 crash); the source build in production's environment reproduces both binaries byte for byte | `uv run --with <wheel> pytest -m "network and extractor" tests/parity/extractor/test_extractor_golden.py` (F1); all variants: `tests/parity/extractor/run_parity.py`; [extractor.md](extractor.md#parity-results) |
| R2 | The whole worker (`extract-classify`, JAX backend) on a TPU v4-8 vs production: every fastText survivor's pooled and ModernBERT scores and decision; kept files | F1–F5: 50,524 survivors | Production's TPU type (F3, F5): **bit-identical**, all 15,425 pooled and 6,163 ModernBERT scores; F3's `kept/` file has production's MD5. Other types (F1, F2, F4; 35,099 survivors): pooled \|ΔP\| p99 1.4e-4 (max 7.7e-4), ModernBERT p99 6.2e-3 (max 0.024); 99.977% of decisions agree; 8 flips, all ModernBERT, within 0.0056 of 0.4378. **Pass** (gates: ≥ 99.9% agreement; every flip within 0.02 of a pooled threshold or 0.05 of the ModernBERT one; [report](../tests/parity/reports/r2_tpu_v4.json)) | `python -m tests.parity.r2_phase_b run --fixtures F1 F2 F3 F4 F5 --output OUT --report OUT/r2.json -- --backend jax ...` (maintainers); [running.md](running.md#checking-a-run-against-production) |
| R3 | R2 on GPUs with the PyTorch backend; same gates, every WARC counted as another device type | F1–F5 | In progress | `python -m tests.parity.r3_phase_b run --fixtures F1 F2 F3 F4 F5 --output OUT --report OUT/r3.json -- --backend torch --torch-device cuda ...` (maintainers) |
| R4 T0 | `atlantic-cc dedup` vs Marin's as-run dedup code on the same inputs, five criteria: survivor and removal sets; `(doc_id, exact_copies, fuzzy_cluster_size)`; decompressed outputs; stats; `.gz` bytes | first 32 files of the DCLM 400m-1x pool (production's kept files for the 10,364 WARCs of DCLM's 400M-1x pool): 553.7 MB, 303,479 documents | **all five equal** (299,906 unique, 2,040 components, 4,103 removals by c075, the containment step; 295,803 documents out); also with other partition counts | `atlantic-cc dedup verify --candidate RUN --reference REF`, reference from `tests/parity/dedup/asrun_driver.py` (maintainers); `tests/parity/dedup/reports/t0.json` |
| R4 T1 | Same | unit 0 of the pool: 493 files, 7.99 GB, 4,410,306 documents | **all five equal** (89,937 exact duplicates, 55,393 components, 121,765 removals, 4,198,604 documents out in 16 files); also with other partition counts | same; `t1.json` |
| R4 T2 | Same, against the outputs and intermediates of Marin's own dedup run of the pool | the whole pool: 10,364 files, 172.8 GB, 95,466,571 documents | **all five equal**: stats 160 fields, 0 differences; `.gz` bytes of all 334 output files (5,843,777 exact duplicates, 89,622,794 unique, 3,728,645 components, 7,268,475 removals, 82,354,319 documents out); four 16-CPU AVX2 hosts | same; `t2.json`; [dedup.md](dedup.md#parity-results) |
| R5 | Decontamination replayed with the shipped hash lists vs the release's `decon_matches` and production's mark and filter outputs. D: 66,558 boilerplate phrase hashes dropped before scoring; S′: the 10,944,079 eval-phrase hashes that occur in the pool | public: all of CC-NEWS-2016-08 (43,264 documents), the contaminated files of CC-NEWS-2024-05 and CC-SUPPLEMENTAL-2026-22 (849), CC-MAIN-2014-23's contaminated file and 2 of its 311 parts (799,314); 4 internal dedup output shards (176,304); the eval index, bloom filter and drop sets | **0 mismatches**: 2 of 43,264, 849 of 849 and 15,072 = 15,072 flagged as stored, with equal names; mark attributes and the 25 removed documents equal on the internal shards; D recomputed with the same digest; every S′ and D hash and all 133,669,639 eval-index rows are bloom members; index and bloom rebuilt from the as-run eval corpus are equal (bloom byte-identical); drop sets recomputed equal for all 127 sources | `ATLANTIC_CC_DECON_ARTIFACTS=<dir with S′> uv run pytest -m network tests/parity/decon`; reports `tests/parity/decon/reports/`; [decon.md](decon.md#verification) |
| R6 | Release regroup vs production's intermediate and published files | the part plan over 1,292,989 pieces (27,985 parts, 249 crawls); the manifest and both paths files; 4 part files; all 222 pieces of one 50-file batch; the WARC path map and decontamination lookup | **byte-identical**, every file | `python -m tests.parity.release.r6_regroup` (`plan`, `manifest`, `parts`, `pieces`; maintainers); reports `tests/parity/release/reports/`; [release_format.md](release_format.md#reproduction-results) |
| R7 | Published records vs production's per-WARC `kept/` files; published objects vs the upload records | 820,897 records in 43 files (35 crawls, 1,633 WARCs); all 27,988 objects | **0 mismatches**: every record is in its WARC's kept file with equal URL, crawl, scores and text (after dedup's whitespace compaction, which changed 238); `xxh3_128(text)` unique; `decon_matches` empty exactly in `part-*` files; each WARC's records contiguous, WARCs in `warc_hash` order. The upload records, from which `v1.0.checksums.md5` is built, match every object's MD5 and size | `python -m tests.parity.release.r7_consistency --files 40 --row-groups 2 --report OUT.json` (maintainers); anyone: `atlantic-cc release verify --root https://data.commoncrawl.org/projects/atlantic-cc --checksums v1.0.checksums.md5` |
| M1 | ModernBERT (Ettin-68M) scorers, JAX and PyTorch, vs production's stored scores and vs the as-run Levanter code (the JAX library production ran the models with) on the same CPU. Gates: float32 within 1e-5 of Levanter; vs stored, \|ΔP\| p99 ≤ 0.02, ≥ 99.5% decision agreement on the natural strata (documents not oversampled near a threshold), every flip within 0.05 of 0.4378 | 2,000 released documents from 4 crawls, oversampling near-threshold documents; 1,629 in the band (pooled P in [0.079, 0.8883)) | **pass**: all M1 and M2 tests (10/10) on 16 cores of an AMD EPYC 7B12 ([report](../tests/parity/models/reports/m1_m2_cpu.json); on a CPU the bf16 and JAX ModernBERT runs score a 64-document sample, half near the threshold). Float32 vs Levanter: PyTorch within 2.0e-6 (512 documents), JAX within 5.4e-7 (64). Vs stored: PyTorch CPU float32 \|ΔP\| p99 7.6e-3, 98.8% of decisions (19 flips ≤ 0.0054 from 0.4378), 99.7% natural; PyTorch CPU bf16, all 1,629: p99 7.8e-3, 99.4% (10 flips ≤ 0.0054), 99.8% natural ([report](../tests/parity/models/reports/m1_m2_cpu_torch_bf16.json)). TPU v4, JAX: \|ΔP\| p99 7.1e-3 (max 0.012); 99.63% of decisions (6 flips, ≤ 0.005 from 0.4378), 100% natural; 97.9% of the documents production scored on v4 bit-identical (the rest batched differently; [report](../tests/parity/models/reports/m1_m2_tpu_v4.json)). Apple GPU (MPS) bf16: p99 8.1e-3, 99.0%, 99.6% natural ([report](../tests/parity/models/reports/m1_terminal_mps.json)) | `uv run pytest -m "slow and network" tests/parity/models/test_m1_m2_realdocs.py`; on a TPU, `python -m tests.parity.models.realdocs --runs jax:bf16 --terminal all --report OUT.json`; [classifiers.md](classifiers.md#measured-agreement) |
| M2 | Pooled-transformer scorers, same comparisons. Gates: float32 within 1e-5 of Levanter; bf16 vs Levanter bf16 on a CPU p99 ≤ 5e-4 (max ≤ 2e-3); vs stored p99 ≤ 5e-3, ≥ 99.5% route agreement on the natural strata, every flip within 0.02 of its threshold | the same 2,000 documents | **pass** (same run, [report](../tests/parity/models/reports/m1_m2_cpu.json)): JAX = Levanter bit for bit on the same CPU, bf16 and float32; PyTorch float32 within 6.3e-7; PyTorch with Levanter's CPU numerics vs Levanter bf16 p99 1.6e-4 (max 5.1e-4). TPU v4, JAX: \|ΔP\| p99 1.7e-4 (max 4.7e-4), 100% of routes; all 277 documents production scored on v4 bit-identical ([report](../tests/parity/models/reports/m1_m2_tpu_v4.json)). PyTorch CPU, default TPU emulation: p99 2.1e-4 (max 6.0e-4), 100% of routes | same |
| M3 | fastText: production's call (low-level `predict`, no trailing newline) vs stored `fasttext_score`; pip packages against each other | 83,262 released documents from 4 crawls; 500 for the packages | **all equal** (max \|Δ\| 0) with fasttext-wheel 0.9.2 on Linux x86_64; fasttext-wheel 0.9.2 (NumPy 2 and 1), fasttext 0.9.3, fasttext-numpy2-wheel 0.9.2 and fasttext-numpy2 0.10.4 identical; a trailing newline moves scores by up to 0.70 (p99 0.043) ([report](../tests/parity/models/reports/m3_fasttext_vs_stored.json), [packages](../tests/parity/models/reports/m3_fasttext_packages.json)) | `uv run pytest -m "slow and network" tests/parity/models/test_m3_fasttext.py`; [classifiers.md](classifiers.md#fasttext-agreement) |
| M4 | The three Hugging Face repos used as a user would: fresh environments, offline from the staged files (or from the Hub with `--hub`), card snippets run verbatim | transformers 5.12.1 and 4.48.3 | **pass**: no load warnings; snippet outputs equal this repository's scorers (fastText exactly, pooled within 1e-5, ModernBERT in bf16 within 2e-2; [report](../tests/parity/models/reports/m4_staged.json)) | `python hf/models/verify_staged.py --report OUT.json [--hub]` |
| M5 | Repo and card checks: the two required paragraphs, `license: openmdw-1.1`, a `LICENSE` byte-identical to the official OpenMDW-1.1 text, every staged file as listed in `manifest.json` | 3 repos | **pass**, staged and on the Hub | `uv run python hf/models/check_repo.py <folder or repo id>`; `uv run pytest tests/unit/models/test_m5_card_templates.py` |

## Known differences

- **Catalog vs dedup input: 37 documents.** The per-WARC catalog's `n_kept` sums to 61,211,321,797, 37 more
  than the 61,211,321,760 rows dedup read from `kept/`. 29 WARCs account for −3 of it (next item); the
  other +40 is not located, which would take reading all 8,582,115 kept-file footers. The gap is accepted:
  the release is built from the kept files, and the catalog only summarizes them.
- **Production is not bit-reproducible across TPU types either.** 76 WARCs were processed twice, in
  different regions, and consolidation kept one `kept/` file for each. For 29 of them the two runs kept
  different numbers of documents: the catalog row (from one run) and the kept file (from the other) differ
  by 1 to 8. This matches R2, where a few ModernBERT decisions per WARC flip on another TPU type.
- **Dedup tie order depends on the CPU.** c075 solve batches of more than 256 members pick their 32 rarest
  probe 3-grams and 32 best candidates with `np.argpartition`, whose choice among tied counts depends on the
  SIMD kernel NumPy dispatches. x86_64 with AVX2 and no AVX-512 (the release ran on AMD EPYC 7B12) gives the
  release's choice. AVX-512 kernels (Intel Skylake-SP and later, AMD Zen 4) choose differently; the default
  `--simd mask-avx512` masks them and restores the AVX2 choice exactly, shown on an AMD EPYC 9B14 against the
  as-run solver's goldens (synthetic clusters only: no AVX-512 host runs next to the data). ARM CPUs, and
  x86_64 with AVX2 masked, use NumPy's generic selection, which cannot reproduce the release: on a synthetic
  300-member cluster it changes 5 of 151 removals. In the v1.0 run, 30–35% of the 18.5B component members
  were in components of more than 256 members. On real data, re-solving the whole T2 pool with the generic
  kernel changes 18,456 of its 7,268,475 removals (0.25%; 17,049 other documents are removed instead) in all
  56 partitions ([report](../tests/parity/dedup/reports/t2_tie_scalar.json)).
  `python -m tests.parity.dedup.simd_probe` checks a host.
- **Other platforms.** Only Linux x86_64 is in the contract. A macOS arm64 build of the extractor matched
  production on every F1 and F3 page, but fastText scores differ there in the last bits (97.3% equal, the
  rest within 2.7e-7). On x86_64 CPUs the JAX ModernBERT port differs from Levanter in bf16 (float32
  accumulation order; up to 2.6e-3 in probability on synthetic batches); on TPU v4 it matches production bit
  for bit (R2).
- **WARCs without documents.** 5,978 WARCs gave no documents: 3,905 empty at the source, 101 abandoned
  after three attempts that hung the extractor or killed the worker, and 1,422 refused (HTTP 403) and 550
  throttled (HTTP 503) at audit time. A rerun meets the same files, but HTTP failures can differ
  ([worklist.md](worklist.md#warcs-without-output)).
- **Decontamination from scratch.** The replay with D and S′ is exact on released text. Rebuilding the eval
  corpus from its sources is best effort (unpinned lm-eval datasets, gated sets, moved pins); shipped
  per-record digests locate the drift ([decon.md](decon.md#rebuild-what-can-and-cannot-be-reproduced)).
