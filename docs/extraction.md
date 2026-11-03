# Extraction: phase A of the per-WARC worker

Phase A turns one Common Crawl WARC into its *fastText survivors*: the HTML responses whose
extracted main content passes the fastText gate, with the raw extracted text. Phase B
(`pipeline/phase_b.py`) tokenizes and scores those survivors with the two transformer
classifiers and writes `kept/` and `tombstones/`.

```
download (data.commoncrawl.org) -> select text/html responses -> decode charset
  -> population prefilter -> resiliparse-rs main-content extraction (crash-isolated)
  -> normalize -> fastText P(useful) >= 0.0048 -> presurvivor rows
```

Everything here is a port of the code that built Atlantic-CC v1.0 (Marin tag `lpv11-8m-as-run`:
`experiments/baseline_collection/decode_warcs_clean.py`, `experiments/fast_curation/{fused_phase,
preprocess,cpu_phase_c}.py`). Algorithms, constants, ordering and dtypes are unchanged; only
imports, I/O and orchestration differ. On Linux x86_64 with CPython 3.12 and the pinned wheels,
phase A reproduces the release exactly (see [Reproducibility](#reproducibility)).

## Using it

```python
from atlantic_cc.classify.fasttext_gate import FastTextGate
from atlantic_cc.extract.pool import open_extractor_pool
from atlantic_cc.pipeline.phase_a import run_phase_a

gate = FastTextGate.load("model.bin")  # local path or fsspec URL
with open_extractor_pool(n_procs=8, pkg_dir=None) as pool:  # pkg_dir: see "Installing the extractor"
    presurvivors, counters = run_phase_a(
        "s3://commoncrawl/crawl-data/CC-NEWS/2017/07/CC-NEWS-20170728033529-00050.warc.gz", gate, pool
    )
```

`presurvivors` has `schemas.PRESURVIVOR_SCHEMA` (`doc_id, url, warc_hash, snapshot,
fasttext_score: float32, text: large_string`), one row per survivor in WARC record order.
`counters` is a `PhaseACounters`:

| field | meaning (production marker field) |
|---|---|
| `n_in` | decoded `text/html` responses (`a_n_in`) |
| `n_prefilter_dropped` | out of the population, **plus** extractor crashes (`a_n_prefilter_dropped`) |
| `n_extract_empty` | in the population but the extracted text is empty (`a_n_extract_empty`) |
| `n_extract_crashed` | pages whose extraction killed or failed its process (`a_n_extract_crashed`) |
| `download_failed`, `download_error` | every download attempt failed; zero rows (`a_download_failed`) |
| `records` | diagnostics: records skipped before `n_in` (exceptions, U+FFFD), brotli-encoded pages |

`n_in = n_prefilter_dropped + n_extract_empty + (fastText rejects, not counted) + len(presurvivors)`.

## Stage by stage, with every preserved quirk

### Identity (`warc/identity.py`)

- `warc_path` is the full `s3://commoncrawl/...` key; `warc_hash = sha256(warc_path)[:12]` (hex).
  It names the per-WARC output files.
- `snapshot`: `CC-NEWS-YYYY-MM` when the path contains `CC-NEWS/YYYY/MM/`, else the first
  `CC-MAIN-YYYY-WW` or `CC-SUPPLEMENTAL-YYYY-WW` in the path, else `""`.
- `doc_id` = `WARC-Record-ID` with surrounding whitespace, `<...>` and a `urn:uuid:` prefix removed
  (a bare UUID). `url` = `WARC-Target-URI` or `""`.

### Download (`warc/download.py`)

- `https://data.commoncrawl.org/<key>`, `requests` streaming with a 600 s connect/read timeout.
- Up to 5 attempts. HTTP 429/503 sleep `5 * 2**attempt` s and retry; any other request error
  (including 403/404) also sleeps and retries, except after the last attempt.
- The body is read in 1 MiB pieces under a 1200 s wall-clock deadline, and the whole compressed
  WARC is held in memory.
- **Content-Length guard**: a body whose length differs from `Content-Length` is retried, unless the
  server set `Content-Encoding`. Production added this guard during the run: WARCs processed up to
  2026-09-07 (rung 1 and the 2M ladder, shards 0-8079) had no protection against a truncated HTTP
  200, which would have parsed as fewer records. All 57 pre-guard WARCs in R1 (10 of them rung 1)
  match exactly, so none of them was truncated (see [R1 results](#r1-results)).
- When every attempt fails, phase A returns zero rows and `download_failed`. Any other error
  propagates; production then retried the WARC and, after three starts without completion,
  recorded it as abandoned with zero documents (101 WARCs).

### Record selection (`warc/records.py`)

warcio **1.8.1** `ArchiveIterator` with default arguments. A record is kept iff it is a `response`
with parsed HTTP headers and its HTTP `Content-Type` contains `text/html` (case-insensitive
substring). Consequences:

- no HTTP status filter (404 pages are processed); `application/xhtml+xml` is excluded;
- responses with a zero-length block or a non-HTTP target URI (`dns:`) have no HTTP headers and are
  skipped;
- the payload is `record.content_stream().read()`: chunked transfer encoding is removed and
  `gzip`/`deflate` content encodings are decompressed. warcio 1.8.1 does **not** recognize
  `x-gzip` (1.8.2 does), and decompresses `br` only if the `brotli` module is importable. The
  pinned environment has no brotli, so such payloads are decoded compressed (mojibake) and almost
  always drop out later. Do not install brotli.
- an exception while handling one record (for example a declared `charset=hex`, see below) skips
  that record; it is not counted in `n_in`.
- an exception raised by the iterator itself (a corrupt gzip member, or a `response` record without
  `WARC-Target-URI`, which makes warcio raise `AttributeError`) propagates and fails the WARC.

### Charset cascade (`warc/decode.py`)

Strict at every step; a step is rejected if it fails or if its output contains U+FFFD:

1. byte-order mark: `EF BB BF` -> `utf-8-sig`, `FF FE`/`FE FF` -> `utf-16`;
2. HTTP `Content-Type` charset (`charset\s*=\s*["']?([a-zA-Z0-9_\-:.]+)`);
3. `<meta ... charset ...>` in the **first 1024 bytes**;
4. strict UTF-8;
5. `charset_normalizer.from_bytes(raw).best()` (3.5.0, default arguments), accepted only if its
   encoding is multi-byte or its coherence is at least 0.50;
6. cp1252, then latin-1 (which maps every byte).

Labels from steps 2 and 3 in the latin-1/ASCII family (`iso-8859-1`, `latin1`, `us-ascii`,
`windows-1252`, ...) are decoded as cp1252, as browsers do. Every other label is resolved by
Python's codec registry, with no further WHATWG mapping. Behaviors kept on purpose:

- **Any multi-byte guess is trusted.** Step 5 accepts a multi-byte guess whatever its coherence.
  Undeclared cp1251 Russian is guessed as cp1251 (coherence 0.56) and decodes correctly, while the
  same text in undeclared KOI8-R is guessed as `shift_jis_2004` and decodes as half-width katakana.

- **Literal U+FFFD.** A valid UTF-8 page that contains U+FFFD fails steps 1-4 (U+FFFD is in the
  output). It is then decoded with charset_normalizer's guess when that guess is accepted (a short
  page `x \ufffd y` (a literal U+FFFD) becomes `x 魹ｽ y` through a `shift_jis_2004` guess), otherwise as cp1252
  mojibake (`ï¿½`), or as latin-1 when one of cp1252's five undefined bytes is present. The record's
  "still contains U+FFFD" check is therefore unreachable.
- **Bytes-to-bytes codecs.** A declared `hex`, `base64`, `zlib` or `rot13` passes
  `codecs.lookup`, then `bytes.decode` raises `LookupError`; the record is skipped.

### Population prefilter (`extract/prefilter.py`)

A page is extracted only if `normalize(body_strip(html))` is non-empty. `body_strip` removes
`<script>...</script>` (case-insensitive, across lines), then concatenates the contents of every
`<body ...>...</body>` (non-greedy), or keeps the whole page when there is no body element. Tags
are not removed, so `<body><br></body>` is in the population and `<body></body>` with all content
after it is not. Out-of-population pages count in `n_prefilter_dropped` (0.43% of `n_in` in a
2,223-WARC sample). The classifier thresholds were calibrated on this population.

### Extraction (`extract/engine.py`, `extract/pool.py`)

`resiliparse._extract_rs.extract_plain_text(html, main_content=True, preserve_formatting="markdown")`
from the resiliparse-rs fork `marin-community/atlantic-cc-resiliparse` (tag `atlantic-cc-v1.0`) at
commit `850891b277e919f671ec19f12aafdce451b327f4`, with the binding's defaults (`list_bullets`,
`alt_texts`, `comments` and `post_meta` on; `links`, `form_fields`, `noscript` and
`hidden_elements` off). Main-content selection is a fixed heuristic plus an embedded
gradient-boosted model, so it is deterministic. Screens applied in the worker process before the
engine:

- pages longer than 50,000,000 code points give `""`;
- pages containing `<frameset` or `<FRAMESET` give `""` (they crash the engine); the check is
  case-sensitive, so `<FrameSet` reaches the engine;
- any exception from the engine except `KeyboardInterrupt`/`SystemExit` gives `""` (pyo3's
  `PanicException` is a `BaseException`; a lone surrogate raises `UnicodeEncodeError`).

The engine can still kill its process (SIGSEGV; about 4.4e-7 of pages, mostly pages Common Crawl
truncated at 1 MiB). The pool isolates that:

- pages go to `spawn`ed workers in tasks of at most 25 pages and 32 MiB of characters, in order;
- when anything fails, the 2,000-record chunk is re-run one page per task on a single reused worker,
  and a page whose task dies or raises yields `("", False)`. It counts as crashed **and** as
  prefilter-dropped (production's fused path; rung 1 counted it as an empty extraction instead,
  which leaves the outputs unchanged);
- if more than half of a chunk of at least 8 pages crashes, the install is broken and phase A
  raises instead of producing empty documents;
- workers pin OpenMP/BLAS/MKL/numexpr/Rayon to one thread.

Close the pool (`close()` or the `with` block) before the process that owns it exits; `close()`
waits for the workers. A pool owned by a `multiprocessing` child must be closed before that child's
queue finalizers run (they close the executor's feeder thread at exit priority 10): close it before
the child's target returns, or from a `multiprocessing.util.Finalize` with `exitpriority` above 10.
Closed any later, the workers never receive their shutdown sentinel and the owning process hangs at
exit. The per-WARC worker (`pipeline/worker.py`) does the latter.

There is no per-document timeout (production's `justext_timeout` did not apply to this engine). A
page that hangs the engine hangs the WARC, exactly as in production. A reimplementation with a
timeout would produce documents the release does not contain.

After extraction, `text = text.encode("utf-8", "ignore").decode("utf-8")` (a no-op for engine
output). Empty text counts in `n_extract_empty` (1.82% of `n_in`). There is no minimum length:
whitespace-only text continues and scores 0.0 at the gate. Extracted text can contain U+FFFD
(lexbor maps invalid character references to it); nothing filters it.

### Normalization (`classify/text.py`)

`re.sub(r"\s+", " ", text).strip().lower()`: collapse whitespace, then lowercase, with Python
3.12's Unicode 15.0 tables (29 whitespace code points; ZERO WIDTH SPACE and BOM are not
whitespace; `İ` lowers to two code points). There is no character cap. fastText and both
transformers see exactly this string; `text` in the output stays raw.

### fastText gate (`classify/fasttext_gate.py`)

- Model: `model.bin` (918,710,997 bytes, md5 `AHpq5KR/JN7qQyprSoAyWw==`), loaded with
  fasttext-wheel 0.9.2.
- Score: `model.f.predict(text, 2, 0.0, "strict")`, the **low-level** call, with **no trailing
  newline**. fastText's high-level `model.predict` appends `"\n"`, which adds the end-of-sentence
  token and changes every probability (and fails under NumPy 2). The 0.0048 threshold was
  calibrated without it.
- The probability is the one paired with `__label__useful`, in either tuple order. fastText
  returns `exp(log(p + 1e-5))` as float32, so a score is P(useful) + about 1e-5. Empty text, and a
  text fastText refuses (for example one containing a lone surrogate), score 0.0.
- Gate: a document is dropped iff `float32(score) < float32(0.0048)`. Production compared the
  Python float with the float64 literal 0.0048; for float32 scores the two agree, since no float32
  lies in [0.0048, float32(0.0048)).
- `fasttext_score` in the output is that float32, exactly.

### Tokenization (`classify/tokenize.py`, used by phase B)

ModernBERT-base BPE (`answerdotai/ModernBERT-base` at revision
`8949b909ec900327062f0ebf497f51aef5e6f0c8`) through gigatoken 0.10.0. Production built gigatoken
from `transformers.AutoTokenizer.from_pretrained(...)`; the port loads the revision's
`tokenizer.json` with `tokenizers` instead, which serializes to the same JSON (1,483,360
characters, identical to the transformers 5.12.1 backend's), so neither path changes and importing
the module does not load transformers or torch. On 630 texts (the U-TOK cases, the startup texts
and 600 real pages; 4.6M tokens) both paths give byte-identical ids before and after the switch, at
`max_length` 8192, 2048 and 128.

- only the first 65,536 code points of the normalized text are tokenized (8 per token of the
  8,192-token budget); for text with more than 8 characters per token this cap, not the token
  budget, ends the document;
- ids are `[CLS]=50281 + body[:8190] + [SEP]=50282`, int32, so `n_tokens <= 8192`;
- the terminal model reads `truncate_ids(ids, 2048)` = `ids[:2047] + [SEP]` for longer documents,
  which equals tokenizing at `max_length=2048`;
- added-token literals that survive lowercasing are tokenized as added tokens by both
  implementations: `<|endoftext|>` -> 50279, `<|padding|>` -> 1, `[unused0]`..`[unused82]` ->
  50285..50367;
- `load_tokenizer()` runs production's startup gate: gigatoken must equal the HF `tokenizers`
  reference (`enable_truncation(8192, longest_first, right)` + `encode_batch_fast`) on seven
  adversarial texts, or it refuses to start. `tokenize_batch_reference` is that reference path.

## Installing the extractor

How the extension is built, packaged, installed and verified (wheels, the source build that
reproduces production's binaries byte for byte, the golden test to run after installing) is in
[extractor.md](extractor.md). In short:

The engine is a pyo3 extension (not abi3) plus `liblexbor.so.2`, built for CPython 3.12 on Linux
x86_64. Exactness is established for the binaries production used (`_extract_rs.so` md5
`mjHiDXINNTxVCUttKER1Iw==`, `liblexbor.so.2` md5 `hX51w84LHTU6X/dTPVUyfw==`). Two ways to install:

- **As a wheel** (`resiliparse-rs-atlantic-cc`, built by the scripts in
  [`scripts/build_resiliparse_rs/`](../scripts/build_resiliparse_rs/), either repackaging the
  as-run binaries byte for byte or building from source at 850891b). Install it into the
  environment and pass `pkg_dir=None`.
- **From an artifact directory** laid out like production's
  (`manifest.json`, `resiliparse_py.tar.gz`, `_extract_rs.so`, `liblexbor.so.2`; local path or
  any fsspec URL):

  ```
  atlantic-cc extractor-install --artifact <dir-or-url> --dest /opt/atlantic_cc/extractor
  ```

  This checks that the manifest pins commit 850891b and the running CPython minor, unpacks the
  package, smoke-tests it in a spawned process and prints `pkg_dir` (and `asrun_binaries`: whether
  both libraries are byte-identical to production's). Pass that `pkg_dir` to the pool. The
  destination must allow executable mappings; `/tmp` is `noexec` on some clusters.

`atlantic-cc extractor-check [--pkg-dir DIR]` runs the smoke extraction alone and reports
`asrun_binaries` for whatever would load: the artifact under `DIR`, or the installed wheel.

## Reproducibility

Contract: on Linux x86_64, CPython 3.12, warcio 1.8.1, charset-normalizer 3.5.0, fasttext-wheel
0.9.2, gigatoken 0.10.0 / tokenizers 0.22.2 and the as-run resiliparse-rs binaries, phase A
reproduces each WARC exactly: the survivor set equals the release's kept ∪ tombstone doc_ids, kept
text is byte-identical, and `fasttext_score`, `n_tokens` and the phase-A counters are equal.

Known sources of difference, none of which is a code difference:

- **Truncated downloads before the Content-Length guard** (processed up to 2026-09-07, shards
  0-8079): a production download cut short would show up as a smaller `n_in` and missing documents.
- **Unavailable WARCs**: 5,978 WARCs have zero documents in the release: 3,905 empty at source, 101
  abandoned after repeated hangs, and 1,422 HTTP 403 and 550 HTTP 503 at audit time.
- **Platform**: fastText's float32 arithmetic depends on the build (macOS arm64 can differ from the
  manylinux wheel in the last bits), the engine's model uses the host libm, and other Python
  versions have different Unicode tables.
- **brotli**: installing the `brotli` module changes how `br`-encoded pages decode.

Evidence:

| test | what it checks |
|---|---|
| U-SEL `tests/unit/warc/test_u_sel_records.py` | selection rules, warcio 1.8.1 decoding, skipped vs propagated errors, identity |
| U-DEC `tests/unit/warc/test_u_dec_charset.py` | cascade vectors produced by the as-run `decode_payload` |
| download `tests/unit/warc/test_download.py` | retries, back-off, deadline, Content-Length guard |
| U-PRE `tests/unit/extract/test_u_pre_*.py` | prefilter, screens, crash isolation with real process deaths, task sizing, smoke test |
| U-NORM `tests/unit/classify/test_u_norm.py` | Python 3.12 whitespace set and lowercasing |
| U-FT `tests/unit/classify/test_u_ft.py` | low-level call without newline, label lookup, failures, float32 threshold |
| U-TOK `tests/unit/classify/test_u_tok.py` | gigatoken == HF on 23 adversarial cases at 8192 and 128, exact ids, the cap, `truncate_ids` |
| `tests/integration/test_phase_a_synthetic.py` | every record fate and counter end to end, chunking invariance, failures |
| I-WARC `tests/integration/test_phase_a_golden.py` | fixtures F1-F6 against the committed goldens (`network`, `extractor`, `slow`) |
| R1 `tests/parity/r1_phase_a.py` | the same comparison against production's buckets on the Marin cluster |

The goldens in `tests/golden/per_warc/<warc_hash>/` were built by
`tests/golden/build_per_warc_goldens.py` from production's outputs; see that script for their format.
Rebuilt on the cluster from production's outputs, they are byte-identical to the committed files.

### R1 results

On 2026-11-03, `tests/parity/r1_phase_a.py` reran phase A on the Marin cluster (Iris CPU jobs in
us-central2: Linux x86_64, CPython 3.12.15, glibc 2.41, the pinned wheels, production's extractor
binaries and fastText model) and compared every WARC with production's `kept/` and tombstones,
each read in its own region. Per-WARC results: [`tests/parity/r1_report.json`](../tests/parity/r1_report.json).

| work list | WARCs | exact |
|---|---|---|
| fixtures F1-F6 | 6 | 6 |
| 3 WARCs from each of 19 chosen shards (rung 1, the 2M ladder, the 8M segment, CC-NEWS, CC-SUPPLEMENTAL, every region) | 57 | 57 |
| 1 WARC from each of 143 shards drawn uniformly at random | 143 | 143 |
| **total** | **206** | **206** |

Every check held on every WARC: the survivor set equals kept ∪ tombstones, in the same order; kept
text is byte-identical; `fasttext_score` and `n_tokens` are equal; and so are the five phase-A
counters. The 206 WARCs hold 7,385,216 HTML responses, 4,128,481 fastText survivors and 1,391,125
kept documents. They cover all six regions production ran in (TPU v4, v5e, v5p and v6e hosts), both
code paths (10 rung-1 WARCs), both Content-Length eras (57 pre-guard, 149 post-guard), and 103
snapshots: CC-MAIN-2013-48 to CC-MAIN-2026-39, CC-NEWS 2017-07 to 2026-09, and
CC-SUPPLEMENTAL-2026-22. The 7 extractor crashes production recorded in these WARCs were reproduced.
I-WARC also passes on the same platform (6/6).
