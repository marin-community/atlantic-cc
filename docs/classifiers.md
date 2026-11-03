# Classifiers

Atlantic-CC v1.0 keeps a page if a three-stage classifier cascade (production spec
`lpv11_fastpipe_v2_1_fused`) predicts that it has useful content. This page describes the three
models, the exact inputs they expect, how this repository runs them, and how far their scores move
between implementations and hardware.

## The cascade

| stage | model | Hugging Face repo | input | rule (float32 comparisons) |
|---|---|---|---|---|
| 1 | fastText | [`marin-community/atlantic-cc-fasttext`](https://huggingface.co/marin-community/atlantic-cc-fasttext) | normalized text | P ≥ 0.0048 → stage 2, else drop |
| 2 | pooled transformer (25.9M parameters) | [`marin-community/atlantic-cc-pooled-transformer`](https://huggingface.co/marin-community/atlantic-cc-pooled-transformer) | ≤ 8,192 tokens | P ≥ 0.8883 → keep; P < 0.079 → drop; else stage 3 |
| 3 | ModernBERT (Ettin-68M) terminal | [`marin-community/atlantic-cc-modernbert`](https://huggingface.co/marin-community/atlantic-cc-modernbert) | first 2,048 tokens | P ≥ 0.4378 → keep, else drop |

The terminal model is a fine-tuned [Ettin-68M encoder](https://huggingface.co/jhu-clsp/ettin-encoder-68m),
which has the ModernBERT architecture; the code calls it "terminal" (`load_terminal`, `models/ettin_*.py`).

P is P(useful): the fastText probability of `__label__useful`, or `softmax(float32 logits)[1]` for the
neural models. Production compared float32 scores with the thresholds under NumPy 2 (NEP 50) rules,
so the effective thresholds are the float32 roundings, for example float32(0.4378) = 0.43779999 <
0.4378 (`atlantic_cc.spec`). Over the full v1.0 catalog (all 8.58M WARCs), 56.5% of HTML responses
passed fastText. Of those survivors, the pooled model accepted 18.3%, sent 34.5% to the terminal
model and dropped 47.1%; the terminal model kept 43.0% of its band documents, so 33.2% of the
survivors were kept.

The labels come from an LLM-based extraction pipeline: `useful` means it produced extracted content
for the page, `not useful` that it reported none. On a frozen, WARC-disjoint test set (6,477 pages
from 89 held-out WARCs, one per snapshot) the best F1 over a threshold sweep is 0.8196 (fastText),
0.8483 (pooled) and 0.8805 (ModernBERT); the bootstrap standard deviation is 0.0077. Training details are
in [Model details](#model-details). Internal names such as "pooled-90M" and "ettin68-10M" refer to
training-set sizes (88.8M and 10M pages), not parameter counts.

## Inputs

1. Extraction: resiliparse-rs `extract_plain_text(html, main_content=True, preserve_formatting="markdown")`
   at the pinned commit (`spec.EXTRACTOR_COMMIT`); see [extraction](extraction.md). The `text` column of the
   release is this output, except that deduplication shortened the rare whitespace runs longer than 128
   characters to 128, which normalization makes irrelevant.
2. Normalization (`classify.text.normalize_text`): `re.sub(r"\s+", " ", text).strip().lower()`.
   Empty results are dropped without scoring.
3. fastText (`classify.fasttext_gate`): the low-level `model.f.predict(text, 2, 0.0, "strict")` with
   **no trailing newline**. The high-level `model.predict()` appends `"\n"`, which adds an
   end-of-sentence token; see [fastText agreement](#fasttext-agreement).
4. Tokens (`classify.tokenize`): ModernBERT-base BPE (pinned revision) of the first 65,536 characters,
   `[CLS]` + at most 8,190 tokens + `[SEP]`; the terminal model reads `truncate_ids(ids, 2048)`, which
   equals a tokenization at `max_length=2048`. The pooled and ModernBERT repos ship these tokenizer files
   (byte-identical except `model_max_length`, 2,048 in the ModernBERT repo: the threshold was calibrated at
   2,048 tokens, although the model accepts 8,192).
5. Batching (`classify.batching.score_ids`): production sorted one WARC's documents by length (stable),
   cut them into batches of 32, padded each batch to the smallest length bucket (128, …, 8192) and
   filled short batches with empty rows. bf16 scores depend slightly on the padded shape, so this
   batching is part of the reproducibility contract.

## Running the models

`atlantic_cc.models` exposes both neural models behind one interface:

```python
from atlantic_cc import spec
from atlantic_cc.classify.batching import score_ids
from atlantic_cc.models import Backend, Precision, load_pooled, load_terminal

pooled = load_pooled(spec.POOLED_HF_REPO, Backend.JAX)  # or a local folder; Backend.TORCH
terminal = load_terminal(spec.TERMINAL_HF_REPO, Backend.TORCH, device="cuda")
probs = score_ids(pooled, ids, spec.POOLED_MAX_TOKENS)  # float32, production batching
```

A `Scorer` takes int32 `[B, L]` ids padded with 50283 and a bool `[B, L]` mask of real tokens and
returns float32 `[B]` probabilities. `Precision.BF16` (default) uses production's numerics;
`Precision.FP32` runs everything in float32 and is the structural reference.

| backend | pooled | terminal |
|---|---|---|
| `Backend.JAX` | `models/pooled_jax.py`: pure `jax.numpy`, line for line Levanter | `models/ettin_jax.py`: pure `jax.numpy`, Levanter's all-bf16 path |
| `Backend.TORCH` | `models/pooled_torch.py` (also the Hub remote code) | Hugging Face `ModernBertForSequenceClassification` (`models/ettin_torch.py`) |

The JAX backend shards each batch's rows over all local devices and replicates the parameters,
like production (`Mesh(devices.reshape(n, 1), ("data", "model"))`), and compiles one program per
padded shape. It reads the same released Hugging Face files as PyTorch. On CPUs, run the ModernBERT
model with PyTorch in float32 (`Precision.FP32`) unless the CPU has bfloat16 instructions. On 16 cores
of an AMD EPYC 7B12, which has none, it took 0.43 s per band document in float32 and 3.4 s in bf16; the
JAX backend, which like Levanter computes full attention matrices on every layer, took about 2 s on a
sample of shorter documents. On the 1,629 band documents of the table below, which oversample the
threshold's neighborhood, float32 makes the same keep decision as production for 98.8% and bf16 for
99.4%.

### Numerics of the production models

- **Pooled transformer.** Embeddings, pooling, LayerNorms, softmax and residuals in float32; every
  linear layer (projection, QKV, output, MLP, head) rounds both operands to bfloat16 and accumulates
  in float32 (`jnp.matmul(..., preferred_element_type=float32)`); GELU is the tanh approximation.
  The two attention einsums are float32 operations at JAX's default matmul precision, which on a TPU
  also rounds their operands to bfloat16. The JAX port issues the same operations, so it gets the
  same rounding on TPUs (and float32 attention on CPUs, like Levanter on a CPU). The PyTorch port
  emulates the TPU by default: `bf16_matmul_inputs` and `bf16_attention_inputs` round the operands
  explicitly and multiply in float32, which is exact for bfloat16 products.
- **ModernBERT (Ettin-68M).** Weights rounded to bfloat16 at load and every operation in bfloat16:
  LayerNorm statistics computed in float32 and rounded, RoPE cos/sin in float32 cast to bf16, q
  scaled by 1/8 before q·k, `where(mask, w, -1e9)`, bf16 softmax, GLU MLP with exact GELU, a mean pool
  over real tokens whose denominator is rounded to bf16 (counts above 256 are not exact), bf16
  logits, then a float32 softmax. Its probabilities therefore take discrete values; 0.4378 lies just
  below sigmoid(−0.25) = 0.4378235, where many documents sit.

## Measured agreement

The numbers below come from `tests/golden/models/make_goldens.py --port`, `tests/parity/models/realdocs.py`,
the M1–M4 tests and the R2 harness (`tests/parity/r2_phase_b.py`), run on the Marin cluster. Their reports
are in the repository: `tests/parity/models/reports/` (`port_vs_levanter_{synthetic,realdocs}.json`,
`m1_m2_tpu_v4.json`, `m1_m2_cpu.json`, `m1_m2_cpu_torch_bf16.json`, `m1_terminal_mps.json`,
`m3_fasttext_*.json`, `m4_staged.json`) and `tests/parity/reports/r2_tpu_v4.json`.

**JAX ports vs the as-run Levanter code, same machine.** In one process on an x86_64 CPU (AMD EPYC
7B12, jax 0.11.0), with production's batching:

- pooled transformer: bit-identical logits and probabilities to Levanter in bf16 and in float32, on
  2,000 released documents and on synthetic batches (real and random weights);
- ModernBERT: bit-identical in bf16 on a small random-weight model. With the real weights, on 512
  released band documents, the float32 logits differ by at most 3.1e-6 and the probabilities by at
  most 7.2e-7 (float32 accumulation order of differently laid out einsums). Over 19 bf16 layers those
  differences cross rounding boundaries: on 64 band documents no bf16 probability is identical, and
  they differ by up to 6.1e-3.

PyTorch float32 matches Levanter float32 on the same kind of CPU within 6.3e-7 (pooled, 2,000
documents) and 2.0e-6 (ModernBERT, 512 documents).

**On the TPU type production used, the JAX ports reproduce production exactly.** Running the whole
worker on TPU v4 for two WARCs that production also scored on TPU v4 reproduced every stored pooled
and terminal score bit for bit (15,425 fastText survivors, 6,163 terminal scores) and wrote a `kept/`
file byte-identical to production's.

**Other TPU types differ statistically.** Re-scoring three WARCs on TPU v4 that production scored on
TPU v5e, v6e and v5p (35,099 survivors): pooled |ΔP| p99 1.4e-4 (max 7.7e-4), terminal |ΔP| p99 6.2e-3
(max 0.024); 99.977% of keep/drop decisions agree, and the 8 flips are all terminal decisions within
0.0056 of 0.4378. Production itself is therefore not bit-reproducible across TPU types.

**Ports vs stored production scores on 2,000 released documents** (four crawls, oversampling
documents near each threshold; decision agreement is reported on the other, "natural" strata; CPU
rows: Linux x86_64, 16 cores of an AMD EPYC 7B12):

| implementation | model | median \|ΔP\| | p99 | max | decisions agree (natural strata) |
|---|---|---|---|---|---|
| JAX, TPU v4, bf16 | pooled (2,000) | 0 | 1.7e-4 | 4.7e-4 | 100% routes (100%) |
| JAX, TPU v4, bf16 | terminal (1,629) | 4.6e-4 | 7.1e-3 | 0.012 | 99.6%, 6 flips ≤ 0.005 from 0.4378 (100%) |
| PyTorch, CPU, default (TPU emulation) | pooled | 2.4e-7 | 2.1e-4 | 6.0e-4 | 100% routes (100%) |
| PyTorch, CPU, `bf16_attention_inputs=False` | pooled | 9.8e-5 | 9.9e-4 | 2.4e-3 | 100% routes (100%) |
| PyTorch, CPU, float32 | pooled | 4.3e-4 | 2.3e-3 | 4.5e-3 | 99.9% routes, 2 flips ≤ 0.0005 from a threshold (100%) |
| PyTorch, CPU, float32 | terminal (1,629) | 1.1e-3 | 7.6e-3 | 0.012 | 98.8%, 19 flips ≤ 0.0054 (99.7%) |
| PyTorch, CPU, bf16 | terminal (1,629) | 1.1e-3 | 7.8e-3 | 0.012 | 99.4%, 10 flips ≤ 0.0054 (99.8%) |
| PyTorch, Apple GPU (MPS), bf16 SDPA | terminal (1,629) | 1.1e-3 | 8.1e-3 | 0.011 | 99.0%, 16 flips ≤ 0.0054 (99.6%) |
| JAX, CPU, bf16 | pooled | 1.0e-4 | 9.9e-4 | 2.4e-3 | 100% routes (100%) |
| JAX, CPU, bf16 | terminal (64-document sample) | 1.5e-3 | 7.1e-3 | 8.7e-3 | 2 flips ≤ 0.0035, both near the threshold (28/28 natural) |

The JAX bf16 pooled scores on a CPU differ from the stored TPU scores only because a CPU computes the
attention products in float32; the PyTorch default rounds those operands as a TPU does.

On TPU v4 the JAX ports reproduce the stored score of every document that production scored on TPU v4
(pooled, 277 documents) or almost every one (terminal, 97.9% of 241; the rest were batched differently
here than in production). Those from other TPU types differ as in the whole-WARC comparison above.
JAX "float32" on a TPU is not float32 arithmetic (matrix products run at the TPU's default precision);
use PyTorch float32 for a float32 reference.

### fastText agreement

The production call reproduces the stored `fasttext_score` bit for bit on all 83,262 documents of
the golden's nine release row groups (four crawls), with `fasttext-wheel` 0.9.2 on Linux x86_64. On a
500-document sample, `fasttext-wheel` 0.9.2 (NumPy 2.3.5 and 1.26.4), `fasttext` 0.9.3,
`fasttext-numpy2-wheel` 0.9.2 and `fasttext-numpy2` 0.10.4 give identical scores. A macOS arm64 build
matched 97.3% of the stored scores exactly and the rest within 2.7e-7 (float32 dot products depend
on the build).

Appending the newline that the high-level `predict()` adds (an end-of-sentence token) moves scores by
a median of 6.1e-4, a p99 of 0.043 and up to 0.70. None of the 83,262 released documents crosses
0.0048 that way, but pages near the threshold can.

The model has a third label, `__label__` (an empty name). Production asked for the two most likely
labels, so a page whose two most likely labels exclude `__label__useful` scores 0;
`classify.fasttext_gate` and the model card reproduce that.

## Model details

All three models predict the same label: `useful` if an LLM-based extraction pipeline run on the page
produced extracted content, `not useful` if it reported that the page has no useful content. Pages come
from Common Crawl WARCs; train and test are split by WARC, stratified by snapshot, and the 89 evaluation
WARCs (one per snapshot) are excluded from all training sets. The neural models' training pages are those
that passed an earlier fastText gate on HTML input at threshold 0.0125 (recall about 0.99), so their training
distribution has more useful pages than the cascade input. Training text was extracted from
script-stripped body HTML and production text from the raw HTML; on about 100,000 pages the ModernBERT
model made the same keep decision under both inputs for 99.38% of pages.

**Evaluation.** Best F1 over a threshold sweep on a frozen test set of 6,477 pages (1,513 useful) sampled
from the 89 held-out WARCs:

| model | best F1 |
|---|---|
| fastText | 0.8196 (at threshold 0.48) |
| pooled transformer | 0.8483 |
| ModernBERT (Ettin-68M) | 0.8805 |

The threshold is picked on the test set itself, so these numbers compare models; they are not
operating-point claims. A 300-sample bootstrap gives a standard deviation of 0.0077, so differences below
about 0.015 are ties. The cascade uses fastText at 0.0048, far below its best-F1 threshold, to keep almost
every useful page for the neural stages.

### fastText

- **Training set:** 5,633,122 lines (1,462,564 useful, 4,170,558 not useful), one normalized page per line,
  sampled from 640 training WARCs (22% of each WARC's pages, at most 3.5 negatives per positive);
  internally "TEXT w640".
- **Training:** `fasttext.train_supervised` with a fixed recipe and no autotuning: `dim` 100, `wordNgrams` 2,
  `bucket` 2,000,000, `minCount` 500, `minCountLabel` 0, no character n-grams (`minn` = `maxn` = 0),
  `epoch` 5, `lr` 0.1, `ws` 5, `neg` 5, `t` 1e-4, softmax loss, on CPU. The vocabulary has 283,823 words,
  each seen at least 500 times in the training text.
- **Score:** fastText adds 1e-5 to each probability before taking its logarithm, so the returned score is
  P(useful) + 1e-5; the threshold applies to the returned value. Do not quantize the model (`.ftz`):
  quantization changes the scores.
- **Throughput** in production: about 4,900 documents per second per CPU core.
- **Repo files:** `model.bin` (918,710,997 bytes, byte-identical to the file production loaded),
  `cascade.png`, `thresholds.json`, `manifest.json`, `LICENSE`, `THIRD_PARTY_NOTICES.md`.

### Pooled transformer

```text
token ids --embedding (50,368 x 256, float32)--> [B, T, 256]
  --per window of 64 tokens: mean, max and min over its real tokens--> [B, T/64, 768]
  --linear with bias, plus a learned position embedding per window--> [B, T/64, 512]
  --4 pre-LayerNorm transformer blocks (8 heads, MLP 2,048, tanh GELU, no biases)--> [B, T/64, 512]
  --mean over windows that contain real tokens--> [B, 512]
  --LayerNorm, then linear without bias--> [B, 2] logits
```

- **Architecture:** a window without real tokens is inactive: attention never attends to it and the final
  mean skips it, so padding does not change the scores. LayerNorms use ε = 1e-5. 25,946,624 parameters
  (the original checkpoint has 25,947,392; two query vectors that only other pooling variants use are
  dropped).
- **Training set:** 88,839,133 pages from 3,117 WARCs, 36.8% useful; internally "pooled-90M" after this
  size (it has 25.9M parameters).
- **Training:** from scratch, one epoch: 347,027 steps of 256 documents at 8,192 tokens, Adam (learning
  rate 3e-4, β = (0.9, 0.95), ε = 1e-8, weight decay 0.1, 200 warmup steps, cosine decay to 10%, gradient
  clipping 1.0), float32 parameters with bfloat16 matmul inputs, on TPUs with Levanter.
- **Throughput** in production: about 3,900 documents per second per TPU v6e chip at 8,192 tokens.
- **Repo files:** `model.safetensors` (float32, PyTorch layout), `config.json` and
  `modeling_pooled_transformer.py` (`trust_remote_code`), the ModernBERT-base tokenizer files,
  `levanter/model.eqx` and `levanter/config.json` (the original checkpoint, byte-identical to the one
  production loaded), `cascade.png`, `thresholds.json`, `manifest.json`, `LICENSE`,
  `THIRD_PARTY_NOTICES.md`.

### ModernBERT (Ettin-68M)

- **Architecture:** ModernBERT with 19 layers, hidden size 512, 8 heads, GLU MLP of width 768, exact GELU,
  LayerNorms without bias (ε = 1e-5), RoPE with θ = 160,000 on all layers, global attention on every third
  layer (0, 3, …, 18) and a symmetric local window of ±64 tokens on the others, no biases. Classification
  head: mean over real tokens, dense layer, GELU, LayerNorm, then a linear classifier. 68,408,322
  parameters.
- **Training set:** the first 10,000,000 of 10,002,709 pages from 345 WARCs; internally "ettin68-10M" after
  this size.
- **Training:** initialized from [`jhu-clsp/ettin-encoder-68m`](https://huggingface.co/jhu-clsp/ettin-encoder-68m)
  (encoder and prediction head; new classifier), then one epoch: 39,062 steps of 256 documents at 8,192
  tokens, Adam (learning rate 5e-5, β = (0.9, 0.95), ε = 1e-8, weight decay 0.1, 200 warmup steps, cosine
  decay to 10%, gradient clipping 1.0), float32 parameters with bfloat16 compute, on a TPU v6e-4 with
  Levanter.
- **Throughput** in production: about 140 documents per second per TPU v6e chip at 2,048 tokens.
- **Checkpoint changes** (none changes the outputs): `config.json` names `ModernBertForSequenceClassification`
  and the labels `not_useful` / `useful` (the training framework had saved a masked-LM config);
  `classifier.bias` is stored as explicit zeros (the trained classifier has no bias); every other tensor is
  bitwise identical; the config keeps the transformers 4.x RoPE keys so older versions load the same model.
- **Repo files:** `model.safetensors` (float32), `config.json`, the ModernBERT-base tokenizer files
  (`model_max_length` 2,048), `cascade.png`, `thresholds.json`, `manifest.json`, `LICENSE`,
  `THIRD_PARTY_NOTICES.md`.

### Limitations

- The labels encode one LLM pipeline's notion of useful content, including its mistakes. The models do not
  judge toxicity, safety, personal information, educational value, factuality or licensing, and they were
  not designed for text that was not extracted from web pages.
- They were trained on Common Crawl's natural mix of languages and domains and were not evaluated per
  language or domain. They were never evaluated on text from other extractors, including upstream
  Resiliparse.
- In the cascade the ModernBERT model only sees pages that the pooled transformer scored in
  [0.079, 0.8883); its scores on other pages were not used.
- The fastText vocabulary consists of words that occur at least 500 times in Common Crawl text.
- Scores near the thresholds depend on floating-point details; see [Measured agreement](#measured-agreement).

## The Hugging Face release

`hf/models/stage.py` builds the three repos under `.cache/models/hf_staging/` from the as-run artifacts
(`atlantic_cc.models.asrun`: copied once from a Marin GCS bucket, `--asrun-bucket` for the region you
run in, and checked against the recorded MD5s); nothing is uploaded. Staging is deterministic: every
file's SHA-256 is in `manifest.json`. Each repo has `README.md` (model card, from the templates in
`hf/models/<repo>/` with repo ids and links filled in from `spec`), `cascade.png` (the card's figure,
drawn by `hf/models/cascade_figure.py`), `LICENSE` (OpenMDW-1.1), `THIRD_PARTY_NOTICES.md`,
`thresholds.json` and `manifest.json` (with the as-run source URIs). The cards are short; this page holds
the details they link to.

- fastText: `model.bin`, byte-identical to the production file.
- Pooled: `model.safetensors` converted from Levanter's `model.eqx` (`models/pooled_eqx.py`; weights
  transposed to PyTorch layout, two unused query vectors dropped), `config.json` with `auto_map` to
  `modeling_pooled_transformer.py` (a copy of `models/pooled_torch.py`) and `tokenizer_class`, the
  ModernBERT tokenizer, and the original `levanter/model.eqx` + `levanter/config.json`.
- ModernBERT: the production safetensors re-saved as `ModernBertForSequenceClassification` with label
  names `not_useful`/`useful` and an explicit zero `classifier.bias` (all other tensors bitwise
  equal); `config.json` keeps both the transformers 5 (`rope_parameters`, `layer_types`) and 4.x
  (`global_rope_theta`, `local_rope_theta`) fields, so older transformers versions load the same
  model; the tokenizer's `model_max_length` is 2,048.

`hf/models/check_repo.py` runs the `upload-hf-model` skill checks (required paragraphs, license
metadata, official license text, manifest) on a staged folder or, after upload, on the Hub repo.
`hf/models/verify_staged.py` loads the repos in fresh environments (transformers 5.12.1 and 4.48.3),
offline from the staged folders or, with `--hub`, from the Hub, and runs the card snippets verbatim.

## Tests

M1–M5 are defined in [reproducibility](reproducibility.md).

| test | file | what it checks |
|---|---|---|
| unit | `tests/unit/models/test_u_model_{pooled,ettin}.py` | both ports vs as-run Levanter goldens on tiny random-weight models (float32 exactly, bf16 loosely), padding, batch composition, scorer contract, checkpoint conversion |
| unit | `tests/unit/models/test_u_model_data_parallel.py` | rows sharded over four devices score like one device |
| M5 | `tests/unit/models/test_m5_card_templates.py`, `tests/parity/models/test_m4_m5_staged.py` | card content, license text, manifests |
| M1, M2 | `tests/parity/models/test_m1_m2_realdocs.py` (`slow`, `network`; harness `realdocs.py`) | ports vs as-run Levanter on real documents and vs production TPU scores, with the gates in the test file |
| M3 | `tests/parity/models/test_m3_fasttext.py` (`slow`, `network`) | fastText vs stored scores, across packages, end-of-sentence deviation |
| M4 | `tests/parity/models/test_m4_m5_staged.py` (`slow`, `network`) | fresh-environment round trip |

The default unit tests (39) run in about 15 seconds with tiny random-weight models. The slow tests load
real weights and run as Iris jobs in us-central2; on 16 CPU cores the M1/M2 tests take about 21 minutes,
most of it PyTorch float32 scoring every band document. For example (from the repository root, after
staging the repos):

```bash
python -m pytest tests/parity/models -m "slow or network" -v
python -m tests.parity.models.realdocs --runs jax:bf16 --terminal all --report OUT.json   # on a TPU
```

The goldens come from the as-run code (`tests/golden/models/make_goldens.py`, run inside an export of
the Marin tag `lpv11-8m-as-run`; documents chosen by `select_realdocs.py`). `realdocs.parquet` holds
Levanter's pooled scores for all 2,000 documents in both precisions and its ModernBERT scores for 512
band documents in float32 and 64 in bf16: Levanter's ModernBERT takes about 5 s per document on 16 CPU
cores.
