# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the three Atlantic-CC classifier repos for the Hugging Face Hub in a local folder.

Nothing is uploaded. The script copies the as-run artifacts (once; MD5-checked, see
``atlantic_cc.models.asrun``), converts them, and writes one folder per repo under ``<cache>/hf_staging/``:

- ``atlantic-cc-fasttext``: ``model.bin`` unchanged.
- ``atlantic-cc-pooled-transformer``: ``model.eqx`` converted to ``model.safetensors`` (PyTorch layout),
  a Hugging Face ``config.json`` with ``auto_map`` to the remote code ``modeling_pooled_transformer.py``
  (a copy of ``src/atlantic_cc/models/pooled_torch.py``), the ModernBERT tokenizer files, and the
  untouched Levanter checkpoint under ``levanter/``.
- ``atlantic-cc-modernbert``: the fine-tuned Ettin-68M checkpoint (ModernBERT architecture) re-saved as
  ``ModernBertForSequenceClassification``
  with real label names and an explicit zero ``classifier.bias`` (every other tensor bitwise equal),
  and tokenizer files whose ``model_max_length`` is the operating context, 2,048.

Staging is deterministic: the same artifacts give byte-identical folders (``manifest.json`` lists
the SHA-256 of every file).

Each folder also gets ``README.md`` (from the template next to this script, repo ids and links filled in
from ``spec``), ``atlantic-cc-header.png`` (the banner in ``docs/assets/``), ``cascade.png`` (rendered by
``cascade_figure.py``), ``LICENSE`` (OpenMDW-1.1),
``THIRD_PARTY_NOTICES.md``, ``thresholds.json`` and ``manifest.json``.

Usage::

    uv run python hf/models/stage.py                       # all three repos
    uv run python hf/models/stage.py --repos atlantic-cc-modernbert
    python hf/models/stage.py --asrun-bucket marin-us-central2  # in an Iris job in us-central2
"""

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
from pathlib import Path

import fasttext
import numpy as np
import torch
from safetensors.numpy import load_file, save_file

from atlantic_cc import spec
from atlantic_cc.models import asrun, ettin_torch, pooled_eqx
from atlantic_cc.models.asrun import ASRUN_FILES, AsRunFile
from atlantic_cc.models.pooled_torch import PooledTransformerConfig, PooledTransformerForSequenceClassification

logger = logging.getLogger("stage")

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE = REPO_ROOT / ".cache" / "models"
POOLED_SOURCE = REPO_ROOT / "src" / "atlantic_cc" / "models" / "pooled_torch.py"
LICENSE_FILE = TEMPLATE_DIR / "LICENSE.OpenMDW-1.1"
APACHE_LICENSE_FILE = REPO_ROOT / "LICENSE"
STAGER = "hf/models/stage.py"
DOCS_URL = f"{spec.CODE_REPO_URL}/blob/main/docs/classifiers.md"
FIGURE = "cascade.png"  # rendered by cascade_figure.py next to each card template
HEADER = "atlantic-cc-header.png"  # the project banner at the top of every card
HEADER_SOURCE = REPO_ROOT / "docs" / "assets" / HEADER

FASTTEXT_REPO = spec.FASTTEXT_HF_REPO.split("/")[1]
POOLED_REPO = spec.POOLED_HF_REPO.split("/")[1]
ETTIN_REPO = spec.TERMINAL_HF_REPO.split("/")[1]
REPOS = (FASTTEXT_REPO, POOLED_REPO, ETTIN_REPO)
REPO_IDS = {FASTTEXT_REPO: spec.FASTTEXT_HF_REPO, POOLED_REPO: spec.POOLED_HF_REPO, ETTIN_REPO: spec.TERMINAL_HF_REPO}

POOLED_MODULE = "modeling_pooled_transformer"
POOLED_VOCAB_SIZE = 50368  # as-run: experiments/fast_curation/tpu_phase.py POOLED_VOCAB_SIZE
POOLED_ID2LABEL = {0: "not_useful", 1: "useful"}
POOLED_TOKENIZER_CLASS = "PreTrainedTokenizerFast"  # the class in the copied tokenizer_config.json
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
ETTIN_BASE_MODEL = "jhu-clsp/ettin-encoder-68m"
# Generic PretrainedConfig fields of transformers 4.x that 5.x carries over from the as-run config.json.
ETTIN_STALE_CONFIG_KEYS = (
    "add_cross_attention",
    "cross_attention_hidden_size",
    "decoder_start_token_id",
    "finetuning_task",
    "is_decoder",
    "prefix",
    "pruned_heads",
    "task_specific_params",
    "tf_legacy_loss",
    "tie_encoder_decoder",
    "tokenizer_class",
    "torchscript",
    "use_bfloat16",
)


def sha256_hex(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def _copy_tokenizer(asrun_ettin: Path, out: Path, model_max_length: int) -> None:
    for name in TOKENIZER_FILES:
        shutil.copyfile(asrun_ettin / name, out / name)
    config = json.loads((out / "tokenizer_config.json").read_text())
    config["model_max_length"] = model_max_length
    _write_json(out / "tokenizer_config.json", config)


def stage_fasttext(asrun: Path, out: Path) -> dict:
    os.link(asrun / "model.bin", out / "model.bin")
    model = fasttext.load_model(str(out / "model.bin"))
    args = model.f.getArgs()
    return {
        "n_words": f"{len(model.words):,}",
        "labels": sorted(model.labels),
        "dim": args.dim,
        "word_ngrams": args.wordNgrams,
        "bucket": args.bucket,
        "min_count": args.minCount,
        "epoch": args.epoch,
        "lr": args.lr,
    }


def stage_pooled(asrun: Path, asrun_ettin: Path, out: Path) -> dict:
    raw = json.loads((asrun / "config.json").read_text())
    if raw.pop("config_class") != "PooledTransformerConfig" or raw.pop("dropout") != 0.0:
        raise ValueError(f"unexpected as-run pooled config {raw}")
    num_labels = raw.pop("num_labels")
    config = PooledTransformerConfig(
        vocab_size=POOLED_VOCAB_SIZE,
        bf16_matmul_inputs=True,
        bf16_attention_inputs=True,
        # transformers 4.x pipelines only build a tokenizer for a custom config that names its class
        tokenizer_class=POOLED_TOKENIZER_CLASS,
        id2label=dict(POOLED_ID2LABEL),
        label2id={v: k for k, v in POOLED_ID2LABEL.items()},
        architectures=[PooledTransformerForSequenceClassification.__name__],
        **raw,
    )
    if config.num_labels != num_labels:
        raise ValueError(f"num_labels {num_labels} != {config.num_labels}")
    config.auto_map = {
        "AutoConfig": f"{POOLED_MODULE}.PooledTransformerConfig",
        "AutoModelForSequenceClassification": f"{POOLED_MODULE}.PooledTransformerForSequenceClassification",
    }
    leaves = pooled_eqx.read_eqx_leaves(str(asrun / "model.eqx"), config.num_layers)
    state = pooled_eqx.state_dict_from_eqx(leaves, config.num_layers)
    model = PooledTransformerForSequenceClassification(config)
    expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    if expected != {k: v.shape for k, v in state.items()}:
        raise ValueError("converted pooled state dict does not match the PyTorch module")
    save_file(state, str(out / "model.safetensors"), metadata={"format": "pt"})
    config.save_pretrained(out)
    shutil.copyfile(POOLED_SOURCE, out / f"{POOLED_MODULE}.py")
    _copy_tokenizer(asrun_ettin, out, spec.POOLED_MAX_TOKENS)
    (out / "levanter").mkdir()
    for name in ("model.eqx", "config.json"):
        os.link(asrun / name, out / "levanter" / name)
    n_params = sum(int(np.prod(v.shape)) for v in state.values())
    n_unused = sum(int(np.prod(leaves[name].shape)) for name in pooled_eqx.UNUSED_LEAVES)
    return {"n_params": f"{n_params:,}", "n_params_checkpoint": f"{n_params + n_unused:,}"}


def stage_ettin(asrun: Path, out: Path) -> dict:
    model = ettin_torch.load_model(str(asrun), dtype=torch.float32, attn_implementation="eager")
    with torch.no_grad():
        model.classifier.bias.zero_()
    model.save_pretrained(out)
    original = load_file(str(asrun / "model.safetensors"))
    staged = load_file(str(out / "model.safetensors"))
    if set(staged) != set(original) | {"classifier.bias"}:
        raise ValueError(f"unexpected tensor names: {sorted(set(staged) ^ set(original))}")
    for name, value in original.items():
        if staged[name].dtype != value.dtype or not np.array_equal(staged[name], value):
            raise ValueError(f"{name} changed while re-saving")
    if staged["classifier.bias"].dtype != np.float32 or np.any(staged["classifier.bias"] != 0):
        raise ValueError("classifier.bias must be float32 zeros")
    # transformers 5 stores the RoPE thetas only in rope_parameters; 4.x reads the flat keys and would
    # silently fall back to its 10k local-theta default, so write both (5.x ignores the flat keys).
    config = json.loads((out / "config.json").read_text())
    for key in ETTIN_STALE_CONFIG_KEYS:
        config.pop(key)
    config["global_rope_theta"] = config["rope_parameters"]["full_attention"]["rope_theta"]
    config["local_rope_theta"] = config["rope_parameters"]["sliding_attention"]["rope_theta"]
    (out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    _copy_tokenizer(asrun, out, spec.TERMINAL_MAX_TOKENS)
    return {"n_params": f"{sum(int(np.prod(v.shape)) for v in staged.values()):,}"}


def thresholds_document(role: str) -> dict:
    def f32(x: np.float32) -> dict:
        return {"value": round(float(x), 6), "float32": float(x)}

    return {
        "dataset": "Atlantic-CC",
        "dataset_version": spec.DATASET_VERSION,
        "spec_id": spec.PRODUCTION_SPEC_ID,
        "spec_hash": spec.PRODUCTION_SPEC_HASH,
        "this_model": role,
        "comparison": "keep/drop tests compare float32 P(useful) with float32(threshold) (NumPy 2 rules)",
        "score": "P(useful) = softmax(float32 logits)[1] for the neural models; fastText probability of "
        "__label__useful from the low-level predict without a trailing newline",
        "cascade": [
            {"stage": "fasttext", "repo": spec.FASTTEXT_HF_REPO, "keep_if_at_least": f32(spec.FASTTEXT_KEEP_AT)},
            {
                "stage": "pooled",
                "repo": spec.POOLED_HF_REPO,
                "max_tokens": spec.POOLED_MAX_TOKENS,
                "accept_if_at_least": f32(spec.POOLED_ACCEPT_AT),
                "drop_if_below": f32(spec.POOLED_DROP_BELOW),
                "otherwise": "terminal",
            },
            {
                "stage": "terminal",
                "repo": spec.TERMINAL_HF_REPO,
                "max_tokens": spec.TERMINAL_MAX_TOKENS,
                "keep_if_at_least": f32(spec.TERMINAL_KEEP_AT),
            },
        ],
    }


def fill_template(template: str, values: dict[str, object]) -> str:
    def sub(match: re.Match) -> str:
        return str(values[match.group(1)])

    text = re.sub(r"\{\{(\w+)\}\}", sub, template)
    if "{{" in text:
        raise ValueError("unfilled template placeholder")
    return text


def write_manifest(out: Path, repo: str, sources: dict[str, AsRunFile]) -> None:
    files = []
    for path in sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"):
        rel = path.relative_to(out).as_posix()
        entry: dict[str, object] = {"path": rel, "bytes": path.stat().st_size, "sha256": sha256_hex(path)}
        if rel in sources:
            src = sources[rel]
            entry["as_run_source"] = {"uri": src.uri, "bytes": src.bytes, "md5_b64": src.md5_b64}
        files.append(entry)
    _write_json(
        out / "manifest.json",
        {
            "repo": REPO_IDS[repo],
            "dataset_version": spec.DATASET_VERSION,
            "spec_id": spec.PRODUCTION_SPEC_ID,
            "spec_hash": spec.PRODUCTION_SPEC_HASH,
            "staged_by": STAGER,
            "license": "openmdw-1.1",
            "files": files,
        },
    )


def _unchanged_tokenizer_files() -> dict[str, AsRunFile]:
    """Tokenizer files copied byte for byte (``tokenizer_config.json`` gets a new ``model_max_length``)."""
    return {f.name: f for f in ASRUN_FILES["ettin"] if f.name in ("tokenizer.json", "special_tokens_map.json")}


def stage_repo(repo: str, folders: dict[str, Path], staging: Path) -> Path:
    out = staging / repo
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    if repo == FASTTEXT_REPO:
        values = stage_fasttext(folders["fasttext"], out)
        role, sources = "fasttext", {"model.bin": ASRUN_FILES["fasttext"][0]}
    elif repo == POOLED_REPO:
        values = stage_pooled(folders["pooled"], folders["ettin"], out)
        role = "pooled"
        sources = {f"levanter/{f.name}": f for f in ASRUN_FILES["pooled"]} | _unchanged_tokenizer_files()
    else:
        values = stage_ettin(folders["ettin"], out)
        role = "terminal"
        sources = _unchanged_tokenizer_files()
    shutil.copyfile(LICENSE_FILE, out / "LICENSE")
    shutil.copyfile(TEMPLATE_DIR / repo / FIGURE, out / FIGURE)
    shutil.copyfile(HEADER_SOURCE, out / HEADER)
    notices = (TEMPLATE_DIR / repo / "THIRD_PARTY_NOTICES.md").read_text()
    (out / "THIRD_PARTY_NOTICES.md").write_text(notices.replace("{{apache_license}}", APACHE_LICENSE_FILE.read_text()))
    _write_json(out / "thresholds.json", thresholds_document(role))
    values.update(
        {
            "github_repo": spec.CODE_REPO_URL,
            "docs_url": DOCS_URL,
            "fasttext_repo": spec.FASTTEXT_HF_REPO,
            "pooled_repo": spec.POOLED_HF_REPO,
            "terminal_repo": spec.TERMINAL_HF_REPO,
            "spec_id": spec.PRODUCTION_SPEC_ID,
            "extractor_commit": spec.EXTRACTOR_COMMIT,
            "tokenizer_revision": spec.TOKENIZER_REVISION,
        }
    )
    weights = {"model.bin", "model.safetensors"} & {p.name for p in out.iterdir()}
    for name in weights:
        values[f"sha256_{name.replace('.', '_')}"] = sha256_hex(out / name)
        values[f"bytes_{name.replace('.', '_')}"] = f"{(out / name).stat().st_size:,}"
    readme = fill_template((TEMPLATE_DIR / repo / "README.md").read_text(), values)
    (out / "README.md").write_text(readme)
    write_manifest(out, repo, sources)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="holds asrun/ and hf_staging/")
    parser.add_argument("--repos", nargs="+", choices=REPOS, default=list(REPOS))
    parser.add_argument(
        "--asrun-bucket", default=asrun.CANONICAL_BUCKET, help="regional bucket to copy the as-run files from"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    needed = {FASTTEXT_REPO: ("fasttext",), POOLED_REPO: ("pooled", "ettin"), ETTIN_REPO: ("ettin",)}
    models = tuple(sorted({m for repo in args.repos for m in needed[repo]}))
    folders = asrun.fetch(args.cache / "asrun", bucket=args.asrun_bucket, models=models)
    for repo in args.repos:
        out = stage_repo(repo, folders, args.cache / "hf_staging")
        logger.info("staged %s", out)


if __name__ == "__main__":
    main()
