# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The 24 validation sets of ``full-val``: 16 Paloma subsets, 7 Uncheatable Eval subsets and LIMA.

Best-effort rebuild. Ports ``stage_val``/``_val_sources`` of Marin ``decon_8m.py``, the Uncheatable
normalization of ``marin.datakit.download.uncheatable_eval`` and the LIMA flattening of
``experiments/lima.py`` (tag ``lpv11-8m-as-run``). The as-run inputs were Marin GCS copies:

* Paloma ``allenai/paloma@65cd6fc`` (gated), ``<dir>/val/val*.jsonl.gz``;
* Uncheatable Eval dumps 2025-09-01..14 from GitHub ``ziqing-huang/uncheatable_eval`` (the downloader takes
  the *latest* dump per benchmark, so the as-run files are pinned here by git blob sha);
* LIMA ``GAIR/lima@68958e9`` (gated), ``test.jsonl`` + ``train.jsonl`` flattened to ``User: ...`` /
  ``Assistant: ...`` turns.

Staged layout: ``<root>/<name>/eval.parquet`` with ``id = f"{name}-{i}"`` over non-empty texts in
sorted-file, file-line order.
"""

import base64
import gzip
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import fsspec
import pyarrow as pa
import requests
from huggingface_hub import HfApi, snapshot_download

from atlantic_cc.decon import dropsets as ds

logger = logging.getLogger(__name__)

PALOMA_REPO = "allenai/paloma"
PALOMA_REVISION = "65cd6fc"
# Paloma subset -> directory in the HF dataset.
PALOMA_SUBSETS = {
    "4chan": "4chan_meta_sep",
    "c4_100_domains": "c4_100_domains",
    "c4_en": "c4_en",
    "dolma-v1_5": "dolma-v1_5",
    "dolma_100_programing_languages": "dolma_100_programing_languages",
    "dolma_100_subreddits": "dolma_100_subreddits",
    "falcon-refinedweb": "falcon-refinedweb",
    "gab": "gab",
    "m2d2_s2orc_unsplit": "m2d2_s2orc_unsplit",
    "m2d2_wikipedia_unsplit": "m2d2_wikipedia_unsplit",
    "manosphere_meta_sep": "manosphere_meta_sep",
    "mc4": "mc4",
    "ptb": "ptb",
    "redpajama": "redpajama",
    "twitterAAE_HELM_fixed": "twitterAAE_HELM_fixed",
    "wikitext_103": "wikitext_103",
}

UNCHEATABLE_REPO = "ziqing-huang/uncheatable_eval"
UNCHEATABLE_DATES = ("20250901", "20250914")


@dataclass(frozen=True)
class UncheatableDump:
    benchmark: str
    blob_sha: str
    size: int
    records: int


# The as-run dumps (``raw/uncheatable-eval/latest_43d8c7f4/metadata.json``).
UNCHEATABLE_DUMPS = (
    UncheatableDump("wikipedia_english", "1c706ed818ddd615034c5eb3f9aef7962f0a147f", 2559877, 1000),
    UncheatableDump("github_python", "d6a981faeff5c658cd9ea2fa3b2ad70d50babb76", 4093397, 973),
    UncheatableDump("github_cpp", "bb07ef2214882b5a10b539bea78ee61380d49b94", 3808636, 998),
    UncheatableDump("bbc_news", "3b9c37975f6103ef7a13740f99ca77b547e945a8", 3683284, 1081),
    UncheatableDump("arxiv_physics", "d7ae6cc56c8d4bc87ec7e2a831eac49228f08f38", 5096985, 1000),
    UncheatableDump("arxiv_computer_science", "a98a4a59c278d47a0b9410fc97b4b8515046f07f", 4895344, 1000),
    UncheatableDump("ao3_english", "e3afd5a2d743be9d10c63621e71741729e623a68", 4984067, 1000),
)

LIMA_REPO = "GAIR/lima"
LIMA_REVISION = "68958e9"

TEXT_FIELD_CANDIDATES = (
    "text",
    "body",
    "content",
    "article",
    "document",
    "raw_text",
    "code",
    "message",
    "description",
    "story",
)
LIST_FIELD_CANDIDATES = ("paragraphs", "sentences", "lines", "messages")
ID_FIELD_CANDIDATES = ("id", "uuid", "guid", "doc_id", "document_id", "article_id", "hash", "sha", "uid")


def val_set_names() -> list[str]:
    """The 24 names, in as-run order (``_val_sources``)."""
    return (
        [f"paloma_{s}" for s in PALOMA_SUBSETS] + [f"uncheatable_{d.benchmark}" for d in UNCHEATABLE_DUMPS] + ["lima"]
    )


# ---- Uncheatable Eval normalization (as-run marin.datakit.download.uncheatable_eval) ----------------


def _join_list_field(value: Any) -> str | None:
    if isinstance(value, list):
        text_items = [str(item) for item in value if item is not None]
        if text_items:
            return "\n".join(text_items)
    return None


def uncheatable_text(raw: Any) -> str | None:
    """Text of one raw Uncheatable record (``_extract_text``)."""
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        for key in TEXT_FIELD_CANDIDATES:
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value
        for key in TEXT_FIELD_CANDIDATES:
            joined = _join_list_field(raw.get(key))
            if joined:
                return joined
        for key in LIST_FIELD_CANDIDATES:
            joined = _join_list_field(raw.get(key))
            if joined:
                return joined
        title = raw.get("title")
        body = raw.get("body")
        if isinstance(title, str) and isinstance(body, str):
            combined = f"{title.strip()}\n\n{body.strip()}"
            if combined.strip():
                return combined
        if isinstance(title, str) and title.strip():
            return title
        return json.dumps(raw, ensure_ascii=False)
    return str(raw)


def uncheatable_id(raw: Any, benchmark: str, index: int) -> str:
    if isinstance(raw, dict):
        for key in ID_FIELD_CANDIDATES:
            value = raw.get(key)
            if value:
                return str(value)
        metadata = raw.get("metadata")
        if isinstance(metadata, dict):
            for key in ID_FIELD_CANDIDATES:
                value = metadata.get(key)
                if value:
                    return str(value)
    return f"{benchmark}_{UNCHEATABLE_DATES[0]}to{UNCHEATABLE_DATES[1]}_{index:06d}"


def download_uncheatable(out_dir: str, github_token: str | None = None) -> None:
    """Fetch the pinned dumps by git blob sha and write the normalized ``<benchmark>_<dates>.jsonl.gz`` files."""
    headers = {"Accept": "application/vnd.github+json"}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    for dump in UNCHEATABLE_DUMPS:
        url = f"https://api.github.com/repos/{UNCHEATABLE_REPO}/git/blobs/{dump.blob_sha}"
        response = requests.get(url, headers=headers, timeout=120)
        response.raise_for_status()
        blob = response.json()
        content = base64.b64decode(blob["content"])
        if len(content) != dump.size:
            raise ValueError(f"{dump.benchmark}: blob has {len(content)} bytes, expected {dump.size}")
        payload = json.loads(content)
        name = f"{dump.benchmark}_{UNCHEATABLE_DATES[0]}to{UNCHEATABLE_DATES[1]}"
        with fsspec.open(f"{out_dir}/{name}.jsonl.gz", "wb") as raw, gzip.open(raw, "wt", encoding="utf-8") as out:
            for index, record in enumerate(payload):
                text = uncheatable_text(record)
                if text is None or not str(text).strip():
                    raise ValueError(f"record {index} in {name} does not contain text")
                normalized = {
                    "id": uncheatable_id(record, dump.benchmark, index),
                    "text": text,
                    "source": f"{dump.benchmark}:{UNCHEATABLE_DATES[0]}to{UNCHEATABLE_DATES[1]}",
                }
                json.dump(normalized, out, ensure_ascii=False)
                out.write("\n")
        if len(payload) != dump.records:
            raise ValueError(f"{dump.benchmark}: {len(payload)} records, expected {dump.records}")


# ---- Hugging Face downloads (gated: accept the terms and pass a token) --------------------------------


def resolve_revision(repo_id: str, short_sha: str, token: str | None = None) -> str:
    """Full commit sha of a dataset repo revision given by its short sha (Marin records short shas)."""
    for commit in HfApi(token=token).list_repo_commits(repo_id, repo_type="dataset"):
        if commit.commit_id.startswith(short_sha):
            return commit.commit_id
    raise ValueError(f"{repo_id}: no commit starts with {short_sha}")


def download_paloma(out_dir: str, token: str | None = None) -> str:
    """Download the 16 subsets' ``val`` files of ``allenai/paloma@65cd6fc`` into *out_dir* (local)."""
    patterns = [f"{d}/val/val*.jsonl.gz" for d in PALOMA_SUBSETS.values()]
    revision = resolve_revision(PALOMA_REPO, PALOMA_REVISION, token)
    snapshot_download(
        PALOMA_REPO, repo_type="dataset", revision=revision, allow_patterns=patterns, local_dir=out_dir, token=token
    )
    return revision


def download_lima(out_dir: str, token: str | None = None) -> list[str]:
    """Download ``GAIR/lima@68958e9`` ``test.jsonl`` and ``train.jsonl`` into *out_dir* (local)."""
    revision = resolve_revision(LIMA_REPO, LIMA_REVISION, token)
    snapshot_download(
        LIMA_REPO, repo_type="dataset", revision=revision, allow_patterns=["*.jsonl"], local_dir=out_dir, token=token
    )
    return [f"{out_dir}/test.jsonl", f"{out_dir}/train.jsonl"]


# ---- LIMA (as-run experiments/lima.py) --------------------------------------------------------------


def lima_text(record: dict) -> str | None:
    """Flatten a LIMA record's alternating ``conversations`` turns (``None`` when there are none)."""
    turns = record.get("conversations", [])
    if not turns:
        return None
    return "\n\n".join(f"{'User' if i % 2 == 0 else 'Assistant'}: {content}" for i, content in enumerate(turns))


def flatten_lima(input_files: list[str], out_file: str) -> int:
    """Flatten LIMA jsonl files (sorted: ``test.jsonl``, ``train.jsonl``) to one ``{"text"}`` jsonl.gz."""
    rows = 0
    with fsspec.open(out_file, "wb") as dst, gzip.GzipFile(fileobj=dst, mode="wb", mtime=0) as gz:
        for path in sorted(input_files):
            with fsspec.open(path, "r") as src:
                for line in src:
                    line = line.strip()
                    if not line:
                        continue
                    text = lima_text(json.loads(line))
                    if text is None:
                        continue
                    gz.write((json.dumps({"text": text}, sort_keys=True) + "\n").encode("utf-8"))
                    rows += 1
    return rows


# ---- staging (as-run decon_8m.stage_val) -------------------------------------------------------------


def _read_jsonl_texts(path: str) -> Iterator[str]:
    with fsspec.open(path, "rb") as raw, gzip.open(raw, "rt", encoding="utf-8") as f:
        for line in f:
            text = json.loads(line).get("text")
            if text:
                yield text


def _glob(pattern: str) -> list[str]:
    fs, path = fsspec.core.url_to_fs(pattern)
    protocol = pattern.split("://", 1)[0] + "://" if "://" in pattern else ""
    return sorted(protocol + p for p in fs.glob(path))


def val_sources(paloma_dir: str, uncheatable_dir: str, lima_file: str) -> list[tuple[str, str]]:
    """``(eval name, file glob)`` for every validation set, in as-run order."""
    paloma = [(f"paloma_{s}", f"{paloma_dir}/{d}/val/val*.jsonl.gz") for s, d in PALOMA_SUBSETS.items()]
    uncheatable = [
        (f"uncheatable_{d.benchmark}", f"{uncheatable_dir}/{d.benchmark}_*.jsonl.gz") for d in UNCHEATABLE_DUMPS
    ]
    return [*paloma, *uncheatable, ("lima", lima_file)]


def stage_val(root: str, paloma_dir: str, uncheatable_dir: str, lima_file: str) -> dict:
    """Write ``<root>/<name>/eval.parquet`` (``id, text``) for each validation set and a manifest."""
    manifest = {}
    for name, pattern in val_sources(paloma_dir, uncheatable_dir, lima_file):
        files = _glob(pattern)
        if not files:
            raise FileNotFoundError(f"{name}: nothing matches {pattern}")
        texts = [text for f in files for text in _read_jsonl_texts(f)]
        rows = [{"id": f"{name}-{i}", "text": text} for i, text in enumerate(texts)]
        ds.write_parquet(f"{root}/{name}/eval.parquet", pa.Table.from_pylist(rows))
        manifest[name] = {"source": pattern, "files": len(files), "records": len(rows)}
        logger.info("stage-val %s: %d records from %d files", name, len(rows), len(files))
    with fsspec.open(f"{root}/_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest
