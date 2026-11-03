# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The classifier files production loaded, and an MD5-checked copy of them (maintainers only).

Production read the three classifiers from Marin's GCS buckets (``atlantic_cc.spec.ASRUN_*``); every
regional bucket holds byte-identical copies. The released Hugging Face repos are built from these
files (``hf/models/stage.py``), and the model goldens are computed from them. Reading the buckets
needs Marin's GCS credentials; users load the released repos instead.
"""

import base64
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

import fsspec

from atlantic_cc import spec

CANONICAL_BUCKET = "marin-us-east5"


@dataclass(frozen=True)
class AsRunFile:
    uri: str
    bytes: int
    md5_b64: str

    @property
    def name(self) -> str:
        return self.uri.rsplit("/", 1)[1]

    def in_bucket(self, bucket: str) -> str:
        """This file's URI in another regional bucket (same object path)."""
        prefix = f"gs://{CANONICAL_BUCKET}/"
        if not self.uri.startswith(prefix):
            raise ValueError(f"{self.uri} is not in {prefix}")
        return f"gs://{bucket}/{self.uri[len(prefix) :]}"


def _sibling(weights: dict, name: str, size: int, md5_b64: str) -> AsRunFile:
    return AsRunFile(weights["uri"].rsplit("/", 1)[0] + "/" + name, size, md5_b64)


# model -> the files production read (sizes and MD5s of the as-run GCS objects)
ASRUN_FILES: dict[str, tuple[AsRunFile, ...]] = {
    "fasttext": (AsRunFile(**spec.ASRUN_FASTTEXT_MODEL),),
    "pooled": (
        AsRunFile(**spec.ASRUN_POOLED_MODEL),
        _sibling(spec.ASRUN_POOLED_MODEL, "config.json", 274, "uOID68j9AGesP5bUIujVhA=="),
    ),
    "ettin": (
        AsRunFile(**spec.ASRUN_TERMINAL_MODEL),
        _sibling(spec.ASRUN_TERMINAL_MODEL, "config.json", 2273, "/heEOGO/j3P/OEzcO0O05Q=="),
        _sibling(spec.ASRUN_TERMINAL_MODEL, "tokenizer.json", 3583228, "CdGZvkWWQLr9rrHwJ8nK6Q=="),
        _sibling(spec.ASRUN_TERMINAL_MODEL, "tokenizer_config.json", 20840, "Q8uarHmDETyPmTOCga95KA=="),
        _sibling(spec.ASRUN_TERMINAL_MODEL, "special_tokens_map.json", 694, "kJdTGH4E4/oYwxAOrqMEgg=="),
    ),
}


def md5_b64(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return base64.b64encode(h.digest()).decode()


def _intact(path: Path, f: AsRunFile) -> bool:
    return path.exists() and path.stat().st_size == f.bytes and md5_b64(path) == f.md5_b64


def fetch(
    dest: Path, *, bucket: str = CANONICAL_BUCKET, models: tuple[str, ...] = tuple(ASRUN_FILES)
) -> dict[str, Path]:
    """Copy the as-run files of ``models`` from ``bucket`` into ``dest/<model>/`` and check their MD5s.

    Files already present and intact are not copied again. Read from the bucket of the region you run
    in (all regions hold the same objects). Returns ``model -> folder``.
    """
    folders = {}
    for model in models:
        folder = dest / model
        folder.mkdir(parents=True, exist_ok=True)
        for f in ASRUN_FILES[model]:
            local = folder / f.name
            if not _intact(local, f):
                partial = local.with_name(local.name + ".part")
                with fsspec.open(f.in_bucket(bucket), "rb") as src, open(partial, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 24)
                partial.rename(local)
            if not _intact(local, f):
                raise RuntimeError(f"{local} does not match the as-run object {f.uri} ({f.bytes} B, md5 {f.md5_b64})")
        folders[model] = folder
    return folders
