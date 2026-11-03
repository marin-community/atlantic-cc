# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers of the model parity tests (M1-M5 in ``docs/reproducibility.md``).

Models come from the locally staged repos (``hf/models/stage.py``) when present, else from the
Hugging Face Hub. Real-document texts are re-read from the public release files
(``atlantic_cc.release.params.HTTPS_ROOT``; row groups cached under ``.cache/models/release/row_groups/``);
the committed golden ``tests/golden/models/realdocs.parquet`` only holds their locations, token-id
hashes and scores.
"""

import hashlib
import json
from pathlib import Path

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from atlantic_cc import spec
from atlantic_cc.release.params import HTTPS_ROOT, VERSION

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_DIR = REPO_ROOT / "tests" / "golden" / "models"
CACHE = REPO_ROOT / ".cache" / "models"
STAGING_DIR = CACHE / "hf_staging"
ROW_GROUP_CACHE = CACHE / "release" / "row_groups"
REPORT_DIR = CACHE / "parity"
RELEASE_URL = f"{HTTPS_ROOT}/{VERSION}/"
HUB_REPOS = {"fasttext": spec.FASTTEXT_HF_REPO, "pooled": spec.POOLED_HF_REPO, "terminal": spec.TERMINAL_HF_REPO}


def model_source(model: str, staging: Path = STAGING_DIR) -> str:
    """The staged folder of ``model`` (fasttext | pooled | terminal) if it exists, else its Hub repo id."""
    local = staging / HUB_REPOS[model].split("/")[1]
    return str(local) if local.is_dir() else HUB_REPOS[model]


def realdocs_golden(path: str = str(GOLDEN_DIR / "realdocs.parquet")) -> pa.Table:
    with fsspec.open(path, "rb") as f:
        return pq.read_table(f)


def warc_devices() -> dict[str, str]:
    """warc_path -> the TPU type production scored that WARC on ("unknown" when not recorded)."""
    table = json.loads((GOLDEN_DIR / "warc_devices.json").read_text())
    return {path: row["device"] or "unknown" for path, row in table.items()}


def release_row_group(source: str, row_group: int, columns: list[str]) -> pa.Table:
    """One row group of a public release file, cached locally."""
    local = ROW_GROUP_CACHE / f"{source.replace('/', '__')}.rg{row_group}.parquet"
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        with fsspec.open(RELEASE_URL + source, "rb", block_size=1 << 22) as f:
            table = pq.ParquetFile(f).read_row_group(row_group)
        pq.write_table(table, local)
    return pq.read_table(local, columns=columns)


def fetch_texts(golden: pa.Table) -> list[str]:
    """Release text of every golden row, looked up by (file, row group, record id)."""
    by_id: dict[str, str] = {}
    locations = set(zip(golden.column("source").to_pylist(), golden.column("row_group").to_pylist(), strict=True))
    for source, row_group in sorted(locations):
        t = release_row_group(source, row_group, ["record_id", "text"])
        by_id.update(zip(t.column("record_id").to_pylist(), t.column("text").to_pylist(), strict=True))
    return [by_id[r] for r in golden.column("record_id").to_pylist()]


def release_rows(golden: pa.Table, columns: list[str]) -> pa.Table:
    """Every row of the release row groups the golden documents come from (not only the golden ones)."""
    locations = set(zip(golden.column("source").to_pylist(), golden.column("row_group").to_pylist(), strict=True))
    return pa.concat_tables(release_row_group(s, rg, columns) for s, rg in sorted(locations))


def ids_sha256(ids: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(ids, dtype="<i4").tobytes()).hexdigest()


def abs_diff_summary(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """|a - b| over the positions where both are finite."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    both = np.isfinite(a) & np.isfinite(b)
    d = np.abs(a[both] - b[both])
    if not d.size:
        return {"n": 0}
    return {
        "n": int(d.size),
        "max": float(d.max()),
        "p99": float(np.quantile(d, 0.99)),
        "p50": float(np.median(d)),
        "mean": float(d.mean()),
        "identical_fraction": float(np.mean(d == 0)),
    }


def write_json(path: str, payload: dict) -> None:
    """Write ``payload`` as indented JSON to a local path or an object-store URL."""
    if "://" not in path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    with fsspec.open(path, "w") as f:
        f.write(json.dumps(payload, indent=2) + "\n")


def write_report(name: str, payload: dict) -> str:
    """Write ``.cache/models/parity/<name>.json`` and return its path."""
    path = str(REPORT_DIR / f"{name}.json")
    write_json(path, payload)
    return path
