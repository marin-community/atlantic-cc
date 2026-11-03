# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""M1 (ModernBERT terminal) and M2 (pooled transformer) on real documents (``docs/reproducibility.md``).

The ~2,000 documents of ``tests/golden/models/realdocs.parquet`` come from public Atlantic-CC v1.0 files,
stratified by stored pooled probability, length and closeness to each threshold
(``tests/golden/models/select_realdocs.py``). For each the golden holds the scores of the as-run Levanter
code on CPU (``make_goldens.py``) and production's TPU scores stored in the release. This module re-reads
the texts, checks that tokenization reproduces the golden token-id hashes, scores them with this
repository's ports using production's batching (``classify.batching.score_ids``) and compares:

- with Levanter on CPU, per precision (structural parity);
- with the stored TPU scores (statistical parity), overall, per crawl, per production TPU type and for the
  "natural" strata (the near-threshold strata are over-sampled), including decision agreement and the
  distance of every flipped document to the threshold that decided it.

The release contains only kept documents, so decisions are compared on kept documents only.

Run on the cluster (``slow``; the pytest gates are in ``test_m1_m2_realdocs.py``)::

    python -m tests.parity.models.realdocs --runs torch:fp32 torch:bf16 jax:fp32 --report OUT.json
"""

import argparse
import json
import os
import platform
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy as np
import pyarrow as pa

from atlantic_cc import spec
from atlantic_cc.classify.batching import score_ids
from atlantic_cc.classify.text import normalize_text
from atlantic_cc.classify.tokenize import load_tokenizer, tokenize_batch, truncate_ids
from atlantic_cc.models import Backend, Precision, load_pooled, load_terminal
from tests.parity.models import common

NEAR_STRATA = ("near_pooled_drop", "near_pooled_accept_below", "near_pooled_accept_above", "near_terminal_keep")
POOLED_THRESHOLDS = np.array([spec.POOLED_DROP_BELOW, spec.POOLED_ACCEPT_AT], dtype=np.float64)


@dataclass(frozen=True)
class Run:
    """One scorer configuration. ``levanter_cpu`` configures the PyTorch pooled port like Levanter on a CPU
    (bf16 matmul inputs, float32 attention) instead of the TPU emulation that is its default."""

    backend: Backend
    precision: Precision
    levanter_cpu: bool = False

    @property
    def name(self) -> str:
        return f"{self.backend}-{self.precision}" + ("-levanter-cpu" if self.levanter_cpu else "")

    @classmethod
    def parse(cls, text: str) -> "Run":
        backend, precision, *rest = text.split(":")
        return cls(Backend(backend), Precision(precision), levanter_cpu=rest == ["levanter-cpu"])


@dataclass(frozen=True)
class Documents:
    golden: pa.Table
    ids: list[np.ndarray]  # pooled input, <= 8,192 tokens
    terminal_ids: list[np.ndarray]  # truncated to 2,048 tokens as production did

    def column(self, name: str) -> np.ndarray:
        return self.golden.column(name).to_numpy(zero_copy_only=False)


def load_documents(golden: pa.Table, texts: list[str]) -> Documents:
    """Tokenize the release texts exactly as production did and check them against the golden hashes."""
    ids = tokenize_batch(load_tokenizer(), [normalize_text(t) for t in texts])
    hashes = [common.ids_sha256(x) for x in ids]
    expected = golden.column("ids_sha256").to_pylist()
    mismatched = [i for i, (a, b) in enumerate(zip(hashes, expected, strict=True)) if a != b]
    if mismatched:
        raise AssertionError(f"tokenization differs from the golden for {len(mismatched)} documents: {mismatched[:5]}")
    return Documents(golden, ids, [truncate_ids(x, spec.TERMINAL_MAX_TOKENS) for x in ids])


def _pooled_scorer(run: Run, source: str, device: str):
    if run.levanter_cpu:
        from atlantic_cc.models.torch_scorer import load_pooled_scorer  # noqa: PLC0415 (optional torch)

        return load_pooled_scorer(source, device=device, bf16_attention_inputs=False)
    return load_pooled(source, run.backend, precision=run.precision, device=device)


def score(run: Run, docs: Documents, terminal_rows: np.ndarray, *, staging: Path, device: str = "cpu") -> dict:
    """Pooled probabilities for every document and terminal probabilities for ``terminal_rows`` (NaN elsewhere)."""
    t0 = time.time()
    pooled = score_ids(
        _pooled_scorer(run, common.model_source("pooled", staging), device), docs.ids, spec.POOLED_MAX_TOKENS
    )
    pooled_seconds = time.time() - t0
    terminal = np.full(len(docs.ids), np.nan, dtype=np.float32)
    t0 = time.time()
    if not run.levanter_cpu and len(terminal_rows):
        scorer = load_terminal(
            common.model_source("terminal", staging), run.backend, precision=run.precision, device=device
        )
        chunk = [docs.terminal_ids[i] for i in terminal_rows]
        terminal[terminal_rows] = score_ids(scorer, chunk, spec.TERMINAL_MAX_TOKENS)
    return {
        "pooled": pooled,
        "terminal": terminal,
        "seconds": {"pooled": round(pooled_seconds, 1), "terminal": round(time.time() - t0, 1)},
    }


def _pooled_route(p: np.ndarray) -> np.ndarray:
    """0 = drop, 1 = band (terminal model), 2 = accept; float32 comparisons as in production."""
    p32 = p.astype(np.float32)
    return np.where(p32 >= spec.POOLED_ACCEPT_AT, 2, np.where(p32 >= spec.POOLED_DROP_BELOW, 1, 0))


def _decisions(mine: np.ndarray, stored: np.ndarray, model: str) -> dict:
    """Decision agreement with production on the given documents and the flips' distance to the threshold."""
    if model == "pooled":
        flips = _pooled_route(mine) != _pooled_route(stored)
        distance = np.abs(stored[:, None] - POOLED_THRESHOLDS[None, :]).min(axis=1)
    else:
        flips = (mine.astype(np.float32) >= spec.TERMINAL_KEEP_AT) != (
            stored.astype(np.float32) >= spec.TERMINAL_KEEP_AT
        )
        distance = np.abs(stored - float(spec.TERMINAL_KEEP_AT))
    return {
        "n": int(len(mine)),
        "agreement": float(1 - flips.mean()) if len(mine) else None,
        "n_flips": int(flips.sum()),
        "max_flip_distance_to_threshold": float(distance[flips].max()) if flips.any() else 0.0,
    }


def _vs_stored(mine: np.ndarray, stored: np.ndarray, model: str) -> dict:
    return {"abs_diff": common.abs_diff_summary(mine, stored), "decisions": _decisions(mine, stored, model)}


def compare(docs: Documents, scores: dict, *, device_of: dict[str, str]) -> dict:
    """Comparisons of one run's scores with the golden's Levanter and stored production scores."""
    report = {}
    stratum = docs.column("stratum").astype(str)
    natural = ~np.isin(stratum, NEAR_STRATA)
    crawl = docs.column("crawl").astype(str)
    device = np.array([device_of.get(p, "unknown") for p in docs.column("warc_path")])
    for model, stored_name in (("pooled", "stored_pooled_prob"), ("terminal", "stored_modernbert_prob")):
        mine = scores[model].astype(np.float64)
        rows = np.isfinite(mine)
        stored = docs.column(stored_name).astype(np.float64)
        entry = {"levanter": {}}
        for precision in Precision:
            name = f"levanter_{model}_prob_{precision}"
            if name in docs.golden.column_names:  # a golden may hold Levanter scores for some models only
                entry["levanter"][str(precision)] = common.abs_diff_summary(mine, docs.column(name).astype(np.float64))
        rows &= np.isfinite(stored)
        entry["stored_tpu"] = {
            "all": _vs_stored(mine[rows], stored[rows], model),
            "natural_strata": _vs_stored(mine[rows & natural], stored[rows & natural], model),
            "by_crawl": {
                c: _vs_stored(mine[rows & (crawl == c)], stored[rows & (crawl == c)], model)
                for c in sorted(set(crawl[rows]))
            },
            "by_production_device": {
                d: _vs_stored(mine[rows & (device == d)], stored[rows & (device == d)], model)
                for d in sorted(set(device[rows]))
            },
        }
        report[model] = entry
    return report


def terminal_rows(docs: Documents, selection: str) -> np.ndarray:
    """Documents the terminal model scores: ``all`` band documents, or a boolean column of the golden
    (``terminal_fp32`` / ``terminal_bf16``: the documents Levanter scored in that precision)."""
    if selection == "all":
        return np.flatnonzero(np.isfinite(docs.column("stored_modernbert_prob").astype(np.float64)))
    return np.flatnonzero(docs.column(selection).astype(bool))


def run_all(runs: list[Run], docs: Documents, *, staging: Path, terminal: str, device: str) -> dict:
    """Score and compare every run; the terminal model scores ``terminal_rows(docs, terminal)``."""
    device_of = common.warc_devices()
    rows = terminal_rows(docs, terminal)
    out = {}
    for run in runs:
        scores = score(run, docs, rows, staging=staging, device=device)
        out[run.name] = {"seconds": scores["seconds"], **compare(docs, scores, device_of=device_of)}
        out[run.name]["scores"] = {k: scores[k].tolist() for k in ("pooled", "terminal")}
        print(run.name, json.dumps({k: v for k, v in out[run.name].items() if k != "scores"}), flush=True)
    return out


def accelerator() -> str:
    """Where the JAX backend runs, e.g. ``TPU v4 x4`` or ``cpu``."""
    import jax  # noqa: PLC0415 (optional jax)

    devices = jax.local_devices()
    return f"{devices[0].device_kind} x{len(devices)}" if devices[0].platform != "cpu" else "cpu"


def environment() -> dict:
    """The machine, the CPUs this process may use, the JAX accelerator and the library versions."""
    cpuinfo = Path("/proc/cpuinfo")
    lines = cpuinfo.read_text().splitlines() if cpuinfo.exists() else []
    cpu_models = [line.split(":", 1)[1].strip() for line in lines if line.startswith("model name")]
    return {
        "machine": f"{platform.system()} {platform.machine()}",
        "cpu": cpu_models[0] if cpu_models else platform.processor(),
        "usable_cpus": len(os.sched_getaffinity(0)) if platform.system() == "Linux" else os.cpu_count(),
        "jax_devices": accelerator(),
        "versions": {name: metadata.version(name) for name in ("jax", "torch", "transformers", "tokenizers")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", type=Run.parse, required=True, help="backend:precision[:levanter-cpu]")
    parser.add_argument("--golden", default=str(common.GOLDEN_DIR / "realdocs.parquet"))
    parser.add_argument("--staging", default=str(common.STAGING_DIR), help="folder of the staged model repos")
    parser.add_argument(
        "--terminal", default="all", choices=["all", "terminal_fp32", "terminal_bf16"], help="see terminal_rows()"
    )
    parser.add_argument("--device", default="cpu", help="PyTorch device")
    parser.add_argument("--report", required=True, help="JSON report (local or gs://)")
    args = parser.parse_args()
    golden = common.realdocs_golden(args.golden)
    docs = load_documents(golden, common.fetch_texts(golden))
    report = {
        "environment": environment(),
        "terminal": args.terminal,
        "runs": run_all(args.runs, docs, staging=Path(args.staging), terminal=args.terminal, device=args.device),
    }
    common.write_json(args.report, report)
    print("wrote", args.report, flush=True)


if __name__ == "__main__":
    main()
