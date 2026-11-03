# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reference outputs (goldens) of the two neural classifiers from the as-run Levanter code. Maintainers only.

Runs on CPU inside an export of the Marin tag ``lpv11-8m-as-run`` (Levanter, haliax and the as-run worker
module ``experiments.fast_curation.tpu_phase``), with a copy of this repository's ``src/`` and ``tests/`` on
``PYTHONPATH``. The committed goldens came from Iris CPU jobs in us-central2::

    git -C marin archive lpv11-8m-as-run | tar -x -C asrun_tree
    mkdir -p asrun_tree/atlcc && cp -r src tests asrun_tree/atlcc/          # from this repository
    git -C asrun_tree init -q      # its own repo, so the Iris bundle ignores no enclosing .gitignore
    cd asrun_tree && iris --cluster=marin job run --region us-central2 --cpu 16 --memory 64GB \\
        --enable-extra-resources -e JAX_PLATFORMS cpu --include 'atlcc/tests/**/*' \\
        --exclude '^experiments/scaling_law_sweeps/dclm_core/custom_tasks/' --exclude '^mixtrends\\.html$' \\
        --exclude '^scripts/ray_multihost_vllm/' --job-name atlcc-goldens -- bash -c \\
        "PYTHONPATH=atlcc/src:atlcc:. taskset -c 0-15 python atlcc/tests/golden/models/make_goldens.py \\
            synthetic --out OUT --port && PYTHONPATH=atlcc/src:atlcc:. taskset -c 0-15 python \\
            atlcc/tests/golden/models/make_goldens.py realdocs --selection SELECTION --out OUT --port"

Iris drops ``tests/`` directories from bundles unless included. ``taskset`` matters: the CPU workers are
TPU hosts that show 240 CPUs under a 16-CPU quota, and XLA otherwise starts 240 threads.

``synthetic`` writes ``synthetic.json`` + ``synthetic_ids.npz``: logits and probabilities for fixed synthetic
token batches, for the tiny random-weight models of ``tests/unit/models/random_models.py`` and for the real
checkpoints, at production precision (bf16) and in float32.

``realdocs`` scores the documents ``select_realdocs.py`` picked from the public release with production
batching (stable length sort, batches of 32, length buckets, empty filler rows) and writes
``realdocs.parquet``: per document its location in the release, a token-id hash, the stored production scores
and the Levanter CPU scores. The pooled model scores every document in both precisions; the terminal model
scores the band documents in float32 (all, or ``--terminal-fp32-docs`` of them including the bf16 sample) and a
sample in bf16, half of it near the threshold. No document text is written.

``--port`` also runs this repository's JAX ports on exactly the same batches in the same process and writes
``port_vs_levanter_{synthetic,realdocs}.json`` (max |difference| and the fraction of identical values).
"""

import argparse
import dataclasses
import hashlib
import json
import platform
import shutil
import subprocess
import tempfile
import time
import zlib
from collections.abc import Callable
from pathlib import Path

import fsspec
import haliax as hax
import jax
import jax.numpy as jnp
import levanter.models.pooled_transformer as pooled_module
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from experiments.baseline_collection.comparison_sample import normalize_text
from experiments.fast_curation import batch_format, preprocess, tpu_phase
from haliax import Axis
from haliax.partitioning import ResourceAxis, set_mesh
from jax.sharding import Mesh
from levanter.layers.attention import AttentionMask
from levanter.models.modernbert import ModernBertConfig, load_hf_sequence_classifier
from levanter.models.pooled_transformer import PooledTransformerConfig, load_pooled_transformer_classifier
from levanter.utils.tree_utils import inference_mode

from atlantic_cc.models import asrun, ettin_jax, pooled_eqx, pooled_jax
from atlantic_cc.models.jax_scorer import useful_probability
from tests.unit.models import random_models as rm

BATCH_SIZE = 32
PAD, SEP = 50283, 50282
POOLED_MAX, TERMINAL_MAX = 8192, 2048
TOKENIZER_REF = "answerdotai/ModernBERT-base"
PRECISIONS = ("bf16", "fp32")
DTYPES = {"bf16": jnp.bfloat16, "fp32": jnp.float32}
NEAR_TERMINAL_STRATUM = "near_terminal_keep"
SEED = 0

# name -> (padded length, row lengths); a 0-length row is an empty filler row, as production adds.
SYNTHETIC_BATCHES = {
    "pooled_tiny_256": (256, [256, 200, 129, 65, 64, 17, 2, 0]),
    "pooled_tiny_128": (128, [128, 100, 64, 63, 17, 2, 0, 0]),
    "pooled_8192": (8192, [8192, 3001, 2049, 640, 129, 65, 64, 2]),
    "pooled_128": (128, [128, 100, 64, 63, 17, 2, 0, 0]),
    "ettin_tiny_128": (128, [128, 100, 64, 17, 9, 3, 2, 0]),
    "ettin_tiny_64": (64, [64, 40, 17, 9, 3, 2, 0, 0]),
    "ettin_2048": (2048, [2048, 1500, 700, 130, 64, 3, 2, 0]),
    "ettin_128": (128, [128, 100, 64, 17, 3, 2, 0, 0]),
}
RANDOM_SEED = {"pooled": rm.POOLED_SEED, "ettin": rm.ETTIN_SEED}
TINY_ETTIN_MAX = 512

# (ids int32 [B, L], real bool [B, L]) -> float32 logits [B, 2] / probabilities [B]
BatchFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclasses.dataclass(frozen=True)
class Runner:
    logits: BatchFn
    probs: BatchFn


def _cpu_model() -> str:
    out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    return next((line.split(":", 1)[1].strip() for line in out.splitlines() if line.startswith("Model name")), "")


def environment() -> dict:
    return {
        "marin_tag": "lpv11-8m-as-run",
        "jax": jax.__version__,
        "platform": jax.default_backend(),
        "n_devices": len(jax.devices()),
        "machine": f"{platform.system()} {platform.machine()}",
        "cpu": _cpu_model(),
        "python": platform.python_version(),
    }


def make_mesh() -> Mesh:
    devices = np.array(jax.devices())
    return Mesh(devices.reshape(len(devices), 1), (ResourceAxis.DATA, ResourceAxis.MODEL))


def levanter_runner(model) -> Runner:
    """Production's jitted score function (``tpu_phase.make_score_fn``) and the same jit returning logits."""

    def _logits_impl(m, tokens, mask):
        return m(tokens, mask).astype(jnp.float32)

    logits_fn = hax.named_jit(_logits_impl, axis_resources={"batch": "data"})
    probs_fn = tpu_phase.make_score_fn()

    def call(fn, ids: np.ndarray, real: np.ndarray) -> np.ndarray:
        batch, pos = Axis("batch", ids.shape[0]), Axis("position", ids.shape[1])
        seg = hax.named(np.where(real, 0, -1).astype(np.int32), (batch, pos))
        mask = AttentionMask(is_causal=False).with_segment_ids(seg, seg)
        return np.asarray(fn(model, hax.named(ids, (batch, pos)), mask).array, dtype=np.float32)

    return Runner(logits=lambda i, r: call(logits_fn, i, r), probs=lambda i, r: call(probs_fn, i, r))


def port_runner(logits_impl: Callable, params) -> Runner:
    """A port's forward ``logits_impl(params, ids, real)``, jitted with the parameters as arguments (as the
    port's scorer does) and the float32 cast inside the jit, like production's score function."""
    params = jax.device_put(params)
    logits_fn = jax.jit(lambda p, i, r: logits_impl(p, i, r).astype(jnp.float32))
    probs_fn = jax.jit(lambda p, i, r: useful_probability(logits_impl(p, i, r)))
    return Runner(
        logits=lambda i, r: np.asarray(logits_fn(params, i, r), dtype=np.float32),
        probs=lambda i, r: np.asarray(probs_fn(params, i, r), dtype=np.float32),
    )


def pooled_port(eqx_dir: Path, precision: str) -> Runner:
    cfg = json.loads((eqx_dir / "config.json").read_text())
    leaves = pooled_eqx.read_eqx_leaves(str(eqx_dir / "model.eqx"), cfg["num_layers"])
    state = pooled_eqx.state_dict_from_eqx(leaves, cfg["num_layers"])
    params = pooled_jax.params_from_state_dict(state, cfg["num_layers"])
    spec = pooled_jax.PooledJaxSpec(
        pool_window=cfg["pool_window"],
        num_heads=cfg["num_heads"],
        max_seq_len=cfg["max_seq_len"],
        matmul_dtype=DTYPES[precision],
    )
    return port_runner(lambda p, i, r: pooled_jax.forward(p, i, r.astype(jnp.float32), spec), params)


def ettin_port(hf_dir: Path, precision: str) -> Runner:
    params, spec, _ = ettin_jax.load_model(str(hf_dir), dtype=DTYPES[precision])
    return port_runner(lambda p, i, r: ettin_jax.logits_fn(p, i, r, spec), params)


def load_pooled_levanter(eqx_dir: Path, max_seq_len: int, vocab_size: int):
    raw = json.loads((eqx_dir / "config.json").read_text())
    raw.pop("config_class")
    config = dataclasses.replace(PooledTransformerConfig(**raw), max_seq_len=max_seq_len)
    return inference_mode(load_pooled_transformer_classifier(config, str(eqx_dir), vocab_size=vocab_size), True)


def load_ettin_levanter(hf_dir: Path, max_seq_len: int, precision: str):
    """``tpu_phase.load_model`` (vanilla attention, 2 labels, pad 50283) with a selectable dtype."""
    converter = ModernBertConfig().hf_checkpoint_converter(ref_checkpoint=str(hf_dir))
    hf_config = converter.hf_config_from_hf_checkpoint(str(hf_dir))
    config = dataclasses.replace(
        ModernBertConfig.from_hf_config(hf_config),
        max_seq_len=max_seq_len,
        attn_backend="vanilla",
        num_labels=2,
        pad_token_id=PAD,
    )
    model = load_hf_sequence_classifier(config, str(hf_dir), axis_mapping=None, dtype=DTYPES[precision])
    return inference_mode(model, True)


def set_pooled_precision(precision: str) -> None:
    """Levanter's pooled ``_matmul`` reads this module global at trace time (bf16 in production)."""
    pooled_module.COMPUTE_DTYPE = DTYPES[precision]


def asrun_spec(folders: dict[str, Path]) -> argparse.Namespace:
    """The fields of the production ``PipelineSpec`` that ``load_pooled_model`` reads, for local copies."""
    tpu_phase._assert_ckpt_in_region = lambda ckpt, bucket: None  # the guard is about GCS regions
    return argparse.Namespace(
        pooled_ckpt_for=lambda bucket: str(folders["pooled"]), max_length=POOLED_MAX, pad_token_id=PAD
    )


def diff_summary(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    both = np.isfinite(a) & np.isfinite(b)
    d = np.abs(a[both] - b[both])
    return {
        "n": int(d.size),
        "max_abs": float(d.max()) if d.size else 0.0,
        "identical_fraction": float(np.mean(d == 0)) if d.size else 1.0,
        "nan_positions_equal": bool(np.array_equal(np.isnan(a), np.isnan(b))),
    }


def write_outputs(out: str, files: dict[str, bytes]) -> None:
    for name, data in files.items():
        with fsspec.open(f"{out.rstrip('/')}/{name}", "wb") as f:
            f.write(data)
        print(f"wrote {out.rstrip('/')}/{name} ({len(data)} bytes)", flush=True)


def synthetic(args: argparse.Namespace) -> None:
    work = Path(args.work)
    folders = asrun.fetch(work / "asrun", bucket=args.asrun_bucket, models=("pooled", "ettin"))
    tiny_pooled, tiny_ettin = work / "pooled_tiny", work / "ettin_tiny"
    rm.write_pooled_levanter(tiny_pooled, RANDOM_SEED["pooled"])
    rm.write_ettin_hf(tiny_ettin, RANDOM_SEED["ettin"])
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copyfile(folders["ettin"] / name, tiny_ettin / name)  # Levanter's converter loads one
    ids_out = {
        name: rm.random_documents(zlib.crc32(name.encode()), lengths)
        for name, (_, lengths) in SYNTHETIC_BATCHES.items()
    }
    cases, port_report = [], {}
    mesh = make_mesh()
    with set_mesh(mesh):
        for precision in PRECISIONS:
            set_pooled_precision(precision)
            pooled_asrun, _ = tpu_phase.load_pooled_model(asrun_spec(folders), mesh, args.asrun_bucket)
            levanter = {
                ("pooled", "random"): levanter_runner(
                    load_pooled_levanter(tiny_pooled, rm.POOLED_TINY["max_seq_len"], rm.VOCAB_SIZE)
                ),
                ("pooled", "asrun"): levanter_runner(pooled_asrun),
                ("ettin", "random"): levanter_runner(load_ettin_levanter(tiny_ettin, TINY_ETTIN_MAX, precision)),
                ("ettin", "asrun"): levanter_runner(load_ettin_levanter(folders["ettin"], TERMINAL_MAX, precision)),
            }
            ports = {
                ("pooled", "random"): pooled_port(tiny_pooled, precision),
                ("pooled", "asrun"): pooled_port(folders["pooled"], precision),
                ("ettin", "random"): ettin_port(tiny_ettin, precision),
                ("ettin", "asrun"): ettin_port(folders["ettin"], precision),
            }
            for name, (length, lengths) in SYNTHETIC_BATCHES.items():
                kind = "pooled" if name.startswith("pooled") else "ettin"
                weights = "random" if "_tiny_" in name else "asrun"
                ids, real = rm.padded_batch(ids_out[name], lengths, length)
                runner = levanter[(kind, weights)]
                logits, probs = runner.logits(ids, real), runner.probs(ids, real)
                cases.append(
                    {
                        "model": kind,
                        "weights": f"random:seed={RANDOM_SEED[kind]}" if weights == "random" else "asrun",
                        "batch": name,
                        "precision": precision,
                        "logits": logits.astype(np.float64).tolist(),
                        "probs": probs.astype(np.float64).tolist(),
                    }
                )
                print(name, precision, np.round(probs, 5), flush=True)
                if args.port:
                    port = ports[(kind, weights)]
                    key = f"{name}-{precision}"
                    port_report[key] = {
                        "logits": diff_summary(port.logits(ids, real), logits),
                        "probs": diff_summary(port.probs(ids, real), probs),
                    }
                    print("  port vs levanter", port_report[key], flush=True)
    batches = {name: {"length": length, "lengths": lengths} for name, (length, lengths) in SYNTHETIC_BATCHES.items()}
    doc = {"generated_with": environment(), "batches": batches, "cases": cases}
    with tempfile.TemporaryFile() as f:
        np.savez_compressed(f, **ids_out)
        f.seek(0)
        npz = f.read()
    files = {"synthetic.json": (json.dumps(doc, indent=1) + "\n").encode(), "synthetic_ids.npz": npz}
    if args.port:
        report = {"environment": environment(), "cases": port_report}
        files["port_vs_levanter_synthetic.json"] = (json.dumps(report, indent=1) + "\n").encode()
    write_outputs(args.out, files)


def production_batches(id_lists: list[list[int]], max_len: int):
    """``tpu_phase.score_survivors``' batching: ``(row indices, ids [32, L], real [32, L])`` per batch."""
    order = np.argsort([len(x) for x in id_lists], kind="stable")
    for start in range(0, len(id_lists), BATCH_SIZE):
        idx = order[start : start + BATCH_SIZE]
        chunk = [id_lists[i] for i in idx]
        target = batch_format.bucket_for(max(len(c) for c in chunk), max_len)
        ids, seg = batch_format.pad_batch(list(chunk) + [[]] * (BATCH_SIZE - len(chunk)), target, PAD)
        yield idx, np.asarray(ids, np.int32), np.asarray(seg) >= 0


def score_rows(runner: Runner, id_lists: list[list[int]], max_len: int) -> tuple[np.ndarray, np.ndarray]:
    logits = np.zeros((len(id_lists), 2), dtype=np.float32)
    probs = np.zeros(len(id_lists), dtype=np.float32)
    for idx, ids, real in production_batches(id_lists, max_len):
        logits[idx] = runner.logits(ids, real)[: len(idx)]
        probs[idx] = runner.probs(ids, real)[: len(idx)]
    return logits, probs


def ids_sha256(ids: list[int]) -> str:
    return hashlib.sha256(np.asarray(ids, dtype="<i4").tobytes()).hexdigest()


def fp32_terminal_rows(band: np.ndarray, bf16_rows: np.ndarray, n_docs: int | None) -> np.ndarray:
    """float32 terminal rows: every band document, or the bf16 sample plus a seeded sample up to ``n_docs``."""
    if n_docs is None:
        return np.flatnonzero(band)
    rest = np.setdiff1d(np.flatnonzero(band), bf16_rows)
    extra = np.random.default_rng(SEED + 1).permutation(rest)[: max(n_docs - len(bf16_rows), 0)]
    return np.sort(np.concatenate([bf16_rows, extra]))


def bf16_terminal_rows(stratum: np.ndarray, band: np.ndarray, n_docs: int) -> np.ndarray:
    """bf16 terminal sample: half near-threshold band documents, the rest other band documents (seeded)."""
    rng = np.random.default_rng(SEED)
    near = rng.permutation(np.flatnonzero(band & (stratum == NEAR_TERMINAL_STRATUM)))[: n_docs // 2]
    rest = rng.permutation(np.flatnonzero(band & (stratum != NEAR_TERMINAL_STRATUM)))[: n_docs - len(near)]
    return np.sort(np.concatenate([near, rest]))


def realdocs(args: argparse.Namespace) -> None:
    work = Path(args.work)
    folders = asrun.fetch(work / "asrun", bucket=args.asrun_bucket, models=("pooled", "ettin"))
    with fsspec.open(args.selection, "rb") as f:
        sel = pq.read_table(f)
    texts = [normalize_text(t) for t in sel.column("text").to_pylist()]
    ids = preprocess.tokenize_trunc_batch(preprocess.load_tokenizer(TOKENIZER_REF), texts, POOLED_MAX)
    if ids != sel.column("input_ids").to_pylist():
        raise RuntimeError("the as-run HF tokenization differs from the selection's gigatoken ids")
    stratum = np.asarray(sel.column("stratum").to_pylist())
    band = ~np.isnan(sel.column("modernbert_prob").to_numpy(zero_copy_only=False).astype(np.float64))
    bf16_rows = bf16_terminal_rows(stratum, band, args.terminal_bf16_docs)
    rows = {"fp32": fp32_terminal_rows(band, bf16_rows, args.terminal_fp32_docs), "bf16": bf16_rows}
    terminal_ids = [batch_format.truncate_ids(x, TERMINAL_MAX, SEP) for x in ids]
    columns: dict[str, pa.Array] = {
        name: sel.column(name)
        for name in ("source", "row_group", "record_id", "crawl", "warc_path", "stratum", "n_tokens")
    }
    columns["ids_sha256"] = pa.array([ids_sha256(x) for x in ids])
    columns["stored_fasttext_score"] = sel.column("fasttext_score")
    columns["stored_pooled_prob"] = sel.column("pooled_prob")
    columns["stored_modernbert_prob"] = sel.column("modernbert_prob")
    for precision, chosen in rows.items():
        mask = np.zeros(len(ids), dtype=bool)
        mask[chosen] = True
        columns[f"terminal_{precision}"] = pa.array(mask)
    port_report: dict[str, dict] = {}
    env = environment()

    def write() -> None:
        table = pa.table(columns).replace_schema_metadata({"generated_with": json.dumps(env)})
        sink = pa.BufferOutputStream()
        pq.write_table(table, sink, compression="zstd")
        files = {"realdocs.parquet": sink.getvalue().to_pybytes()}
        if args.port:
            report = {"environment": env, "runs": port_report}
            files["port_vs_levanter_realdocs.json"] = (json.dumps(report, indent=1) + "\n").encode()
        write_outputs(args.out, files)

    def compare(name: str, port: Runner, id_lists: list, max_len: int, logits: np.ndarray, probs: np.ndarray) -> None:
        t0 = time.time()
        p_logits, p_probs = score_rows(port, id_lists, max_len)
        port_report[name] = {"logits": diff_summary(p_logits, logits), "probs": diff_summary(p_probs, probs)}
        print(f"  port {name} ({time.time() - t0:.0f}s) vs levanter: {port_report[name]}", flush=True)

    mesh = make_mesh()
    with set_mesh(mesh):
        for precision in PRECISIONS:
            set_pooled_precision(precision)
            model, _ = tpu_phase.load_pooled_model(asrun_spec(folders), mesh, args.asrun_bucket)
            t0 = time.time()
            logits, probs = score_rows(levanter_runner(model), ids, POOLED_MAX)
            print(f"levanter pooled {precision}: {len(ids)} docs in {time.time() - t0:.0f}s", flush=True)
            columns[f"levanter_pooled_logits_{precision}"] = pa.array(list(logits), pa.list_(pa.float32(), 2))
            columns[f"levanter_pooled_prob_{precision}"] = pa.array(probs, pa.float32())
            if args.port:
                compare(
                    f"pooled-{precision}", pooled_port(folders["pooled"], precision), ids, POOLED_MAX, logits, probs
                )
            write()
        for precision in ("fp32", "bf16"):
            chosen = rows[precision]
            chunk = [terminal_ids[i] for i in chosen]
            t0 = time.time()
            model = load_ettin_levanter(folders["ettin"], TERMINAL_MAX, precision)
            logits, probs = score_rows(levanter_runner(model), chunk, TERMINAL_MAX)
            print(f"levanter terminal {precision}: {len(chunk)} docs in {time.time() - t0:.0f}s", flush=True)
            full_logits = np.full((len(ids), 2), np.nan, dtype=np.float32)
            full_probs = np.full(len(ids), np.nan, dtype=np.float32)
            full_logits[chosen], full_probs[chosen] = logits, probs
            columns[f"levanter_terminal_logits_{precision}"] = pa.array(list(full_logits), pa.list_(pa.float32(), 2))
            columns[f"levanter_terminal_prob_{precision}"] = pa.array(full_probs, pa.float32())
            if args.port:
                compare(
                    f"terminal-{precision}", ettin_port(folders["ettin"], precision), chunk, TERMINAL_MAX, logits, probs
                )
            write()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["synthetic", "realdocs"])
    parser.add_argument("--out", required=True, help="output folder (local or gs://)")
    parser.add_argument("--work", default="/tmp/a2_goldens", help="local scratch for the model files")
    parser.add_argument("--asrun-bucket", default="marin-us-central2", help="regional bucket holding the models")
    parser.add_argument("--selection", help="realdocs: selection.parquet written by select_realdocs.py")
    parser.add_argument("--terminal-bf16-docs", type=int, default=64)
    parser.add_argument("--terminal-fp32-docs", type=int, help="float32 terminal sample size (default: every band doc)")
    parser.add_argument("--port", action="store_true", help="also compare this repository's JAX ports")
    args = parser.parse_args()
    {"synthetic": synthetic, "realdocs": realdocs}[args.command](args)


if __name__ == "__main__":
    main()
