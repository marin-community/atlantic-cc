# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``atlantic-cc extract-classify``: run the per-WARC worker over one shard of a WARC list.

Writes ``kept/``, ``tombstones/``, ``markers/`` and ``attempts/`` under ``--output`` (see
``atlantic_cc.pipeline.outputs``). Rerunning the same command resumes: WARCs with a marker are
skipped. Exit status: 0 when every WARC of the shard has a marker, 1 when some WARC raised in
phase A (rerun to retry it), and 70 when phase A stalled (a supervisor should restart the worker).
"""

import argparse
import functools
import logging
import os
import shutil
import sys

import fsspec

from atlantic_cc import spec
from atlantic_cc.classify.tokenize import load_tokenizer, tokenize_batch
from atlantic_cc.extract.install import install_artifact
from atlantic_cc.io.resolve import store_for
from atlantic_cc.models import Backend, Precision, load_pooled, load_terminal, resolve_model_dir
from atlantic_cc.pipeline.phase_b import run_phase_b
from atlantic_cc.pipeline.worker import (
    STALL_MINUTES,
    CascadePhaseA,
    PhaseB,
    WorkerStalledError,
    read_warc_list,
    run_worker,
    select_shard,
)

logger = logging.getLogger(__name__)

FASTTEXT_FILE = "model.bin"  # in the Hub repo and in a staged repo folder
STALL_EXIT_CODE = 70  # production's stall-watchdog exit code
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/atlantic_cc")


def local_copy(source: str, cache_dir: str) -> str:
    """``source`` unless it is a URL; a URL file or folder is copied once to ``cache_dir/<bucket path>``."""
    if "://" not in source:
        return source
    fs, path = fsspec.core.url_to_fs(source)
    path = path.rstrip("/")
    dest = os.path.join(cache_dir, path)
    if os.path.exists(dest):
        return dest
    partial = f"{dest}.partial-{os.getpid()}"
    shutil.rmtree(partial, ignore_errors=True)
    if fs.isfile(path):
        os.makedirs(os.path.dirname(partial), exist_ok=True)
        fs.get_file(path, partial)
    else:
        for remote in fs.find(path):
            local = os.path.join(partial, os.path.relpath(remote, path))
            os.makedirs(os.path.dirname(local), exist_ok=True)
            fs.get_file(remote, local)
    os.replace(partial, dest)
    logger.info("copied %s -> %s", source, dest)
    return dest


def fasttext_model_path(source: str, cache_dir: str) -> str:
    """The local ``model.bin``: a file, a repo folder, a Hub repo id, or a URL of either (copied once)."""
    local = local_copy(source, cache_dir)
    if os.path.isfile(local):
        return local
    return os.path.join(resolve_model_dir(local), FASTTEXT_FILE)


def describe_device(backend: Backend, torch_device: str) -> tuple[str, int]:
    """``(device kind, device count)`` phase B runs on, as production's markers recorded them ("TPU v4", 4)."""
    if backend is Backend.JAX:
        import jax  # noqa: PLC0415 (optional jax)

        devices = jax.local_devices()
        return devices[0].device_kind, len(devices)
    if torch_device.startswith("cuda"):
        import torch  # noqa: PLC0415 (optional torch)

        return torch.cuda.get_device_name(torch_device), 1
    return torch_device, 1


def default_torch_device() -> str:
    import torch  # noqa: PLC0415 (optional torch)

    return "cuda" if torch.cuda.is_available() else "cpu"


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "extract-classify",
        help="download, extract and classify WARCs into kept/ and tombstones/",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    work = parser.add_argument_group("work")
    work.add_argument(
        "--warc-list",
        required=True,
        help="paths file in Common Crawl's warc.paths format (optionally .gz, local or a URL); lines are "
        "bucket-relative paths or s3://commoncrawl/ URLs",
    )
    work.add_argument("--output", required=True, help="output root: a local directory or a gs:// or s3:// prefix")
    work.add_argument("--shard-index", type=int, default=0)
    work.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="partition by int(sha256(url)[:8], 16) %% num_shards; 32000 reproduces production's base shards",
    )
    models = parser.add_argument_group("models")
    models.add_argument("--backend", type=Backend, choices=list(Backend), required=True)
    models.add_argument(
        "--precision",
        type=Precision,
        choices=list(Precision),
        default=Precision.BF16,
        help="bf16 reproduces production numerics; fp32 is the structural reference",
    )
    models.add_argument(
        "--torch-device", default=None, help="PyTorch device for both models (default: cuda if available, else cpu)"
    )
    models.add_argument(
        "--fasttext-model",
        default=spec.FASTTEXT_HF_REPO,
        help="Hub repo id, repo folder or model.bin; folders and files may be gs:// or s3:// URLs",
    )
    models.add_argument(
        "--pooled-model", default=spec.POOLED_HF_REPO, help="Hub repo id or repo folder (local or a URL)"
    )
    models.add_argument(
        "--terminal-model", default=spec.TERMINAL_HF_REPO, help="Hub repo id or repo folder (local or a URL)"
    )
    extraction = parser.add_argument_group("extraction")
    extraction.add_argument(
        "--extractor",
        default=None,
        help="prefix of a prebuilt resiliparse-rs artifact to install (local or URL); omit when resiliparse "
        "with the _extract_rs extension is installed as a package",
    )
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help="local copies of models given as URLs, and the unpacked --extractor; must allow executable "
        "mappings (not a noexec /tmp)",
    )
    scheduling = parser.add_argument_group("scheduling")
    scheduling.add_argument(
        "--phase-a-procs",
        type=int,
        required=True,
        help="WARCs extracted in parallel; each process holds the 0.9 GB fastText model",
    )
    scheduling.add_argument(
        "--extract-procs", type=int, default=6, help="resiliparse-rs processes per phase-A process (production: 6)"
    )
    scheduling.add_argument("--queue-depth", type=int, default=6, help="WARCs in flight (bounds RAM; production: 6)")
    scheduling.add_argument(
        "--stall-minutes",
        type=float,
        default=STALL_MINUTES,
        help="exit with status 70 when no WARC finishes phase A for this long",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    root = args.output.rstrip("/")
    warcs = select_shard(read_warc_list(args.warc_list), args.shard_index, args.num_shards)
    logger.info(
        "shard %d/%d of %s: %d WARCs -> %s", args.shard_index, args.num_shards, args.warc_list, len(warcs), root
    )

    pkg_dir = None
    if args.extractor is not None:
        installed = install_artifact(args.extractor, os.path.join(args.cache_dir, "resiliparse_rs"))
        pkg_dir = installed.pkg_dir
    phase_a = CascadePhaseA(
        fasttext_model=fasttext_model_path(args.fasttext_model, args.cache_dir),
        extractor_pkg_dir=pkg_dir,
        extract_procs=args.extract_procs,
    )

    torch_device = args.torch_device or (default_torch_device() if args.backend is Backend.TORCH else "")
    device, n_devices = describe_device(args.backend, torch_device)
    load_options = {"precision": args.precision, "device": torch_device or "cpu"}
    phase_b = PhaseB(
        classify=functools.partial(
            run_phase_b,
            tokenize=functools.partial(tokenize_batch, load_tokenizer()),
            pooled=load_pooled(local_copy(args.pooled_model, args.cache_dir), args.backend, **load_options),
            terminal=load_terminal(local_copy(args.terminal_model, args.cache_dir), args.backend, **load_options),
        ),
        backend=str(args.backend),
        device=device,
        n_devices=n_devices,
    )
    logger.info("phase B: %s backend on %d x %s (%s)", args.backend, n_devices, device, args.precision)

    try:
        report = run_worker(
            warcs,
            store_for(root),
            root,
            phase_a,
            phase_b,
            phase_a_procs=args.phase_a_procs,
            queue_depth=args.queue_depth,
            stall_seconds=args.stall_minutes * 60,
        )
    except WorkerStalledError:
        logger.exception(
            "phase A stalled; exiting with status %d so a supervisor restarts this worker", STALL_EXIT_CODE
        )
        logging.shutdown()
        os._exit(STALL_EXIT_CODE)  # a hung phase-A process would block a normal interpreter exit
    logger.info("done: %s", report)
    if report.n_errors:
        logger.error("%d WARCs raised in phase A; rerun the same command to retry them", report.n_errors)
        sys.exit(1)
