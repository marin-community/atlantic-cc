# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the AS-RUN Marin rdedup (tag ``lpv11-8m-as-run``) on a list of kept files, as a parity reference.

This file runs inside a Marin checkout of the tag, never inside Atlantic-CC: it imports ``experiments.fast_curation
.rdedup`` and drives it the way its own ``test_pipeline`` does (``build_plan`` + ``write_plan``, then one
``worker.slot_loop``). The run root and output root use the as-run stores (GCS for ``gs://``, its ``LocalStore``
otherwise), so the reference run's plan, done marks, intermediates and outputs have the as-run layout that
``atlantic-cc dedup verify`` reads.

    git -C marin archive lpv11-8m-as-run | tar -x -C asrun && cp asrun_driver.py asrun/
    cd asrun && python asrun_driver.py --listing inputs.parquet --limit-files 32 --unit-gb 0.1 \\
        --run-root gs://.../t0/asrun/run --output-root gs://.../t0/asrun/out --cpus 16 --memory-gb 60
"""

import argparse
import contextlib
import dataclasses
import io
import json
import os
import platform
import socket
import subprocess
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from experiments.fast_curation.rdedup import stages, worker
from experiments.fast_curation.rdedup.plan import build_plan, write_plan
from experiments.fast_curation.rdedup.store import store_for
from experiments.fast_curation.rdedup.tasks import ClaimConfig
from numpy._core import _multiarray_umath


def _runtime(role: str) -> dict:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        np.show_runtime()
    lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout if platform.system() == "Linux" else ""
    features = _multiarray_umath.__cpu_features__
    return {
        "role": role,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "time": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "numpy_cpu_baseline": list(_multiarray_umath.__cpu_baseline__),
        "numpy_cpu_dispatch": list(_multiarray_umath.__cpu_dispatch__),
        "numpy_cpu_enabled": sorted(name for name, on in features.items() if on),
        "NPY_DISABLE_CPU_FEATURES": os.environ.get("NPY_DISABLE_CPU_FEATURES", ""),
        "numpy_show_runtime": buffer.getvalue(),
        "lscpu": lscpu,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--listing", required=True, help="Parquet of (path, size), already sorted by path.")
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--unit-gb", type=float, default=8.0)
    parser.add_argument("--unit-starts", default=None, help="JSON list overriding the packed unit starts.")
    parser.add_argument("--files-per-output", type=int, default=32)
    parser.add_argument("--exact-partitions", type=int, default=None)
    parser.add_argument("--lsh-partitions", type=int, default=None)
    parser.add_argument("--units-per-node-partition", type=int, default=None)
    parser.add_argument("--c075-partitions", type=int, default=None)
    parser.add_argument("--max-cluster-size", type=int, default=None, help="Monkeypatch (synthetic goldens only).")
    parser.add_argument("--solve-max-cluster-chars", type=int, default=None, help="Monkeypatch (goldens only).")
    parser.add_argument("--solve-max-document-chars", type=int, default=None, help="Monkeypatch (goldens only).")
    parser.add_argument("--text-max-document-chars", type=int, default=None, help="Monkeypatch (goldens only).")
    parser.add_argument("--cpus", type=int, required=True)
    parser.add_argument("--memory-gb", type=float, required=True)
    args = parser.parse_args()

    table = pq.read_table(args.listing, columns=["path", "size"])
    paths, sizes = table.column("path").to_pylist(), table.column("size").to_pylist()
    if args.limit_files is not None:
        paths, sizes = paths[: args.limit_files], sizes[: args.limit_files]
    if paths != sorted(paths):
        raise ValueError("listing must be sorted by path")
    plan = build_plan(
        run_root=args.run_root,
        output_root=args.output_root,
        spec_id="lpv11_fastpipe_v2_1_fused",
        sizes=sizes,
        unit_bytes=int(args.unit_gb * 1e9),
        rules=["c075"],
        files_per_output=args.files_per_output,
    )
    overrides = {
        "unit_starts": json.loads(args.unit_starts) if args.unit_starts else None,
        "exact_partitions": args.exact_partitions,
        "lsh_partitions": args.lsh_partitions,
        "units_per_node_partition": args.units_per_node_partition,
        "c075_partitions": args.c075_partitions,
    }
    plan = dataclasses.replace(plan, **{k: v for k, v in overrides.items() if v is not None})
    patches = {
        "MAX_CLUSTER_SIZE": args.max_cluster_size,
        "SOLVE_MAX_CLUSTER_CHARS": args.solve_max_cluster_chars,
        "SOLVE_MAX_DOCUMENT_CHARS": args.solve_max_document_chars,
        "TEXT_MAX_DOCUMENT_CHARS": args.text_max_document_chars,
    }
    for name, value in patches.items():
        if value is not None:
            if args.cpus != 1:
                raise ValueError("monkeypatched limits only reach in-process tasks: use --cpus 1")
            setattr(stages, name, value)
    store = store_for(plan.run_root)
    write_plan(store, plan, paths, sizes)
    info = _runtime("asrun")
    print(info["numpy_show_runtime"], info["lscpu"], flush=True)
    store.write(f"{plan.run_root}/runtime/asrun-{info['host']}-{info['pid']}.json", json.dumps(info, indent=1).encode())
    started = time.time()
    code = worker.slot_loop(0, plan.run_root, cpus=args.cpus, memory_gb=args.memory_gb, claims=ClaimConfig())
    print(f"as-run rdedup finished with {code} in {time.time() - started:.0f}s", flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
