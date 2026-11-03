# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU-feature control and the runtime record every dedup process leaves behind.

The c075 solver's ``np.argpartition`` breaks ties in the order of the SIMD kernel NumPy picks at runtime. The
release was built on AVX2 hosts (TPU v4 VMs, AMD EPYC), so by default every dedup command masks AVX-512 through
``NPY_DISABLE_CPU_FEATURES`` (re-executing itself before any work) and refuses to run if the mask did not take.
"""

import contextlib
import io
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import time
from enum import StrEnum

import numpy as np
from numpy._core import _multiarray_umath

from atlantic_cc.io.store import Store

logger = logging.getLogger(__name__)

NUMPY_DISABLE_ENV = "NPY_DISABLE_CPU_FEATURES"
_AVX512_PREFIX = "AVX512"


class SimdPolicy(StrEnum):
    """Which NumPy SIMD kernels the solver may use."""

    MASK_AVX512 = "mask-avx512"  # AVX2 tie order on any x86_64 host: reproduces the release
    NATIVE = "native"  # whatever the CPU offers; AVX-512 hosts may remove different documents


def enabled_avx512_features() -> list[str]:
    """AVX-512 targets NumPy would dispatch to on this CPU."""
    features = _multiarray_umath.__cpu_features__
    return [name for name in _multiarray_umath.__cpu_dispatch__ if name.startswith(_AVX512_PREFIX) and features[name]]


def tie_order_platform() -> str:
    """Which ``np.argpartition`` kernel family breaks ties here: ``x86_64-avx512``, ``x86_64-avx2`` or
    ``<machine>-scalar`` (no SIMD selection kernel: ARM, or x86_64 with AVX2 masked). The release was built on
    ``x86_64-avx2``."""
    machine = platform.machine()
    features = _multiarray_umath.__cpu_features__
    if machine not in ("x86_64", "AMD64"):
        return f"{machine}-scalar"
    if features.get("AVX512_SKX"):
        return "x86_64-avx512"
    return "x86_64-avx2" if features.get("AVX2") else "x86_64-scalar"


def apply_simd_policy(policy: SimdPolicy) -> None:
    """Make this process obey ``policy``; for ``mask-avx512`` on an AVX-512 CPU this re-executes the process.

    The variable must be set before NumPy initializes, so the process replaces itself with the same command line
    and the mask in its environment; process pools it starts inherit the mask.
    """
    if policy is SimdPolicy.NATIVE:
        return
    enabled = enabled_avx512_features()
    if not enabled:
        return
    if os.environ.get(NUMPY_DISABLE_ENV):
        raise RuntimeError(f"{NUMPY_DISABLE_ENV}={os.environ[NUMPY_DISABLE_ENV]!r} left {enabled} enabled")
    logger.info("masking %s with %s and re-executing", enabled, NUMPY_DISABLE_ENV)
    sys.stdout.flush()
    sys.stderr.flush()
    environment = {**os.environ, NUMPY_DISABLE_ENV: " ".join(enabled)}
    os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]], environment)


def _numpy_runtime_text() -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        np.show_runtime()
    return buffer.getvalue()


def _lscpu() -> str:
    if platform.system() != "Linux":
        return ""
    return subprocess.run(["lscpu"], capture_output=True, text=True, check=True).stdout


def runtime_info() -> dict:
    """Host, interpreter and NumPy dispatch facts that bear on reproducibility."""
    features = _multiarray_umath.__cpu_features__
    return {
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
        "tie_order_platform": tie_order_platform(),
        NUMPY_DISABLE_ENV: os.environ.get(NUMPY_DISABLE_ENV, ""),
        "numpy_show_runtime": _numpy_runtime_text(),
        "lscpu": _lscpu(),
    }


def record_runtime(store: Store, run_root: str, role: str) -> dict:
    """Log :func:`runtime_info` and keep a copy at ``{run_root}/runtime/{role}-{host}-{pid}.json``."""
    info = {"role": role, **runtime_info()}
    logger.info(
        "runtime: %s numpy %s enabled=%s %s=%r",
        info["machine"],
        info["numpy"],
        info["numpy_cpu_enabled"],
        NUMPY_DISABLE_ENV,
        info[NUMPY_DISABLE_ENV],
    )
    logger.info("np.show_runtime():\n%s", info["numpy_show_runtime"])
    if info["lscpu"]:
        logger.info("lscpu:\n%s", info["lscpu"])
    path = f"{run_root.rstrip('/')}/runtime/{role}-{info['host']}-{info['pid']}.json"
    store.write(path, json.dumps(info, sort_keys=True, indent=1).encode())
    return info
