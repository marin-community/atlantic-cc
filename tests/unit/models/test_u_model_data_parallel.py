# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The JAX scorers shard batch rows over all local devices, as production did on 4- and 8-chip hosts.

This module sorts first so that its subprocess starts while the test process is still small.

Runs a subprocess with four virtual CPU devices and a batch of 7 rows (not a multiple of 4, so the
scorer adds a filler row) and compares with single-device scores of the same rows.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from atlantic_cc.models import Backend, Precision, load_pooled, load_terminal
from tests.unit.models import goldens
from tests.unit.models import random_models as rm

pytest.importorskip("jax")

REPO_ROOT = Path(__file__).resolve().parents[3]
ROWS = 7
CHILD = """
import json, sys
import jax
from atlantic_cc.models import Backend, Precision, load_pooled, load_terminal
from tests.unit.models import goldens
assert len(jax.local_devices()) == 4, jax.local_devices()
out = {}
for name, loader, batch, model_dir in (
    ("pooled", load_pooled, "pooled_tiny_256", sys.argv[1]),
    ("ettin", load_terminal, "ettin_tiny_128", sys.argv[2]),
):
    ids, mask, _ = goldens.batch(batch)
    out[name] = loader(model_dir, Backend.JAX, precision=Precision.FP32).score(ids[:7], mask[:7]).tolist()
print(json.dumps(out))
"""


def test_rows_sharded_over_four_devices_match_one_device(tmp_path):
    rm.write_pooled_hf(tmp_path / "pooled", rm.POOLED_SEED)
    rm.write_ettin_hf(tmp_path / "ettin", rm.ETTIN_SEED)
    env = {**os.environ, "XLA_FLAGS": "--xla_force_host_platform_device_count=4", "JAX_PLATFORMS": "cpu"}
    proc = subprocess.run(
        [sys.executable, "-c", CHILD, str(tmp_path / "pooled"), str(tmp_path / "ettin")],
        env=env,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    four = json.loads(proc.stdout.strip().splitlines()[-1])
    for name, loader, batch in (("pooled", load_pooled, "pooled_tiny_256"), ("ettin", load_terminal, "ettin_tiny_128")):
        ids, mask, _ = goldens.batch(batch)
        one = loader(str(tmp_path / name), Backend.JAX, precision=Precision.FP32).score(ids[:ROWS], mask[:ROWS])
        np.testing.assert_allclose(np.array(four[name], dtype=np.float32), one, rtol=0, atol=1e-6, equal_nan=True)
