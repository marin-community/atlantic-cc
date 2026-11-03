# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Data-parallel JAX scoring: parameters replicated, batch rows sharded over the local devices.

Production built ``Mesh(devices.reshape(n_dev, 1), ("data", "model"))`` and jitted the scorer with
the batch axis on ``"data"`` (as-run: ``experiments/fast_curation/fused_phase.py`` and
``tpu_phase.make_score_fn``), so each chip scored ``batch / n_dev`` rows of every batch. This
module reproduces that layout; XLA compiles one program per ``(rows, length)`` shape.
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

DATA_AXIS = "data"
MODEL_AXIS = "model"

ProbsFn = Callable[[Any, jax.Array, jax.Array], jax.Array]


def useful_probability(logits: jax.Array) -> jax.Array:
    """Softmax over the 2 labels in float32, class 1 (as-run ``tpu_phase.make_score_fn``)."""
    return jax.nn.softmax(logits.astype(jnp.float32), axis=-1)[:, 1]


class JaxScorer:
    """Scores padded token batches with a jitted ``probs_fn(params, ids, real) -> float32 [rows]``.

    ``ids`` and ``real`` are padded with extra all-pad rows up to a multiple of the device count and
    with pad columns up to a multiple of ``length_multiple``; the extra outputs are dropped.
    """

    def __init__(self, probs_fn: ProbsFn, params: Any, *, max_tokens: int, length_multiple: int, pad_token_id: int):
        devices = jax.local_devices()
        mesh = Mesh(np.array(devices).reshape(len(devices), 1), (DATA_AXIS, MODEL_AXIS))
        replicated = NamedSharding(mesh, PartitionSpec())
        rows = NamedSharding(mesh, PartitionSpec(DATA_AXIS))
        self.max_tokens = max_tokens
        self._n_devices = len(devices)
        self._length_multiple = length_multiple
        self._pad_token_id = pad_token_id
        self._params = jax.device_put(params, replicated)
        self._fn = jax.jit(probs_fn, in_shardings=(replicated, rows, rows), out_shardings=rows)

    def score(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """int32 ``[B, L]`` ids padded with the pad token and bool ``[B, L]`` real-token mask -> float32 ``[B]``."""
        b, t = input_ids.shape
        if t > self.max_tokens:
            raise ValueError(f"batch length {t} exceeds max_tokens={self.max_tokens}")
        if b == 0:
            return np.zeros((0,), dtype=np.float32)
        rows = b + (-b) % self._n_devices
        cols = t + (-t) % self._length_multiple
        ids = np.full((rows, cols), self._pad_token_id, dtype=np.int32)
        real = np.zeros((rows, cols), dtype=bool)
        ids[:b, :t] = input_ids
        real[:b, :t] = attention_mask
        return np.asarray(self._fn(self._params, ids, real), dtype=np.float32)[:b]
