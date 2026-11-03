# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure-JAX forward of the pooled-transformer classifier with Levanter's numerics.

Line-for-line port of ``levanter.models.pooled_transformer.PooledTransformerClassifier._forward``
(as-run: ``lib/levanter/src/levanter/models/pooled_transformer.py``) with plain ``jax.numpy`` arrays,
so that on a given backend it lowers to the same operations as production: linear layers take
bfloat16 inputs and accumulate in float32, everything else (including the attention einsums, which
run at the backend's default matmul precision) is float32.
"""

import functools
import json
import math
import os
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file

from atlantic_cc.models.jax_scorer import JaxScorer, useful_probability

NEG_INF = -1e30
LN_EPS = 1e-5  # hard-coded in Levanter's _layer_norm

Params = dict[str, Any]


@dataclass(frozen=True)
class PooledJaxSpec:
    """Static hyperparameters of the forward pass."""

    pool_window: int
    num_heads: int
    max_seq_len: int
    matmul_dtype: Any  # jnp.bfloat16 reproduces production; jnp.float32 is the fp32 reference


def params_from_state_dict(state_dict: dict[str, np.ndarray], num_layers: int) -> Params:
    """Convert the Hugging Face (``nn.Linear`` [out, in]) state dict to Levanter's [in, out] layout."""
    sd = {k: np.asarray(v, dtype=np.float32) for k, v in state_dict.items()}
    layers = []
    for i in range(num_layers):
        p = f"layers.{i}."
        layers.append(
            {
                "ln1_g": sd[p + "ln1.weight"],
                "ln1_b": sd[p + "ln1.bias"],
                "ln2_g": sd[p + "ln2.weight"],
                "ln2_b": sd[p + "ln2.bias"],
                "wqkv": np.ascontiguousarray(sd[p + "wqkv.weight"].T),
                "wo": np.ascontiguousarray(sd[p + "wo.weight"].T),
                "w1": np.ascontiguousarray(sd[p + "w1.weight"].T),
                "w2": np.ascontiguousarray(sd[p + "w2.weight"].T),
            }
        )
    return {
        "embed": sd["embed.weight"],
        "proj_w": np.ascontiguousarray(sd["proj.weight"].T),
        "proj_b": sd["proj.bias"],
        "pos_embed": sd["pos_embed.weight"],
        "layers": layers,
        "head_g": sd["head_norm.weight"],
        "head_b": sd["head_norm.bias"],
        "head_w": np.ascontiguousarray(sd["head.weight"].T),
    }


def _matmul(x: jax.Array, w: jax.Array, dtype: Any) -> jax.Array:
    out = jnp.matmul(x.astype(dtype), w.astype(dtype), preferred_element_type=jnp.float32)
    return out.astype(jnp.float32)


def _layer_norm(x: jax.Array, gamma: jax.Array, beta: jax.Array) -> jax.Array:
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) * jax.lax.rsqrt(var + LN_EPS) * gamma + beta


def _layer(layer: Params, x: jax.Array, valid: jax.Array, spec: PooledJaxSpec) -> jax.Array:
    b, s, d = x.shape
    h, hd = spec.num_heads, d // spec.num_heads
    normed = _layer_norm(x, layer["ln1_g"], layer["ln1_b"])
    qkv = _matmul(normed, layer["wqkv"], spec.matmul_dtype).reshape(b, s, 3, h, hd)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # [b, s, h, hd]
    scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(hd)
    scores = jnp.where(valid[:, None, None, :].astype(bool), scores, NEG_INF)
    attn = jax.nn.softmax(scores, axis=-1)
    ctx = jnp.einsum("bhqk,bkhd->bqhd", attn, v).reshape(b, s, d)
    x = x + _matmul(ctx, layer["wo"], spec.matmul_dtype)
    normed = _layer_norm(x, layer["ln2_g"], layer["ln2_b"])
    mlp = _matmul(jax.nn.gelu(_matmul(normed, layer["w1"], spec.matmul_dtype)), layer["w2"], spec.matmul_dtype)
    return x + mlp


def _pool_windows(emb: jax.Array, mask: jax.Array, pool_window: int) -> tuple[jax.Array, jax.Array]:
    b, t, e = emb.shape
    s, w = t // pool_window, pool_window
    wemb = emb.reshape(b, s, w, e)
    wmask = mask.reshape(b, s, w)
    counts = wmask.sum(axis=2, keepdims=True)
    valid = (counts[..., 0] > 0).astype(jnp.float32)
    denom = jnp.maximum(counts, 1.0)
    m3 = wmask[..., None]
    mean = (wemb * m3).sum(axis=2) / denom
    mx = jnp.where(valid[..., None] > 0, jnp.where(m3 > 0, wemb, NEG_INF).max(axis=2), 0.0)
    mn = jnp.where(valid[..., None] > 0, jnp.where(m3 > 0, wemb, -NEG_INF).min(axis=2), 0.0)
    return jnp.concatenate([mean, mx, mn], axis=-1), valid


def forward(params: Params, ids: jax.Array, mask: jax.Array, spec: PooledJaxSpec) -> jax.Array:
    """``[b, t]`` int32 ids and float32 real-token mask (t a multiple of ``pool_window``) -> f32 logits."""
    t = ids.shape[1]
    if t % spec.pool_window:
        raise ValueError(f"sequence length {t} must be divisible by pool_window={spec.pool_window}")
    if t > spec.max_seq_len:
        raise ValueError(f"sequence length {t} exceeds max_seq_len={spec.max_seq_len}")
    s = t // spec.pool_window
    emb = jnp.take(params["embed"], ids, axis=0).astype(jnp.float32)
    pooled, valid = _pool_windows(emb, mask, spec.pool_window)
    h = _matmul(pooled, params["proj_w"], spec.matmul_dtype) + params["proj_b"] + params["pos_embed"][:s]
    for layer in params["layers"]:
        h = _layer(layer, h, valid, spec)
    pooled_doc = (h * valid[..., None]).sum(axis=1) / jnp.maximum(valid.sum(axis=1, keepdims=True), 1.0)
    normed = _layer_norm(pooled_doc, params["head_g"], params["head_b"])
    return _matmul(normed, params["head_w"], spec.matmul_dtype)


def _probs(params: Params, ids: jax.Array, real: jax.Array, spec: PooledJaxSpec) -> jax.Array:
    return useful_probability(forward(params, ids, real.astype(jnp.float32), spec))


def load_model(model_dir: str, *, matmul_dtype: Any) -> tuple[Params, PooledJaxSpec, int]:
    """``(params, spec, pad_token_id)`` from a staged Hugging Face folder (``config.json`` + ``model.safetensors``)."""
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    if cfg["pool_kind"] != "meanmaxmin" or cfg["final_pool"] != "mean":
        raise ValueError(f"unsupported pooled config: pool_kind={cfg['pool_kind']!r} final_pool={cfg['final_pool']!r}")
    spec = PooledJaxSpec(
        pool_window=cfg["pool_window"],
        num_heads=cfg["num_heads"],
        max_seq_len=cfg["max_seq_len"],
        matmul_dtype=matmul_dtype,
    )
    params = params_from_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), cfg["num_layers"])
    return params, spec, cfg["pad_token_id"]


def load_scorer(model_dir: str, *, matmul_dtype: Any) -> JaxScorer:
    """Data-parallel scorer from a staged Hugging Face folder."""
    params, spec, pad_token_id = load_model(model_dir, matmul_dtype=matmul_dtype)
    return JaxScorer(
        functools.partial(_probs, spec=spec),
        params,
        max_tokens=spec.max_seq_len,
        length_multiple=spec.pool_window,
        pad_token_id=pad_token_id,
    )
