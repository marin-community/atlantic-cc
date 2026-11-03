# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure-JAX ModernBERT (Ettin-68M) sequence classifier with Levanter's all-bfloat16 numerics.

Port of ``levanter.models.modernbert.ModernBertForSequenceClassification`` as the Atlantic-CC worker ran
it (``load_hf_sequence_classifier(..., dtype=bfloat16)``, vanilla attention, ``classifier_pooling="mean"``).
Every op keeps Levanter's dtype and order. On TPU v4 this reproduces production's scores bit for bit; on
XLA:CPU some float32 dot products accumulate in a different order than Levanter's (max |Δlogit| 3e-6 in
float32), which bf16 rounding occasionally amplifies. The numerics:

- weights rounded to the compute dtype once at load (round-to-nearest-even);
- LayerNorm without bias: mean and variance computed in float32 from the bf16 input and rounded to
  bf16, then ``(x - mean) * rsqrt(var + eps)`` and the gain in bf16;
- RoPE (theta from the config for global and local layers): inverse frequencies in float32 at trace
  time, cos/sin in float32 then cast to bf16, rotate-half;
- attention: q scaled by ``1/sqrt(head_dim)`` in bf16, bf16 q.k, ``where(mask, w, -1e9)``, bf16
  softmax, bf16 weights.v; the mask is "same segment" (pads only see pads) and, on local layers, a
  symmetric window ``|i - j| <= local_attention // 2``; every ``global_every``-th layer is global;
- GLU MLP with exact (erf) GELU; final LayerNorm;
- mean pool over real tokens in bf16 (sums accumulate in float32, the token count is rounded to bf16);
- head dense -> exact GELU -> LayerNorm, bias-free classifier, bf16 logits; softmax in float32.
"""

import functools
import math
import os
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file
from transformers import ModernBertConfig

from atlantic_cc.models.jax_scorer import JaxScorer, useful_probability
from atlantic_cc.spec import TERMINAL_MAX_TOKENS

MASK_FILL = -1e9  # as-run: levanter/layers/attention.py simple_attention_with_dropout

Params = dict[str, Any]


@dataclass(frozen=True)
class EttinJaxSpec:
    """Static hyperparameters of the forward pass."""

    num_heads: int
    head_dim: int
    global_every: int
    local_radius: int
    global_rope_theta: float
    local_rope_theta: float
    norm_eps: float
    dtype: Any  # jnp.bfloat16 reproduces production; jnp.float32 is the fp32 reference

    def is_global(self, layer_idx: int) -> bool:
        return layer_idx % self.global_every == 0


def params_from_state_dict(state_dict: dict[str, np.ndarray], *, num_layers: int, num_heads: int, dtype: Any) -> Params:
    """Hugging Face ``ModernBertForSequenceClassification`` tensors -> Levanter axis layout in ``dtype``.

    ``Wqkv`` [3*D, D] becomes [3, heads, head_dim, D], attention ``Wo`` [D, D] becomes [D, heads,
    head_dim] and ``Wi`` [2*M, D] becomes [2, M, D], exactly as Levanter unflattens them.
    """

    def cast(name: str) -> np.ndarray:
        return np.asarray(state_dict[name], dtype=np.float32).astype(dtype)

    if "classifier.bias" in state_dict and np.any(np.asarray(state_dict["classifier.bias"]) != 0):
        raise ValueError("classifier.bias is non-zero; the as-run classifier has no bias")
    d = state_dict["model.embeddings.tok_embeddings.weight"].shape[1]
    hd = d // num_heads
    layers = []
    for i in range(num_layers):
        p = f"model.layers.{i}."
        wi = cast(p + "mlp.Wi.weight")
        layers.append(
            {
                "attn_norm": None if i == 0 else cast(p + "attn_norm.weight"),
                "wqkv": cast(p + "attn.Wqkv.weight").reshape(3, num_heads, hd, d),
                "wo": cast(p + "attn.Wo.weight").reshape(d, num_heads, hd),
                "mlp_norm": cast(p + "mlp_norm.weight"),
                "wi": wi.reshape(2, wi.shape[0] // 2, d),
                "mlp_wo": cast(p + "mlp.Wo.weight"),
            }
        )
    return {
        "tok_embeddings": cast("model.embeddings.tok_embeddings.weight"),
        "emb_norm": cast("model.embeddings.norm.weight"),
        "layers": layers,
        "final_norm": cast("model.final_norm.weight"),
        "head_dense": cast("head.dense.weight"),
        "head_norm": cast("head.norm.weight"),
        "classifier": cast("classifier.weight"),
    }


def _layer_norm(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:
    # haliax.nn.LayerNorm (no bias): jnp.mean / jnp.var compute in float32 and return x.dtype
    mean = jnp.mean(x, axis=-1)
    var = jnp.var(x, axis=-1)
    inv = jax.lax.rsqrt(var + eps)
    out = ((x - mean[..., None]) * inv[..., None]).astype(x.dtype)
    return weight.astype(out.dtype) * out


def _rope(x: jax.Array, theta: float) -> jax.Array:
    """Rotary embedding of ``x`` [b, t, h, d] (levanter.layers.rotary.DefaultRotaryEmbeddings)."""
    t, hd = x.shape[1], x.shape[3]
    with jax.ensure_compile_time_eval():
        half = jax.lax.iota(jnp.int32, hd // 2) * 2  # hax.arange(HeadHalfSize, step=2)
        inv_freq = 1.0 / (theta ** (half / hd))
        inv_freq = inv_freq / 1.0  # scale factor of the default rope config
    pos = jax.lax.iota(jnp.int32, t)
    freqs = inv_freq[None, :] * pos[:, None]  # [t, d/2] float32
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    cos = jnp.cos(emb).astype(x.dtype)[None, :, None, :]
    sin = jnp.sin(emb).astype(x.dtype)[None, :, None, :]
    x1, x2 = x[..., : hd // 2], x[..., hd // 2 :]
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    return x * cos + rotated * sin


def _attention(layer: Params, x: jax.Array, mask: jax.Array, theta: float, spec: EttinJaxSpec) -> jax.Array:
    qkv = jnp.einsum("bte,khde->btkhd", x, layer["wqkv"])
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # [b, t, h, d]
    q = _rope(q, theta).astype(q.dtype)
    k = _rope(k, theta).astype(k.dtype)
    q = q * (1 / math.sqrt(spec.head_dim))
    weights = jnp.einsum("bqhd,bkhd->bqhk", q, k)
    weights = jnp.where(mask[:, :, None, :], weights, MASK_FILL)
    weights = jax.nn.softmax(weights, axis=-1).astype(q.dtype)
    out = jnp.einsum("bqhk,bkhd->bqhd", weights, v).astype(x.dtype)
    return jnp.einsum("bthd,ehd->bte", out, layer["wo"])


def _mlp(layer: Params, x: jax.Array) -> jax.Array:
    h = jnp.einsum("bte,gme->btgm", x, layer["wi"])
    return jnp.einsum("btm,em->bte", jax.nn.gelu(h[:, :, 0], approximate=False) * h[:, :, 1], layer["mlp_wo"])


def logits_fn(params: Params, ids: jax.Array, real: jax.Array, spec: EttinJaxSpec) -> jax.Array:
    """``[b, t]`` int32 ids and bool real-token mask -> ``[b, num_labels]`` logits in ``spec.dtype``."""
    t = ids.shape[1]
    x = params["tok_embeddings"][ids]
    x = _layer_norm(x, params["emb_norm"], spec.norm_eps)
    same_segment = real[:, :, None] == real[:, None, :]  # [b, q, k]
    diff = jax.lax.iota(jnp.int32, t)[:, None] - jax.lax.iota(jnp.int32, t)[None, :]
    window = (diff <= spec.local_radius) & (diff >= -spec.local_radius)
    local_mask = same_segment & window[None]
    for i, layer in enumerate(params["layers"]):
        attn_in = x if layer["attn_norm"] is None else _layer_norm(x, layer["attn_norm"], spec.norm_eps)
        if spec.is_global(i):
            x = x + _attention(layer, attn_in, same_segment, spec.global_rope_theta, spec)
        else:
            x = x + _attention(layer, attn_in, local_mask, spec.local_rope_theta, spec)
        x = x + _mlp(layer, _layer_norm(x, layer["mlp_norm"], spec.norm_eps))
    x = _layer_norm(x, params["final_norm"], spec.norm_eps)
    weight = real.astype(x.dtype)
    denom = weight.sum(axis=1)
    pooled = (x * weight[:, :, None]).sum(axis=1) / denom[:, None]
    h = jax.nn.gelu(jnp.einsum("be,oe->bo", pooled, params["head_dense"]), approximate=False)
    h = _layer_norm(h, params["head_norm"], spec.norm_eps)
    return jnp.einsum("be,le->bl", h, params["classifier"])


def spec_from_config(config: ModernBertConfig, dtype: Any) -> EttinJaxSpec:
    """Read the forward-pass hyperparameters, rejecting ModernBERT options this port does not implement."""
    unsupported = {
        "hidden_activation": config.hidden_activation != "gelu",
        "classifier_activation": config.classifier_activation != "gelu",
        "classifier_pooling": config.classifier_pooling != "mean",
        "norm_bias": config.norm_bias,
        "attention_bias": config.attention_bias,
        "mlp_bias": config.mlp_bias,
        "classifier_bias": config.classifier_bias,
    }
    if any(unsupported.values()):
        raise ValueError(f"unsupported ModernBERT options: {[k for k, bad in unsupported.items() if bad]}")
    global_layers = [i for i, kind in enumerate(config.layer_types) if kind == "full_attention"]
    global_every = global_layers[1] - global_layers[0] if len(global_layers) >= 2 else config.num_hidden_layers
    expected = [
        "full_attention" if i % global_every == 0 else "sliding_attention" for i in range(config.num_hidden_layers)
    ]
    if list(config.layer_types) != expected:
        raise ValueError(f"layer_types {config.layer_types} are not periodic with period {global_every}")
    return EttinJaxSpec(
        num_heads=config.num_attention_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        global_every=global_every,
        local_radius=config.local_attention // 2,
        global_rope_theta=config.rope_parameters["full_attention"]["rope_theta"],
        local_rope_theta=config.rope_parameters["sliding_attention"]["rope_theta"],
        norm_eps=config.norm_eps,
        dtype=dtype,
    )


def _probs(params: Params, ids: jax.Array, real: jax.Array, spec: EttinJaxSpec) -> jax.Array:
    return useful_probability(logits_fn(params, ids, real, spec))


def load_model(model_dir: str, *, dtype: Any) -> tuple[Params, EttinJaxSpec, int]:
    """``(params, spec, pad_token_id)`` from a staged Hugging Face folder (``config.json`` + ``model.safetensors``)."""
    config = ModernBertConfig.from_pretrained(model_dir)
    params = params_from_state_dict(
        load_file(os.path.join(model_dir, "model.safetensors")),
        num_layers=config.num_hidden_layers,
        num_heads=config.num_attention_heads,
        dtype=dtype,
    )
    return params, spec_from_config(config, dtype), config.pad_token_id


def load_scorer(model_dir: str, *, dtype: Any, max_tokens: int = TERMINAL_MAX_TOKENS) -> JaxScorer:
    """Data-parallel scorer from a staged Hugging Face folder."""
    params, spec, pad_token_id = load_model(model_dir, dtype=dtype)
    return JaxScorer(
        functools.partial(_probs, spec=spec),
        params,
        max_tokens=max_tokens,
        length_multiple=1,
        pad_token_id=pad_token_id,
    )
