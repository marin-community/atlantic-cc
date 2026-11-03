# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""PyTorch port of the Atlantic-CC pooled-transformer classifier.

This file is self-contained (torch + transformers only) because it is also published as the
``trust_remote_code`` modeling file of the Hugging Face repo ``marin-community/atlantic-cc-pooled-transformer``
(as ``modeling_pooled_transformer.py``). It reproduces Levanter's ``pooled_transformer`` classifier,
the model that produced the release's ``pooled_prob``::

    token ids --embed (f32)--> [B, T, E]
      --mask-aware mean / max / min over windows of ``pool_window`` real tokens--> [B, S, 3E]   (S = T / w)
      --linear (+ bias) + learned super-token position--> [B, S, D]
      --N pre-LayerNorm transformer blocks over super-tokens (no attention / MLP biases)--> [B, S, D]
      --mean over active super-tokens--> [B, D]
      --LayerNorm + linear (no bias)--> [B, num_labels] float32 logits

A window without real tokens is an inactive super-token: attention never attends to it and the final
pooling skips it, so padding is inert. Inputs are right-padded internally to a multiple of ``pool_window``.

Numerics. With ``bf16_matmul_inputs`` (as in training and production) every linear layer rounds its
input and weight to bfloat16 and accumulates in float32, like Levanter's ``_matmul`` on the TPU MXU.
With ``bf16_attention_inputs`` the operands of the two attention einsums (q, k, then the attention
weights and v) are rounded to bfloat16 as well, which is what a TPU does with float32 matmuls at
JAX's default precision. Everything else runs in float32. Production scored on TPUs, so both flags
default to on: on 2,000 released documents this puts the median |P - stored P| at 2e-7 (p99 2e-4),
against 1e-4 (p99 1e-3) without the attention rounding. Turn both off for a pure float32 forward.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

MODEL_TYPE = "marin_pooled_transformer"
POOL_KINDS = ("meanmaxmin",)
FINAL_POOLS = ("mean",)
NEG_INF = -1e30
# jax.nn.gelu(approximate=True) constants, as float32
_SQRT_2_OVER_PI = float(np.sqrt(2 / np.pi).astype(np.float32))
_GELU_CUBIC = 0.044715


class PooledTransformerConfig(PretrainedConfig):
    """Configuration of the pooled super-token transformer classifier.

    Only the released variant is supported: ``meanmaxmin`` window pooling and ``mean`` final pooling.
    The model learns one position embedding per super-token, so inputs longer than ``max_seq_len``
    tokens are rejected (truncate when tokenizing).
    """

    model_type = MODEL_TYPE

    def __init__(
        self,
        vocab_size: int = 50368,
        max_seq_len: int = 8192,
        pool_window: int = 64,
        pool_kind: str = "meanmaxmin",
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        final_pool: str = "mean",
        layer_norm_eps: float = 1e-5,
        bf16_matmul_inputs: bool = True,
        bf16_attention_inputs: bool = True,
        pad_token_id: int = 50283,
        **kwargs,
    ):
        if pool_kind not in POOL_KINDS:
            raise ValueError(f"pool_kind={pool_kind!r} is not supported; expected one of {POOL_KINDS}")
        if final_pool not in FINAL_POOLS:
            raise ValueError(f"final_pool={final_pool!r} is not supported; expected one of {FINAL_POOLS}")
        if max_seq_len % pool_window:
            raise ValueError(f"max_seq_len={max_seq_len} must be a multiple of pool_window={pool_window}")
        if hidden_dim % num_heads:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pool_window = pool_window
        self.pool_kind = pool_kind
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.final_pool = final_pool
        self.layer_norm_eps = layer_norm_eps
        self.bf16_matmul_inputs = bf16_matmul_inputs
        self.bf16_attention_inputs = bf16_attention_inputs
        self.pad_token_id = pad_token_id
        super().__init__(**kwargs)


def _round_bf16(x: torch.Tensor) -> torch.Tensor:
    """Round a float32 tensor to the nearest bfloat16 value, keeping float32 storage."""
    return x.to(torch.bfloat16).float()


def _linear(x: torch.Tensor, layer: nn.Linear, bf16_inputs: bool) -> torch.Tensor:
    """``x @ W.T`` (+ bias in float32). Products of bfloat16 values are exact in float32, so rounding the
    operands and multiplying in float32 equals a bf16-input / f32-accumulate matmul."""
    x, w = x.float(), layer.weight.float()
    if bf16_inputs:
        x, w = _round_bf16(x), _round_bf16(w)
    out = F.linear(x, w)
    if layer.bias is not None:
        out = out + layer.bias.float()
    return out


def _layer_norm(x: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
    # Levanter's _layer_norm: population variance, rsqrt, then scale and shift
    mu = x.mean(dim=-1, keepdim=True)
    var = (x - mu).square().mean(dim=-1, keepdim=True)
    return (x - mu) * torch.rsqrt(var + norm.eps) * norm.weight + norm.bias


def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    # Same operation order as jax.nn.gelu(approximate=True)
    cdf = 0.5 * (1.0 + torch.tanh(_SQRT_2_OVER_PI * (x + _GELU_CUBIC * (x * (x * x)))))
    return x * cdf


class PooledTransformerLayer(nn.Module):
    """Pre-LayerNorm transformer block over super-tokens."""

    def __init__(self, config: PooledTransformerConfig):
        super().__init__()
        d = config.hidden_dim
        self.num_heads = config.num_heads
        self.bf16_matmul_inputs = config.bf16_matmul_inputs
        self.bf16_attention_inputs = config.bf16_attention_inputs
        self.ln1 = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.wqkv = nn.Linear(d, 3 * d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.w1 = nn.Linear(d, d * config.mlp_ratio, bias=False)
        self.w2 = nn.Linear(d * config.mlp_ratio, d, bias=False)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        h = self.num_heads
        hd = d // h
        qkv = _linear(_layer_norm(x, self.ln1), self.wqkv, self.bf16_matmul_inputs).view(b, s, 3, h, hd)
        q, k, v = qkv.unbind(dim=2)  # [b, s, h, hd]
        if self.bf16_attention_inputs:
            q, k, v = _round_bf16(q), _round_bf16(k), _round_bf16(v)
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(hd)
        scores = scores.masked_fill(~valid[:, None, None, :].bool(), NEG_INF)
        attn = torch.softmax(scores, dim=-1)
        if self.bf16_attention_inputs:
            attn = _round_bf16(attn)
        ctx = torch.einsum("bhqk,bkhd->bqhd", attn, v).reshape(b, s, d)
        x = x + _linear(ctx, self.wo, self.bf16_matmul_inputs)
        hidden = _gelu_tanh(_linear(_layer_norm(x, self.ln2), self.w1, self.bf16_matmul_inputs))
        return x + _linear(hidden, self.w2, self.bf16_matmul_inputs)


class PooledTransformerPreTrainedModel(PreTrainedModel):
    config_class = PooledTransformerConfig
    base_model_prefix = "pooled"
    main_input_name = "input_ids"

    def _init_weights(self, module: nn.Module) -> None:
        # Only used for fresh (untrained) models; released weights are loaded over it.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


class PooledTransformerForSequenceClassification(PooledTransformerPreTrainedModel):
    """Pooled super-token transformer with a ``num_labels``-way head (label 1 = useful)."""

    def __init__(self, config: PooledTransformerConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.proj = nn.Linear(3 * config.embed_dim, config.hidden_dim, bias=True)
        self.pos_embed = nn.Embedding(config.max_seq_len // config.pool_window, config.hidden_dim)
        self.layers = nn.ModuleList(PooledTransformerLayer(config) for _ in range(config.num_layers))
        self.head_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.head = nn.Linear(config.hidden_dim, config.num_labels, bias=False)
        self.post_init()

    def _pool_windows(self, emb: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, e = emb.shape
        w = self.config.pool_window
        s = t // w
        wemb = emb.view(b, s, w, e)
        wmask = mask.view(b, s, w)
        counts = wmask.sum(dim=2, keepdim=True)  # [b, s, 1]
        valid = (counts[..., 0] > 0).float()  # [b, s]
        m3 = wmask[..., None]
        mean = (wemb * m3).sum(dim=2) / counts.clamp_min(1.0)
        active = valid[..., None] > 0
        real = m3 > 0
        mx = torch.where(active, torch.where(real, wemb, NEG_INF).amax(dim=2), 0.0)
        mn = torch.where(active, torch.where(real, wemb, -NEG_INF).amin(dim=2), 0.0)
        return torch.cat([mean, mx, mn], dim=-1), valid

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> SequenceClassifierOutput | tuple:
        cfg = self.config
        mask = (input_ids != cfg.pad_token_id) if attention_mask is None else attention_mask.bool()
        t = input_ids.shape[1]
        if t > cfg.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds max_seq_len={cfg.max_seq_len}; truncate when tokenizing")
        pad = (-t) % cfg.pool_window
        if pad:
            input_ids = F.pad(input_ids, (0, pad), value=cfg.pad_token_id)
            mask = F.pad(mask, (0, pad), value=False)
        emb = self.embed(input_ids).float()
        pooled, valid = self._pool_windows(emb, mask.float())
        s = pooled.shape[1]
        positions = torch.arange(s, device=input_ids.device)
        h = _linear(pooled, self.proj, cfg.bf16_matmul_inputs) + self.pos_embed(positions).float()
        for layer in self.layers:
            h = layer(h, valid)
        doc = (h * valid[..., None]).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        logits = _linear(_layer_norm(doc, self.head_norm), self.head, cfg.bf16_matmul_inputs)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels.view(-1))
        if return_dict is False:
            return (logits,) if loss is None else (loss, logits)
        return SequenceClassifierOutput(loss=loss, logits=logits)
