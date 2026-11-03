# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The two neural classifiers of the Atlantic-CC cascade, as PyTorch or JAX scorers.

``load_pooled`` gives the pooled-transformer band model (up to 8,192 tokens) and ``load_terminal`` the
ModernBERT terminal model, a fine-tuned Ettin-68M encoder (2,048 tokens). ``source`` is a Hugging Face
repo id (``spec.POOLED_HF_REPO``, ``spec.TERMINAL_HF_REPO``) or a local folder in the released repo
layout. Both backends take the same padded batches and return P(useful) = softmax(float32 logits)[:, 1].

``Precision.BF16`` (the default) uses production's numerics; ``Precision.FP32`` runs everything in
float32 and is the structural reference. The JAX backend issues Levanter's operations: on the TPU type
production used, with production's batching, it reproduces the stored scores bit for bit. The PyTorch
backend agrees statistically; see ``docs/classifiers.md`` for measured differences.
"""

import os
from enum import StrEnum
from typing import Protocol

import numpy as np
from huggingface_hub import snapshot_download


class Backend(StrEnum):
    TORCH = "torch"
    JAX = "jax"


class Precision(StrEnum):
    BF16 = "bf16"
    FP32 = "fp32"


class Scorer(Protocol):
    max_tokens: int

    def score(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """int32 ``[B, L]`` ids padded with 50283 and bool ``[B, L]`` (True = real token) -> float32 ``[B]``."""
        ...


# PyTorch ModernBERT attention kernel for the terminal model; both agree with Levanter to float noise.
DEFAULT_TORCH_ATTENTION = "sdpa"


def resolve_model_dir(source: str, revision: str | None = None) -> str:
    """Local folder for ``source``: the folder itself, or a (cached) snapshot of the Hub repo."""
    if os.path.isdir(source):
        return source
    return snapshot_download(repo_id=source, revision=revision)


def load_pooled(
    source: str,
    backend: Backend,
    *,
    precision: Precision = Precision.BF16,
    device: str = "cpu",
    revision: str | None = None,
) -> Scorer:
    """Pooled-transformer scorer. ``device`` applies to the PyTorch backend; JAX uses all local devices."""
    model_dir = resolve_model_dir(source, revision)
    if backend is Backend.TORCH:
        from atlantic_cc.models.torch_scorer import load_pooled_scorer  # noqa: PLC0415 (optional torch)

        if precision is Precision.FP32:
            return load_pooled_scorer(model_dir, device=device, bf16_matmul_inputs=False, bf16_attention_inputs=False)
        return load_pooled_scorer(model_dir, device=device)
    import jax.numpy as jnp  # noqa: PLC0415 (optional jax)

    from atlantic_cc.models import pooled_jax  # noqa: PLC0415 (optional jax)

    return pooled_jax.load_scorer(model_dir, matmul_dtype=jnp.bfloat16 if precision is Precision.BF16 else jnp.float32)


def load_terminal(
    source: str,
    backend: Backend,
    *,
    precision: Precision = Precision.BF16,
    device: str = "cpu",
    revision: str | None = None,
    attn_implementation: str = DEFAULT_TORCH_ATTENTION,
) -> Scorer:
    """Ettin-68M terminal scorer (``max_tokens`` 2,048). ``device`` and ``attn_implementation`` apply to PyTorch."""
    model_dir = resolve_model_dir(source, revision)
    if backend is Backend.TORCH:
        import torch  # noqa: PLC0415 (optional torch)

        from atlantic_cc.models import ettin_torch  # noqa: PLC0415 (optional torch)

        dtype = torch.bfloat16 if precision is Precision.BF16 else torch.float32
        return ettin_torch.load_scorer(model_dir, dtype=dtype, device=device, attn_implementation=attn_implementation)
    import jax.numpy as jnp  # noqa: PLC0415 (optional jax)

    from atlantic_cc.models import ettin_jax  # noqa: PLC0415 (optional jax)

    return ettin_jax.load_scorer(model_dir, dtype=jnp.bfloat16 if precision is Precision.BF16 else jnp.float32)
