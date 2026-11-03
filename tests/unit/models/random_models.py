# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tiny deterministic random-weight classifiers in the released file layouts.

Shared by the offline unit tests and ``tests/golden/models/make_goldens.py``, which loads the same
files with the as-run Levanter code to produce the committed reference logits. The models keep the
production vocabulary (so real special-token ids are valid) but are a few layers of width 32, so the
default test suite runs them in seconds. Every parameter is perturbed away from its initializer
(LayerNorm gains and biases too) so that each one affects the output. The tiny Ettin uses different
global and local RoPE thetas and a narrow local window so that both attention variants matter at
test lengths.
"""

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from atlantic_cc.models import pooled_eqx

VOCAB_SIZE = 50368  # ModernBERT's padded vocabulary; covers the special-token ids
PAD, CLS, SEP = 50283, 50281, 50282
BODY_TOKEN_LIMIT = 50280  # ordinary BPE ids are below the added tokens
POOLED_SEED = 1
ETTIN_SEED = 2

POOLED_TINY = {
    "max_seq_len": 512,
    "pool_window": 64,
    "pool_kind": "meanmaxmin",
    "embed_dim": 16,
    "hidden_dim": 32,
    "num_layers": 2,
    "num_heads": 2,
    "mlp_ratio": 2,
    "dropout": 0.0,
    "final_pool": "mean",
    "num_labels": 2,
    "pad_token_id": PAD,
}

ETTIN_TINY = {
    "model_type": "modernbert",
    "architectures": ["ModernBertForSequenceClassification"],
    "vocab_size": VOCAB_SIZE,
    "hidden_size": 32,
    "intermediate_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 2,
    "global_attn_every_n_layers": 3,
    "local_attention": 16,
    "global_rope_theta": 160000.0,
    "local_rope_theta": 10000.0,
    "max_position_embeddings": 512,
    "norm_eps": 1e-5,
    "norm_bias": False,
    "attention_bias": False,
    "mlp_bias": False,
    "classifier_bias": False,
    "classifier_pooling": "mean",
    "hidden_activation": "gelu",
    "classifier_activation": "gelu",
    "pad_token_id": PAD,
    "bos_token_id": CLS,
    "cls_token_id": CLS,
    "eos_token_id": SEP,
    "sep_token_id": SEP,
    "id2label": {"0": "not_useful", "1": "useful"},
    "label2id": {"not_useful": 0, "useful": 1},
}


def _draw(rng: np.random.Generator, shape: tuple[int, ...], kind: str) -> np.ndarray:
    x = rng.standard_normal(shape, dtype=np.float32)
    if kind == "gain":
        return (1.0 + 0.2 * x).astype(np.float32)
    if kind == "bias":
        return (0.1 * x).astype(np.float32)
    if kind == "embed":
        return (0.5 * x).astype(np.float32)
    fan_in = shape[0] if kind.startswith("in_out") else shape[-1]
    return (x / np.sqrt(fan_in) * (2.0 if kind.endswith("head") else 1.0)).astype(np.float32)


def pooled_leaves(seed: int, cfg: dict = POOLED_TINY) -> dict[str, np.ndarray]:
    """Leaves of a random pooled model in Levanter's layout and order ([in, out] matrices)."""
    e, d, f = cfg["embed_dim"], cfg["hidden_dim"], cfg["hidden_dim"] * cfg["mlp_ratio"]
    shapes = {
        "embed": ((VOCAB_SIZE, e), "embed"),
        "pool_query": ((e,), "bias"),
        "proj_w": ((3 * e, d), "in_out"),
        "proj_b": ((d,), "bias"),
        "pos_embed": ((cfg["max_seq_len"] // cfg["pool_window"], d), "bias"),
        "final_query": ((d,), "bias"),
        "head_g": ((d,), "gain"),
        "head_b": ((d,), "bias"),
        "head_w": ((d, cfg["num_labels"]), "in_out_head"),
    }
    for i in range(cfg["num_layers"]):
        p = f"layers.{i}."
        shapes |= {
            p + "ln1_g": ((d,), "gain"),
            p + "ln1_b": ((d,), "bias"),
            p + "ln2_g": ((d,), "gain"),
            p + "ln2_b": ((d,), "bias"),
            p + "wqkv": ((d, 3 * d), "in_out"),
            p + "wo": ((d, d), "in_out"),
            p + "w1": ((d, f), "in_out"),
            p + "w2": ((f, d), "in_out"),
        }
    rng = np.random.default_rng(seed)
    drawn = {name: _draw(rng, shape, kind) for name, (shape, kind) in sorted(shapes.items())}
    return {name: drawn[name] for name in pooled_eqx.leaf_names(cfg["num_layers"])}


def write_pooled_levanter(directory: Path, seed: int) -> None:
    """``model.eqx`` (``np.save`` per leaf, as ``eqx.tree_serialise_leaves`` writes it) + Levanter config."""
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "model.eqx", "wb") as f:
        for leaf in pooled_leaves(seed).values():
            np.save(f, leaf)
    (directory / "config.json").write_text(json.dumps({"config_class": "PooledTransformerConfig", **POOLED_TINY}))


def write_pooled_hf(directory: Path, seed: int) -> None:
    """The same model in the released Hugging Face layout (``config.json`` + ``model.safetensors``)."""
    directory.mkdir(parents=True, exist_ok=True)
    cfg = {k: v for k, v in POOLED_TINY.items() if k not in ("dropout", "num_labels")}
    config = {
        "model_type": "marin_pooled_transformer",
        "vocab_size": VOCAB_SIZE,
        "layer_norm_eps": 1e-5,
        "bf16_matmul_inputs": True,
        "bf16_attention_inputs": True,
        "id2label": {"0": "not_useful", "1": "useful"},
        "label2id": {"not_useful": 0, "useful": 1},
        **cfg,
    }
    (directory / "config.json").write_text(json.dumps(config))
    state = pooled_eqx.state_dict_from_eqx(pooled_leaves(seed), POOLED_TINY["num_layers"])
    save_file(state, str(directory / "model.safetensors"), metadata={"format": "pt"})


def ettin_state_dict(seed: int, cfg: dict = ETTIN_TINY) -> dict[str, np.ndarray]:
    """Random ``ModernBertForSequenceClassification`` tensors (Hugging Face names, [out, in] matrices)."""
    d, m = cfg["hidden_size"], cfg["intermediate_size"]
    shapes = {
        "model.embeddings.tok_embeddings.weight": ((VOCAB_SIZE, d), "embed"),
        "model.embeddings.norm.weight": ((d,), "gain"),
        "model.final_norm.weight": ((d,), "gain"),
        "head.dense.weight": ((d, d), "out_in"),
        "head.norm.weight": ((d,), "gain"),
        "classifier.weight": ((len(cfg["id2label"]), d), "out_in_head"),
    }
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        if i:
            shapes[p + "attn_norm.weight"] = ((d,), "gain")
        shapes |= {
            p + "attn.Wqkv.weight": ((3 * d, d), "out_in"),
            p + "attn.Wo.weight": ((d, d), "out_in"),
            p + "mlp_norm.weight": ((d,), "gain"),
            p + "mlp.Wi.weight": ((2 * m, d), "out_in"),
            p + "mlp.Wo.weight": ((d, m), "out_in"),
        }
    rng = np.random.default_rng(seed)
    state = {name: _draw(rng, shape, kind) for name, (shape, kind) in sorted(shapes.items())}
    state["classifier.bias"] = np.zeros(2, dtype=np.float32)
    return state


def write_ettin_hf(directory: Path, seed: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(ETTIN_TINY))
    save_file(ettin_state_dict(seed), str(directory / "model.safetensors"), metadata={"format": "pt"})


def padded_batch(ids: np.ndarray, lengths: list[int], length: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows ``ids[sum(lengths[:i]):][:lengths[i]]`` right-padded with [PAD] to ``length`` + real-token mask."""
    input_ids = np.full((len(lengths), length), PAD, dtype=np.int32)
    mask = np.zeros((len(lengths), length), dtype=bool)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    for i, n in enumerate(lengths):
        input_ids[i, :n] = ids[offsets[i] : offsets[i] + n]
        mask[i, :n] = True
    return input_ids, mask


def random_documents(seed: int, lengths: list[int]) -> np.ndarray:
    """Concatenated synthetic token rows ``[CLS] body [SEP]`` (a length-0 row is an empty filler row)."""
    rng = np.random.default_rng(seed)
    rows = [[CLS, *rng.integers(0, BODY_TOKEN_LIMIT, n - 2).tolist(), SEP] if n else [] for n in lengths]
    return np.array([t for row in rows for t in row], dtype=np.int32)
