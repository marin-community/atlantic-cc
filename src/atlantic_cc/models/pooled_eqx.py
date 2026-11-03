# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX-free reader for the as-run pooled-transformer checkpoint (Levanter/Equinox ``model.eqx``).

``eqx.tree_serialise_leaves`` writes each array leaf of ``PooledTransformerClassifier`` with
``np.save``, in pytree (dataclass field) order. :func:`state_dict_from_eqx` maps the leaves to the
PyTorch port's ``nn.Linear`` layout (weights transposed to [out, in]); the two unused query vectors
(``pool_query`` for ``attn`` window pooling, ``final_query`` for ``attn`` final pooling) are dropped.
"""

import numpy as np

LAYER_LEAVES = ("ln1_g", "ln1_b", "ln2_g", "ln2_b", "wqkv", "wo", "w1", "w2")
UNUSED_LEAVES = ("pool_query", "final_query")


def leaf_names(num_layers: int) -> list[str]:
    names = ["embed", "pool_query", "proj_w", "proj_b", "pos_embed"]
    for i in range(num_layers):
        names += [f"layers.{i}.{leaf}" for leaf in LAYER_LEAVES]
    return names + ["final_query", "head_g", "head_b", "head_w"]


def read_eqx_leaves(path: str, num_layers: int) -> dict[str, np.ndarray]:
    """All leaves of ``model.eqx``; raises if the file holds more or fewer arrays than expected."""
    names = leaf_names(num_layers)
    with open(path, "rb") as f:
        arrays = [np.load(f, allow_pickle=False) for _ in names]
        trailing = f.read()
    if trailing:
        raise ValueError(f"{path}: {len(trailing)} bytes left after {len(names)} leaves; wrong num_layers?")
    return dict(zip(names, arrays, strict=True))


def state_dict_from_eqx(leaves: dict[str, np.ndarray], num_layers: int) -> dict[str, np.ndarray]:
    """Leaves -> ``PooledTransformerForSequenceClassification`` state dict (float32, contiguous)."""
    sd = {
        "embed.weight": leaves["embed"],
        "proj.weight": leaves["proj_w"].T,
        "proj.bias": leaves["proj_b"],
        "pos_embed.weight": leaves["pos_embed"],
        "head_norm.weight": leaves["head_g"],
        "head_norm.bias": leaves["head_b"],
        "head.weight": leaves["head_w"].T,
    }
    for i in range(num_layers):
        p = f"layers.{i}."
        sd[p + "ln1.weight"] = leaves[p + "ln1_g"]
        sd[p + "ln1.bias"] = leaves[p + "ln1_b"]
        sd[p + "ln2.weight"] = leaves[p + "ln2_g"]
        sd[p + "ln2.bias"] = leaves[p + "ln2_b"]
        for name in ("wqkv", "wo", "w1", "w2"):
            sd[p + name + ".weight"] = leaves[p + name].T
    return {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in sd.items()}
