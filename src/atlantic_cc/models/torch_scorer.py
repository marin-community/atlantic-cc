# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""PyTorch scoring for any Hugging Face sequence classifier with label 1 = useful."""

import numpy as np
import torch
from transformers import PreTrainedModel

from atlantic_cc.models.pooled_torch import PooledTransformerForSequenceClassification


def useful_probability(logits: torch.Tensor) -> torch.Tensor:
    """Softmax over the labels in float32 with JAX's formula (``exp(x - max) / sum``), class 1."""
    logits = logits.float()
    e = torch.exp(logits - logits.amax(dim=-1, keepdim=True))
    return (e / e.sum(dim=-1, keepdim=True))[:, 1]


class TorchScorer:
    """Scores padded token batches with a Hugging Face model in eval mode on ``device``."""

    def __init__(self, model: PreTrainedModel, *, max_tokens: int, device: str):
        self.max_tokens = max_tokens
        self.model = model.to(device).eval()
        self._device = device

    @torch.inference_mode()
    def logits(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> torch.Tensor:
        b, t = input_ids.shape
        if t > self.max_tokens:
            raise ValueError(f"batch length {t} exceeds max_tokens={self.max_tokens}")
        ids = torch.from_numpy(np.asarray(input_ids, dtype=np.int64)).to(self._device)
        mask = torch.from_numpy(np.asarray(attention_mask, dtype=np.int64)).to(self._device)
        return self.model(input_ids=ids, attention_mask=mask).logits

    def score(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """int ``[B, L]`` ids padded with the pad token and bool ``[B, L]`` real-token mask -> float32 ``[B]``."""
        if input_ids.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        probs = useful_probability(self.logits(input_ids, attention_mask))
        return probs.cpu().numpy().astype(np.float32)


def load_pooled_scorer(model_dir: str, *, device: str, **config_overrides) -> TorchScorer:
    """Scorer for a staged pooled-transformer folder; ``config_overrides`` set numerics flags."""
    model = PooledTransformerForSequenceClassification.from_pretrained(model_dir, **config_overrides)
    return TorchScorer(model, max_tokens=model.config.max_seq_len, device=device)
