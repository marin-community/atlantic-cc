# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face ``ModernBertForSequenceClassification`` loader for the Atlantic-CC terminal classifier.

The as-run checkpoint was saved by Levanter's MLM-oriented converter, so its ``config.json`` names
``ModernBertForMaskedLM``, has generic labels and legacy generation keys, and the weights have no
``classifier.bias`` (Levanter's classifier is bias-free). :func:`fixed_config` repairs the config;
the released copy also stores an explicit zero bias, so loading it initializes nothing.

``dtype=torch.bfloat16`` matches production (Levanter ran this model with bf16 weights and compute);
``torch.float32`` gives the fp32 reference.
"""

import torch
from transformers import ModernBertConfig, ModernBertForSequenceClassification

from atlantic_cc.models.torch_scorer import TorchScorer
from atlantic_cc.spec import TERMINAL_MAX_TOKENS

ID2LABEL = {0: "not_useful", 1: "useful"}
LABEL2ID = {label: i for i, label in ID2LABEL.items()}
ARCHITECTURE = "ModernBertForSequenceClassification"


def fixed_config(model_dir: str) -> ModernBertConfig:
    """The checkpoint's ModernBERT config with the classifier's architecture and label names.

    transformers drops the legacy generation keys (``max_length``, ``num_beams``, ...) on load, so a
    config saved from this object no longer carries them.
    """
    config = ModernBertConfig.from_pretrained(model_dir)
    config.architectures = [ARCHITECTURE]
    config.id2label = dict(ID2LABEL)
    config.label2id = dict(LABEL2ID)
    config.problem_type = "single_label_classification"
    return config


def load_model(model_dir: str, *, dtype: torch.dtype, attn_implementation: str) -> ModernBertForSequenceClassification:
    model = ModernBertForSequenceClassification.from_pretrained(
        model_dir, config=fixed_config(model_dir), dtype=dtype, attn_implementation=attn_implementation
    )
    return model.eval()


def load_scorer(
    model_dir: str, *, dtype: torch.dtype, device: str, attn_implementation: str, max_tokens: int = TERMINAL_MAX_TOKENS
) -> TorchScorer:
    model = load_model(model_dir, dtype=dtype, attn_implementation=attn_implementation)
    return TorchScorer(model, max_tokens=max_tokens, device=device)
