# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""lm-eval-harness loading glue for the eval-corpus rebuild (best effort).

Port of Marin ``experiments/datakit/decontam/lmh_loader.py`` (tag ``lpv11-8m-as-run``):

* :func:`trust_remote_code_for_hf` patches ``datasets.load_dataset``/``load_dataset_builder`` to pass
  ``trust_remote_code=True`` (retrying without it for builders that reject the keyword) and to resolve
  legacy no-namespace dataset names (``gsm8k`` -> ``openai/gsm8k``), which ``huggingface_hub`` >= 1.0
  rejects;
* :func:`flatten_task_dict` yields the leaf tasks of a ``get_task_dict`` result;
* :func:`materialize_first_nonempty_split` picks ``test``, else ``validation``, else ``training``.

``datasets``, ``transformers`` and ``lm_eval`` are optional dependencies of the rebuild only (pin:
``stanford-crfm/lm-evaluation-harness@d5e3391f22cde186c827674d5c3ec7c5f4fe0cab``); they are imported
inside the functions that need them.
"""

import functools
import logging
import os
import sys
from collections.abc import Iterator
from typing import Any

from huggingface_hub import HfApi

logger = logging.getLogger(__name__)

LM_EVAL_REPO = "https://github.com/stanford-crfm/lm-evaluation-harness"
LM_EVAL_COMMIT = "d5e3391f22cde186c827674d5c3ec7c5f4fe0cab"

# ``load_dataset`` names that are local builders or paths, never Hub repos.
_PACKAGED_BUILDERS = frozenset(
    {"json", "csv", "parquet", "text", "arrow", "pandas", "imagefolder", "audiofolder", "webdataset", "xml", "sql"}
)


@functools.cache
def canonical_dataset_id(path: str) -> str:
    """Hub repo id that a legacy no-namespace dataset name now lives at (``gsm8k`` -> ``openai/gsm8k``)."""
    if "/" in path or path in _PACKAGED_BUILDERS or os.path.exists(path):
        return path
    return HfApi().dataset_info(path).id


def _with_canonical_path(args: tuple, kwargs: dict) -> tuple[tuple, dict]:
    if args and isinstance(args[0], str):
        return (canonical_dataset_id(args[0]), *args[1:]), kwargs
    if isinstance(kwargs.get("path"), str):
        return args, {**kwargs, "path": canonical_dataset_id(kwargs["path"])}
    return args, kwargs


def trust_remote_code_for_hf() -> None:
    """Pass ``trust_remote_code=True`` to every ``datasets.load_dataset`` call; fall back on kwarg rejection.

    Also sets ``HF_ALLOW_CODE_EVAL=1`` (humaneval's metric) and aliases
    ``transformers.AutoModelForVision2Seq`` (removed in transformers 5, imported by the pinned lm-eval).
    """
    import datasets  # noqa: PLC0415  (optional rebuild dependency)
    import transformers  # noqa: PLC0415

    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    if not hasattr(transformers, "AutoModelForVision2Seq"):
        transformers.AutoModelForVision2Seq = transformers.AutoModel  # type: ignore[attr-defined]

    original_load = datasets.load_dataset
    original_builder = datasets.load_dataset_builder

    def _is_trc_rejection(exc: BaseException) -> bool:
        return isinstance(exc, (TypeError, ValueError)) and "trust_remote_code" in str(exc)

    def patched_load_dataset(*args, **kwargs):
        args, kwargs = _with_canonical_path(args, kwargs)
        if "trust_remote_code" in kwargs:
            return original_load(*args, **kwargs)
        try:
            return original_load(*args, trust_remote_code=True, **kwargs)
        except Exception as exc:
            if _is_trc_rejection(exc):
                return original_load(*args, **kwargs)
            raise

    def patched_builder(*args, **kwargs):
        args, kwargs = _with_canonical_path(args, kwargs)
        if "trust_remote_code" in kwargs:
            return original_builder(*args, **kwargs)
        try:
            return original_builder(*args, trust_remote_code=True, **kwargs)
        except Exception as exc:
            if _is_trc_rejection(exc):
                return original_builder(*args, **kwargs)
            raise

    datasets.load_dataset = patched_load_dataset  # type: ignore[assignment]
    datasets.load_dataset_builder = patched_builder  # type: ignore[assignment]
    for mod in list(sys.modules.values()):
        if mod is None or mod is datasets:
            continue
        if getattr(mod, "load_dataset", None) is original_load:
            mod.load_dataset = patched_load_dataset  # type: ignore[attr-defined]
        if getattr(mod, "load_dataset_builder", None) is original_builder:
            mod.load_dataset_builder = patched_builder  # type: ignore[attr-defined]


def flatten_task_dict(d: dict) -> Iterator[tuple[str, Any]]:
    """Yield ``(task_name, task)`` leaf pairs of a (possibly nested) ``get_task_dict`` result."""
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flatten_task_dict(v)
            continue
        name = k if isinstance(k, str) else getattr(k, "group", str(k))
        yield (name, v)


def materialize_first_nonempty_split(task) -> tuple[str, list] | None:
    """Return ``(split_name, docs)`` for the first non-empty of test, validation, training; else None."""
    for split_name, getter in (
        ("test", task.test_docs),
        ("validation", task.validation_docs),
        ("training", task.training_docs),
    ):
        try:
            docs = list(getter())
        except Exception:
            docs = []
        if docs:
            return (split_name, docs)
    return None
