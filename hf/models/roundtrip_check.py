# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load the Atlantic-CC model repos the way a Hub user would and run their cards' code (test M4).

Runs inside a fresh virtual environment that has only torch, transformers, safetensors,
huggingface_hub and fastText (no ``atlantic_cc``, no JAX), either offline (``HF_HUB_OFFLINE=1`` with
``HF_HOME`` pointing at a cache that ``verify_staged.py`` pre-populated from the staged folders) or
against the Hub after an upload. Prints one JSON document: per repo, the load warnings, the label
configuration, a ``pipeline`` prediction and the stdout of every Python block of its README (blocks
that use the ``atlantic_cc`` package are skipped).

Usage: ``python -I roundtrip_check.py FASTTEXT_REPO POOLED_REPO TERMINAL_REPO``
"""

import contextlib
import io
import json
import logging
import re
import sys

import fasttext
import transformers
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, pipeline

ROLES = ("fasttext", "pooled", "terminal")
SAMPLE = (
    "How to repot a fiddle-leaf fig. Choose a pot 5 cm wider than the root ball and water the plant the day before."
)
WARNING_PATTERNS = ("newly initialized", "not used", "MISSING", "UNEXPECTED", "MISMATCH", "should probably TRAIN")


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _load_warnings(load) -> tuple[object, list[str]]:
    """Run ``load()`` and return its result and every warning-like message it logged or printed."""
    handler = _Collect()
    logger = logging.getLogger("transformers")
    logger.addHandler(handler)
    err, out = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
            result = load()
    finally:
        logger.removeHandler(handler)
    text = "\n".join(handler.messages) + err.getvalue() + out.getvalue()
    return result, [line for line in text.splitlines() if any(p in line for p in WARNING_PATTERNS)]


def readme_blocks(repo: str) -> list[str]:
    with open(hf_hub_download(repo, "README.md")) as f:
        readme = f.read()
    blocks = re.findall(r"```python\n(.*?)```", readme, flags=re.S)
    return [b for b in blocks if "atlantic_cc" not in b]


def run_block(code: str) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, "<README>", "exec"), {"__name__": "__readme__"})  # noqa: S102 (the card's own code)
    return out.getvalue()


def check_neural(name: str, repo: str) -> dict:
    remote = {"trust_remote_code": True} if name == "pooled" else {}
    model, warnings = _load_warnings(lambda: AutoModelForSequenceClassification.from_pretrained(repo, **remote))
    clf = pipeline("text-classification", model=repo, **remote)
    return {
        "load_warnings": warnings,
        "class": type(model).__name__,
        "architectures": model.config.architectures,
        "id2label": {str(k): v for k, v in model.config.id2label.items()},
        "pipeline": clf(SAMPLE),
        "readme_outputs": [run_block(block) for block in readme_blocks(repo)],
    }


def check_fasttext(repo: str) -> dict:
    model = fasttext.load_model(hf_hub_download(repo, "model.bin"))
    return {"labels": sorted(model.labels), "readme_outputs": [run_block(block) for block in readme_blocks(repo)]}


def main() -> None:
    repos = dict(zip(ROLES, sys.argv[1:], strict=True))
    report = {"transformers": transformers.__version__, "python": sys.version.split()[0], "repos": repos}
    report["fasttext"] = check_fasttext(repos["fasttext"])
    for name in ("pooled", "terminal"):
        report[name] = check_neural(name, repos[name])
    print(json.dumps(report))


if __name__ == "__main__":
    main()
