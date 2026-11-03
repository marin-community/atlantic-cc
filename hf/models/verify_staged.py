# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Round-trip the Atlantic-CC model repos in fresh environments (test M4 in ``docs/reproducibility.md``).

For each environment (the pinned transformers 5.12.1 stack and the oldest transformers with
ModernBERT, 4.48), this script creates a virtual environment with only torch, transformers,
safetensors, huggingface_hub and fastText and runs ``roundtrip_check.py`` there: load every model
the way a Hub user would (zero initialization warnings), check the labels, run a ``pipeline`` and
the card snippets verbatim. It compares the snippets' printed scores with this repository's own
scorers on the staged folders and writes a JSON report.

By default the repos are served offline (``HF_HUB_OFFLINE=1``) from a Hugging Face cache built from
the staged folders under their Hub repo ids. With ``--hub`` they are downloaded from the Hub instead
(the post-upload check; ``HF_TOKEN`` is passed through for private repos). Creating virtual
environments is heavy: run this in an Iris job, not on a laptop.

Usage::

    python hf/models/verify_staged.py --staging .cache/models/hf_staging --report OUT.json [--envs 5.12.1 4.48.3]
    python hf/models/verify_staged.py --hub --report OUT.json      # after uploading
"""

import argparse
import ast
import json
import os
import re
import subprocess
from pathlib import Path

import fasttext
import fsspec
import numpy as np
import torch

from atlantic_cc import spec
from atlantic_cc.classify.batching import pad_batch
from atlantic_cc.classify.text import normalize_text
from atlantic_cc.classify.tokenize import load_tokenizer, tokenize_batch
from atlantic_cc.models import Backend, Precision, load_pooled, load_terminal

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER = Path(__file__).resolve().parent / "roundtrip_check.py"
REPOS = {"fasttext": spec.FASTTEXT_HF_REPO, "pooled": spec.POOLED_HF_REPO, "terminal": spec.TERMINAL_HF_REPO}
FAKE_COMMIT = "0" * 40
ENVIRONMENTS = {
    "5.12.1": [
        "torch",
        "transformers==5.12.1",
        "tokenizers==0.22.2",
        "huggingface-hub==1.21.0",
        "safetensors==0.6.2",
        "numpy==2.3.5",
        "fasttext-wheel==0.9.2",
    ],
    "4.48.3": [
        "torch",
        "transformers==4.48.3",
        "huggingface-hub<1.0",
        "safetensors",
        "numpy<2",
        "fasttext-wheel==0.9.2",
    ],
}
# fastText is deterministic; the pooled card loads float32 weights with the same numerics as this
# repository's scorer; the ModernBERT card loads bf16 weights, whose kernels differ across versions.
SNIPPET_ATOL = {"fasttext": 0.0, "pooled": 1e-5, "terminal": 2e-2}


def staged_dir(staging: Path, role: str) -> Path:
    return staging / REPOS[role].split("/")[1]


def build_offline_cache(staging: Path, hf_home: Path) -> None:
    """``hub/models--<org>--<repo>/snapshots/<commit>/`` linking to the staged files, plus ``refs/main``."""
    for role, repo in REPOS.items():
        root = hf_home / "hub" / f"models--{repo.replace('/', '--')}"
        (root / "refs").mkdir(parents=True, exist_ok=True)
        (root / "refs" / "main").write_text(FAKE_COMMIT)
        staged = staged_dir(staging, role)
        for path in staged.rglob("*"):
            if path.is_file():
                link = root / "snapshots" / FAKE_COMMIT / path.relative_to(staged)
                link.parent.mkdir(parents=True, exist_ok=True)
                if not link.exists():
                    link.symlink_to(path.resolve())


def make_env(venv: Path, packages: list[str]) -> Path:
    if not (venv / "bin" / "python").exists():
        subprocess.run(["uv", "venv", "--python", "3.12", str(venv)], check=True)
    python = venv / "bin" / "python"
    subprocess.run(["uv", "pip", "install", "--python", str(python), *packages], check=True)
    return python


def run_checker(python: Path, hf_home: Path, offline: bool) -> dict:
    """``roundtrip_check.py``'s report, or ``{"error": <its stderr>}`` if it failed."""
    env = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ["HOME"],
        "HF_HOME": str(hf_home),
        "TOKENIZERS_PARALLELISM": "false",
    }
    if offline:
        env |= {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    elif "HF_TOKEN" in os.environ:
        env["HF_TOKEN"] = os.environ["HF_TOKEN"]
    cmd = [str(python), "-I", str(CHECKER), REPOS["fasttext"], REPOS["pooled"], REPOS["terminal"]]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if out.returncode:
        return {"error": out.stderr[-4000:]}
    return json.loads(out.stdout.strip().splitlines()[-1])


def card_pages(repo_dir: Path) -> list[str]:
    """The ``pages`` list of the card's usage snippet."""
    block = re.search(r"```python\n(.*?)```", (repo_dir / "README.md").read_text(), flags=re.S).group(1)
    return ast.literal_eval(re.search(r"pages = (\[.*?\])\n", block, flags=re.S).group(1))


def expected_scores(staging: Path) -> dict[str, list[float]]:
    """This repository's scorers on the card pages, with the card's tokenization (padding to the longest)."""
    tok = load_tokenizer()
    ft_dir = staged_dir(staging, "fasttext")
    ft = fasttext.load_model(str(ft_dir / "model.bin"))
    expected = {
        "fasttext": [
            dict((label, prob) for prob, label in ft.f.predict(normalize_text(p), 2, 0.0, "strict"))["__label__useful"]
            for p in card_pages(ft_dir)
        ]
    }
    for role, max_len, loader in (
        ("pooled", spec.POOLED_MAX_TOKENS, load_pooled),
        ("terminal", spec.TERMINAL_MAX_TOKENS, load_terminal),
    ):
        folder = staged_dir(staging, role)
        ids = tokenize_batch(tok, [normalize_text(p) for p in card_pages(folder)], max_length=max_len)
        input_ids, mask = pad_batch(ids, max(len(x) for x in ids))
        expected[role] = loader(str(folder), Backend.TORCH, precision=Precision.BF16).score(input_ids, mask).tolist()
    return expected


def compare(report: dict, expected: dict[str, list[float]]) -> list[str]:
    if "error" in report:
        return [f"roundtrip_check.py failed: {report['error'].strip().splitlines()[-1]}"]
    problems = []
    for role, atol in SNIPPET_ATOL.items():
        outputs = report[role]["readme_outputs"]
        printed = [
            float(x) for line in outputs[0].splitlines() for x in re.findall(r"[-+]?\d*\.\d+(?:e[-+]?\d+)?", line)
        ]
        if len(printed) != len(expected[role]) or not np.allclose(printed, expected[role], rtol=0, atol=atol):
            problems.append(f"{role}: card printed {printed}, expected {expected[role]} (atol {atol})")
        if role != "fasttext":
            if report[role]["load_warnings"]:
                problems.append(f"{role}: load warnings {report[role]['load_warnings']}")
            if report[role]["id2label"] != {"0": "not_useful", "1": "useful"}:
                problems.append(f"{role}: id2label {report[role]['id2label']}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--staging", type=Path, default=REPO_ROOT / ".cache" / "models" / "hf_staging")
    parser.add_argument("--work", type=Path, default=REPO_ROOT / ".cache" / "models" / "m4")
    parser.add_argument("--report", required=True, help="JSON report path (local or gs://)")
    parser.add_argument("--envs", nargs="+", default=list(ENVIRONMENTS), choices=list(ENVIRONMENTS))
    parser.add_argument("--hub", action="store_true", help="download the repos from the Hub (after uploading)")
    args = parser.parse_args()
    torch.set_num_threads(4)
    hf_home = args.work / ("hf_home_hub" if args.hub else "hf_home")
    if not args.hub:
        build_offline_cache(args.staging, hf_home)
    expected = expected_scores(args.staging)
    results = {"repos": REPOS, "source": "hub" if args.hub else "staged", "expected": expected, "environments": {}}
    for version in args.envs:
        python = make_env(args.work / f"venv-transformers-{version}", ENVIRONMENTS[version])
        report = run_checker(python, hf_home, offline=not args.hub)
        results["environments"][version] = {"report": report, "problems": compare(report, expected)}
    with fsspec.open(args.report, "w") as f:
        f.write(json.dumps(results, indent=2) + "\n")
    for version, result in results["environments"].items():
        print(version, "OK" if not result["problems"] else result["problems"], flush=True)
    if any(r["problems"] for r in results["environments"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
