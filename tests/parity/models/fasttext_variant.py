# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score texts with whichever fastText package is installed (run in an isolated environment, M3).

Usage: ``python -I fasttext_variant.py MODEL_BIN TEXTS_JSON OUT_JSON``. Writes the package name and
version and, per text, P(``__label__useful``) from the low-level predict without a trailing newline
(the production call) and with one (what the high-level ``predict`` scores).
"""

import importlib.metadata
import json
import sys

import fasttext


def useful(model, text: str) -> float:
    for a, b in model.f.predict(text, 2, 0.0, "strict"):
        prob, label = (a, b) if isinstance(b, str) else (b, a)
        if label == "__label__useful":
            return float(prob)
    return 0.0


def main() -> None:
    model_path, texts_path, out_path = sys.argv[1:4]
    model = fasttext.load_model(model_path)
    with open(texts_path) as f:
        texts = json.load(f)
    packages = {
        dist.metadata["Name"]: dist.version
        for dist in importlib.metadata.distributions()
        if dist.metadata["Name"] and "fasttext" in dist.metadata["Name"].lower()
    }
    result = {
        "packages": packages,
        "numpy": importlib.metadata.version("numpy"),
        "low_level": [useful(model, t) for t in texts],
        "with_newline": [useful(model, t + "\n") for t in texts],
    }
    with open(out_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
