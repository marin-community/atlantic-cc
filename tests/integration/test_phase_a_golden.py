# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""I-WARC: download each G1 fixture WARC from data.commoncrawl.org, run phase A, compare with the golden.

Exact on Linux x86_64, CPython 3.12, the pinned wheels and the resiliparse-rs build at 850891b:
the survivor set equals the as-run kept ∪ tombstones, kept text is byte-identical (64-bit sha256
prefix), ``fasttext_score`` and ``n_tokens`` are exact, and the phase-A counters equal the as-run
marker. Downloads 0.2-1.3 GB per fixture; select one with ``-k F4``.

Inputs:

* the extractor: an installed ``resiliparse._extract_rs``, or the package directory of an installed
  artifact in ``ATLANTIC_CC_EXTRACTOR_DIR`` (``atlantic-cc extractor-install`` prints it);
* the fastText model: ``ATLANTIC_CC_FASTTEXT_MODEL`` (local path or fsspec URL), else the released
  model on the Hugging Face Hub.
"""

import importlib.util
import json
import os
import sys

import pytest

from atlantic_cc import spec
from atlantic_cc.classify.fasttext_gate import FastTextGate
from atlantic_cc.classify.text import normalize_text
from atlantic_cc.classify.tokenize import load_tokenizer, tokenize_batch
from atlantic_cc.extract.install import activate
from atlantic_cc.extract.pool import open_extractor_pool
from atlantic_cc.pipeline.phase_a import run_phase_a
from tests.parity.r1_compare import compare, golden_dirs, load_golden

pytestmark = [pytest.mark.network, pytest.mark.extractor, pytest.mark.slow]

GOLDENS = {load_golden(d)[0]["fixture"]: d for d in golden_dirs()}


@pytest.fixture(scope="module")
def extractor_pool():
    pkg_dir = os.environ.get("ATLANTIC_CC_EXTRACTOR_DIR")
    saved = list(sys.path)
    activate(pkg_dir)
    try:
        found = importlib.util.find_spec("resiliparse._extract_rs") is not None
    except ModuleNotFoundError:
        found = False
    finally:
        sys.path[:] = saved
    if not found:
        pytest.skip("resiliparse-rs is not installed (see docs/extractor.md)")
    with open_extractor_pool(os.cpu_count() or 1, pkg_dir) as pool:
        yield pool


@pytest.fixture(scope="module")
def gate() -> FastTextGate:
    path = os.environ.get("ATLANTIC_CC_FASTTEXT_MODEL")
    if path is None:
        hub = pytest.importorskip("huggingface_hub")
        try:
            path = hub.hf_hub_download(spec.FASTTEXT_HF_REPO, "model.bin")
        except Exception as e:  # not published yet, or offline
            pytest.skip(f"fastText model unavailable ({type(e).__name__}); set ATLANTIC_CC_FASTTEXT_MODEL")
    return FastTextGate.load(path)


@pytest.mark.parametrize("fixture", sorted(GOLDENS))
def test_phase_a_reproduces_the_release(fixture, extractor_pool, gate):
    meta, ref = load_golden(GOLDENS[fixture])
    table, counters = run_phase_a(meta["warc_path"], gate, extractor_pool)
    n_tokens = [
        len(ids)
        for ids in tokenize_batch(load_tokenizer(), [normalize_text(t) for t in table.column("text").to_pylist()])
    ]
    report = compare(table, counters, ref, n_tokens)
    assert report["exact"], json.dumps(report, indent=2)
