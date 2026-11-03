# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare top-level definitions (docstrings stripped) of the as-run Marin decon code with the port.

Needs a Marin checkout with tag ``lpv11-8m-as-run`` (read only)::

    uv run python tests/parity/decon/asrun_ast.py /path/to/marin > tests/parity/decon/reports/asrun_ast.json
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

MARIN = sys.argv[1]
ACC = str(Path(__file__).resolve().parents[3] / "src" / "atlantic_cc" / "decon")


def tag_source(path):
    return subprocess.run(
        ["git", "-C", MARIN, "show", f"lpv11-8m-as-run:{path}"], capture_output=True, text=True, check=True
    ).stdout


def defs(src):
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign)):
            if isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if not names:
                    continue
                name = names[0]
            else:
                name = node.name
            for n in ast.walk(node):
                if (
                    isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
                    and n.body
                    and isinstance(n.body[0], ast.Expr)
                    and isinstance(getattr(n.body[0], "value", None), ast.Constant)
                    and isinstance(n.body[0].value.value, str)
                ):
                    n.body = n.body[1:] or [ast.Pass()]
            # strip docstring-only attribute strings in class bodies
            if isinstance(node, ast.ClassDef):
                node.body = [
                    b
                    for b in node.body
                    if not (isinstance(b, ast.Expr) and isinstance(getattr(b, "value", None), ast.Constant))
                ] or [ast.Pass()]
            out[name] = ast.dump(node, include_attributes=False)
    return out


pairs = [
    ("experiments/fast_curation/decon_v4.py", f"{ACC}/matching.py", None),
    ("experiments/fast_curation/decon_8m.py", f"{ACC}/names.py", ["_EVAL_ID_RE", "eval_name"]),
    ("experiments/fast_curation/decon_8m.py", f"{ACC}/dropsets.py", ["drop_set_source"]),
    ("experiments/datakit/decontam/prepare_eval_corpus.py", f"{ACC}/eval_corpus.py", None),
    ("experiments/datakit/decontam/lmh_loader.py", f"{ACC}/lmh_loader.py", None),
    (
        "experiments/fast_curation/decon_eval_corpus_fill.py",
        f"{ACC}/eval_corpus.py",
        [
            "lmh_load_name",
            "mmmlu_config",
            "mmmlu_text",
            "squad_v2_text",
            "_INCLUDE_RE",
            "_MMMLU_RE",
            "MMMLU_REPO",
            "SQUAD_V2_PARQUET",
        ],
    ),
]
report = {}
for asrun, port, only in pairs:
    a, p = defs(tag_source(asrun)), defs(open(port).read())
    names = only or sorted(set(a) & set(p))
    same = [n for n in names if n in a and n in p and a[n] == p[n]]
    diff = [n for n in names if n in a and n in p and a[n] != p[n]]
    missing = [n for n in names if n not in p]
    report[f"{asrun} -> {port.split('/')[-1]}"] = {
        "identical": same,
        "different": diff,
        "missing_in_port": missing,
        "asrun_only": sorted(set(a) - set(p)) if only is None else [],
    }
print(json.dumps(report, indent=1))
