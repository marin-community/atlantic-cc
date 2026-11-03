# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib
import subprocess
import sys

from atlantic_cc import cli

# Prints the stage CLI modules and heavy stacks that running one command imported.
PROBE = """
import sys
from atlantic_cc import cli
try:
    cli.main(["worklist", "--help"])
except SystemExit:
    pass
print(sorted(m for m in sys.modules if m.startswith("atlantic_cc.") and m.endswith(".cli") and m != cli.__name__))
print(sorted(m for m in ("transformers", "torch", "jax", "fasttext") if m in sys.modules))
"""


def test_each_stage_module_registers_exactly_its_listed_commands():
    for module in dict.fromkeys(cli.COMMAND_MODULES.values()):
        subparsers = argparse.ArgumentParser().add_subparsers()
        importlib.import_module(module).register(subparsers)
        assert set(subparsers.choices) == {c for c, m in cli.COMMAND_MODULES.items() if m == module}, module


def test_running_a_command_imports_only_its_stage():
    out = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, check=True).stdout
    stage_modules, heavy = out.splitlines()[-2:]
    assert stage_modules == "['atlantic_cc.worklist.cli']"
    assert heavy == "[]"
