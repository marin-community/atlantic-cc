# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``atlantic-cc`` command line entry point.

Each stage package defines its commands with ``register(subparsers)`` in its ``cli`` module. Only the
module that defines the requested command is imported: worker processes started with ``spawn``
re-import this entry module, and some stages load the tokenizer and model stacks at import time.
"""

import argparse
import importlib
import logging
import sys

DESCRIPTION = "Reproduce Atlantic-CC v1.0 from Common Crawl."

# Top-level command -> the stage module whose ``register`` defines it, in pipeline order.
COMMAND_MODULES = {
    "worklist": "atlantic_cc.worklist.cli",
    "extractor-install": "atlantic_cc.extract.cli",
    "extractor-check": "atlantic_cc.extract.cli",
    "extract-classify": "atlantic_cc.pipeline.cli",
    "dedup": "atlantic_cc.dedup.cli",
    "decon": "atlantic_cc.decon.cli",
    "release": "atlantic_cc.release.cli",
}


def build_parser(modules: list[str]) -> argparse.ArgumentParser:
    """Build a parser holding the commands that the given stage modules register."""
    parser = argparse.ArgumentParser(prog="atlantic-cc", description=DESCRIPTION)
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for module in modules:
        importlib.import_module(module).register(subparsers)
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    command = next((arg for arg in argv if not arg.startswith("-")), None)
    if command in COMMAND_MODULES:
        modules = [COMMAND_MODULES[command]]
    else:
        # No or unknown command: load every stage so that --help and the usage error list all commands.
        modules = list(dict.fromkeys(COMMAND_MODULES.values()))
    args = build_parser(modules).parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
