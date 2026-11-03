# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``atlantic-cc extractor-install`` and ``atlantic-cc extractor-check``."""

import argparse
import json
import logging
from dataclasses import asdict

from atlantic_cc.extract.install import install_artifact, is_asrun_build
from atlantic_cc.extract.pool import open_extractor_pool

logger = logging.getLogger(__name__)


def _install(args: argparse.Namespace) -> None:
    installed = install_artifact(args.artifact, args.dest)
    open_extractor_pool(1, installed.pkg_dir).close()
    print(json.dumps(asdict(installed), indent=2))


def _check(args: argparse.Namespace) -> None:
    open_extractor_pool(1, args.pkg_dir).close()
    print(json.dumps({"pkg_dir": args.pkg_dir, "smoke": "passed", "asrun_binaries": is_asrun_build(args.pkg_dir)}))


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "extractor-install",
        help="Install and smoke-test a prebuilt resiliparse-rs artifact (manifest, tarball, .so files).",
    )
    p.add_argument("--artifact", required=True, help="Artifact directory: local path or fsspec URL.")
    p.add_argument(
        "--dest", required=True, help="Unpack directory; must allow executable mappings (not a noexec /tmp)."
    )
    p.set_defaults(func=_install)

    p = subparsers.add_parser(
        "extractor-check",
        help="Smoke-test resiliparse-rs in a spawned child process and report whether it is production's build.",
    )
    p.add_argument(
        "--pkg-dir", default=None, help="An installed artifact's package directory; omit for a regular install."
    )
    p.set_defaults(func=_check)
