# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Install a prebuilt resiliparse-rs artifact so ``resiliparse._extract_rs`` imports from it.

An artifact is a directory (local path or any fsspec URL) laid out like production's
``artifacts/resiliparse_rs/latest/``:

    manifest.json           {"commit_sha": ..., "python_version": "3.12.x", ...}
    resiliparse_py.tar.gz   the fork's ``resiliparse-py/`` tree
    _extract_rs.so          the pyo3 extension (not abi3: needs the manifest's CPython minor)
    liblexbor.so.2          lexbor, found next to the extension through an ``$ORIGIN`` rpath

:func:`install_artifact` checks the manifest pins the extractor commit and the running CPython
minor, unpacks the tree, drops both libraries into ``resiliparse-py/resiliparse/`` and returns the
directory to put first on ``sys.path``. It also reports whether the libraries are byte-identical to
the build production used. The destination must allow ``mmap(PROT_EXEC)``: ``/tmp`` is mounted
``noexec`` on some clusters (Iris CPU workers among them).

A wheel (``resiliparse-rs-atlantic-cc``, see ``docs/extractor.md``) needs none of this: pass
``pkg_dir=None`` to the pool. :func:`is_asrun_build` tells, for either layout, whether the extension
that would load is production's build.
"""

import hashlib
import importlib.util
import json
import logging
import os
import sys
import tarfile
from dataclasses import dataclass

import fsspec

from atlantic_cc import spec

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
TARBALL_NAME = "resiliparse_py.tar.gz"
EXTENSION_NAME = "_extract_rs.so"
LEXBOR_NAME = "liblexbor.so.2"
# lexbor first: the extension resolves it when it is loaded.
ARTIFACT_LIBS = (LEXBOR_NAME, EXTENSION_NAME)
PACKAGE_DIR_NAME = "resiliparse-py"


@dataclass(frozen=True)
class InstalledExtractor:
    pkg_dir: str  # put first on sys.path in every process that imports the extension
    commit: str
    python_version: str
    asrun_binaries: bool  # both libraries match the md5s of the build that produced Atlantic-CC


def check_manifest(manifest: dict) -> None:
    """Raise unless the artifact was built from the pinned commit for the running CPython minor."""
    if manifest["commit_sha"] != spec.EXTRACTOR_COMMIT:
        raise RuntimeError(
            f"resiliparse-rs artifact was built from {manifest['commit_sha']}, Atlantic-CC pins {spec.EXTRACTOR_COMMIT}"
        )
    want = ".".join(manifest["python_version"].split(".")[:2])
    have = f"{sys.version_info.major}.{sys.version_info.minor}"
    if want != have:
        raise RuntimeError(f"resiliparse-rs artifact is a pyo3 build for CPython {want}, but this is CPython {have}")


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def extension_files(pkg_dir: str | None) -> tuple[str, str]:
    """``(extension, lexbor)``: the files ``resiliparse._extract_rs`` loads, from ``pkg_dir`` or the environment.

    Locates the module without importing it; a wheel names the extension
    ``_extract_rs.cpython-312-x86_64-linux-gnu.so``.
    """
    if pkg_dir is not None:
        root = os.path.join(pkg_dir, "resiliparse")
        return os.path.join(root, EXTENSION_NAME), os.path.join(root, LEXBOR_NAME)
    found = importlib.util.find_spec("resiliparse._extract_rs")
    if found is None or found.origin is None:
        raise ModuleNotFoundError("resiliparse._extract_rs is not installed; see docs/extractor.md")
    return found.origin, os.path.join(os.path.dirname(found.origin), LEXBOR_NAME)


def is_asrun_build(pkg_dir: str | None) -> bool:
    """Whether that extension and its lexbor are byte-identical to the build that produced Atlantic-CC."""
    extension, lexbor = extension_files(pkg_dir)
    want = spec.ASRUN_EXTRACTOR_ARTIFACT
    return (
        os.path.exists(lexbor)  # macOS builds ship liblexbor.<version>.dylib instead
        and file_sha256(extension) == want[EXTENSION_NAME]["sha256"]
        and file_sha256(lexbor) == want[LEXBOR_NAME]["sha256"]
    )


def install_artifact(prefix: str, dest: str) -> InstalledExtractor:
    """Verify and unpack the artifact at ``prefix`` under ``dest``; see the module docstring."""
    prefix = prefix.rstrip("/")
    with fsspec.open(f"{prefix}/{MANIFEST_NAME}", "r") as f:
        manifest = json.load(f)
    check_manifest(manifest)

    os.makedirs(dest, exist_ok=True)
    tarball = os.path.join(dest, TARBALL_NAME)
    with fsspec.open(f"{prefix}/{TARBALL_NAME}", "rb") as src, open(tarball, "wb") as dst:
        dst.write(src.read())
    with tarfile.open(tarball) as tf:
        tf.extractall(dest, filter="data")
    pkg_dir = os.path.abspath(os.path.join(dest, PACKAGE_DIR_NAME))
    if not os.path.isdir(pkg_dir):
        raise RuntimeError(f"expected {PACKAGE_DIR_NAME}/ in {prefix}/{TARBALL_NAME}")

    for lib in ARTIFACT_LIBS:
        with fsspec.open(f"{prefix}/{lib}", "rb") as src, open(os.path.join(pkg_dir, "resiliparse", lib), "wb") as dst:
            dst.write(src.read())
    asrun = is_asrun_build(pkg_dir)
    with open(os.path.join(dest, MANIFEST_NAME), "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "installed resiliparse-rs %s from %s -> %s (as-run binaries: %s)",
        manifest["commit_sha"][:7],
        prefix,
        pkg_dir,
        asrun,
    )
    return InstalledExtractor(
        pkg_dir=pkg_dir, commit=manifest["commit_sha"], python_version=manifest["python_version"], asrun_binaries=asrun
    )


def activate(pkg_dir: str | None) -> None:
    """Put an installed artifact's package directory first on ``sys.path`` (no-op for a regular install)."""
    if pkg_dir is None or (sys.path and sys.path[0] == pkg_dir):
        return
    sys.path.insert(0, pkg_dir)
