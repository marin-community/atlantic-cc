# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Installing a prebuilt resiliparse-rs artifact: manifest pins, layout, as-run binary detection."""

import io
import json
import os
import sys
import tarfile
import types
from pathlib import Path

import pytest

from atlantic_cc import spec
from atlantic_cc.extract.install import extension_files, install_artifact, is_asrun_build

RUNNING_PYTHON = f"{sys.version_info.major}.{sys.version_info.minor}.7"


def _artifact(
    root: Path, *, commit: str = spec.EXTRACTOR_COMMIT, python: str = RUNNING_PYTHON, tree: str = "resiliparse-py"
):
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"commit_sha": commit, "python_version": python}))
    with tarfile.open(root / "resiliparse_py.tar.gz", "w:gz") as tf:
        data = b"# fork package\n"
        info = tarfile.TarInfo(f"{tree}/resiliparse/__init__.py")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    (root / "_extract_rs.so").write_bytes(b"not the as-run extension")
    (root / "liblexbor.so.2").write_bytes(b"not the as-run lexbor")
    return str(root)


def test_install_lays_out_the_package(tmp_path):
    installed = install_artifact(_artifact(tmp_path / "artifact"), str(tmp_path / "dest"))
    pkg = Path(installed.pkg_dir)
    assert pkg == (tmp_path / "dest" / "resiliparse-py").resolve() and pkg.is_absolute()
    assert sorted(os.listdir(pkg / "resiliparse")) == ["__init__.py", "_extract_rs.so", "liblexbor.so.2"]
    assert (installed.commit, installed.python_version) == (spec.EXTRACTOR_COMMIT, RUNNING_PYTHON)
    assert not installed.asrun_binaries
    assert json.loads((tmp_path / "dest" / "manifest.json").read_text())["commit_sha"] == spec.EXTRACTOR_COMMIT


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"commit": "7f1a09a" + "0" * 33}, "Atlantic-CC pins"),
        ({"python": "3.11.9"}, "pyo3 build for CPython 3.11"),
        ({"tree": "something-else"}, "expected resiliparse-py/"),
    ],
)
def test_install_refuses_a_mismatched_artifact(tmp_path, kwargs, message):
    with pytest.raises(RuntimeError, match=message):
        install_artifact(_artifact(tmp_path / "artifact", **kwargs), str(tmp_path / "dest"))


def test_a_wheel_install_is_located_without_importing_the_extension(tmp_path, monkeypatch):
    pkg = tmp_path / "resiliparse"
    pkg.mkdir()
    extension = pkg / "_extract_rs.py"  # a wheel names it _extract_rs.cpython-312-x86_64-linux-gnu.so
    extension.write_text("raise AssertionError('the extension must not be imported')\n")
    (pkg / "liblexbor.so.2").write_bytes(b"not the as-run lexbor")
    namespace = types.ModuleType("resiliparse")  # the wheel's namespace package
    namespace.__path__ = [str(pkg)]
    monkeypatch.setitem(sys.modules, "resiliparse", namespace)

    assert extension_files(None) == (str(extension), str(pkg / "liblexbor.so.2"))
    assert not is_asrun_build(None)
