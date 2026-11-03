# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from atlantic_cc.release import regroup
from atlantic_cc.release.params import VERSION
from tests.unit.release import synthetic


@pytest.fixture(scope="session")
def built(tmp_path_factory) -> synthetic.Fixture:
    """The synthetic pool run through warc-paths, pieces, parts and manifest (read-only for tests)."""
    fx = synthetic.build(tmp_path_factory.mktemp("regroup"))
    regroup.write_warc_paths(fx.worklist, fx.stage)
    regroup.pieces(fx.source, fx.decon_run, fx.stage, [], workers=1)
    regroup.parts(fx.out, VERSION, fx.stage, part_mb=1000, zstd_level=9, workers=1)
    regroup.manifest(fx.out, VERSION, fx.stage, fx.decon_run, gzip_mtime=0)
    return fx
