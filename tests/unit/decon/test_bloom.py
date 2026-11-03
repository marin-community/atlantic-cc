# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""U-DECON item 7: the pure-Python bloom reader agrees with dupekit."""

import random

import dupekit_native
import pytest

from atlantic_cc.decon.bloom import PyBloom, bloom_parameters, load_native_bloom


@pytest.mark.parametrize(
    ("capacity", "bits"),
    [(1_000, 43_136), (50_000_000, 2_156_638_136), (167_087_048, 7_206_925_992)],
    ids=["1k", "hero-50M", "full-val"],
)
def test_sizing_matches_dupekit(capacity, bits):
    k, size_in_bits = bloom_parameters(capacity, 1e-9)
    assert (k, size_in_bits) == (29, bits)
    if capacity <= 50_000_000:
        assert dupekit_native.Bloom(capacity, 1e-9).size_in_bits == bits


def test_reader_matches_dupekit_on_members_and_random_probes(tmp_path):
    rng = random.Random(0)
    native = dupekit_native.Bloom(20_000, 1e-9)
    items = [rng.getrandbits(64) for _ in range(20_000)]
    for h in items:
        native.add(h)
    path = tmp_path / "filter.bin"
    path.write_bytes(native.save_bytes())

    reader = PyBloom.open(str(path))
    assert reader.k == 29
    assert all(h in reader for h in items)
    probes = [rng.getrandbits(64) for _ in range(200_000)]
    assert [h in reader for h in probes] == [h in native for h in probes]
    loaded = load_native_bloom(str(path))
    assert all(h in loaded for h in items)


def test_negative_ints_probe_different_bits():
    """Hashes must stay uint64: an int64 view of the same bits is a different item."""
    native = dupekit_native.Bloom(1_000, 1e-9)
    native.add(2**63 + 5)
    assert 2**63 + 5 in native
    assert (2**63 + 5) - 2**64 not in native
