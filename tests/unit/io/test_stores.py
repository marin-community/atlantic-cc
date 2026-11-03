# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import threading

import pytest

from atlantic_cc.io.fsspec_store import FsspecStore
from atlantic_cc.io.local import LocalStore
from atlantic_cc.io.resolve import store_for
from atlantic_cc.io.store import PreconditionFailed, delete_prefix


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def test_create_has_exactly_one_winner_among_racing_threads(tmp_path):
    store = LocalStore()
    path = str(tmp_path / "claims" / "000001.json")
    winners = []

    def attempt(i: int) -> None:
        try:
            store.create(path, f"writer {i}".encode())
            winners.append(i)
        except PreconditionFailed:
            pass

    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(winners) == 1
    assert store.read(path) == f"writer {winners[0]}".encode()
    assert [os.path.basename(i.path) for i in store.list(str(tmp_path) + "/")] == ["000001.json"]


def test_replace_is_a_compare_and_swap_on_the_generation(tmp_path):
    store = LocalStore()
    path = str(tmp_path / "claim.json")
    first = store.create(path, b"a")
    second = store.replace(path, b"b", first)
    assert second > first
    with pytest.raises(PreconditionFailed):
        store.replace(path, b"c", first)
    assert store.read(path) == b"b"
    with pytest.raises(PreconditionFailed):
        store.replace(str(tmp_path / "missing.json"), b"x", second)


def test_every_write_moves_the_generation_even_within_one_clock_tick(tmp_path):
    clock = Clock()
    store = LocalStore(clock=clock)
    path = str(tmp_path / "object.bin")
    generations = [store.write(path, b"same bytes") for _ in range(5)]
    assert generations == sorted(set(generations))
    assert store.stat(path).generation == generations[-1]


def test_pinned_ranged_read_fails_after_a_rewrite(tmp_path):
    store = LocalStore()
    path = str(tmp_path / "blocks" / "000000.bin")
    generation = store.write(path, [b"0123", b"4567", b"89"])
    assert store.read(path, 2, 7, generation=generation) == b"23456"
    assert store.read(path, 8) == b"89"
    assert store.read(path, 5, 5) == b""
    store.write(path, b"0123456789")
    with pytest.raises(PreconditionFailed):
        store.read(path, 2, 7, generation=generation)


def test_listing_matches_path_prefixes_and_skips_unfinished_temporaries(tmp_path):
    store = LocalStore()
    for name in ("done/000000.json", "done/000001.json", "done-extra/x.json", "doneish.json"):
        store.write(str(tmp_path / name), b"{}")
    (tmp_path / "done" / ".000002.json.__tmp__.1.abc").write_bytes(b"partial")
    directory = [os.path.relpath(i.path, tmp_path) for i in store.list(f"{tmp_path}/done/")]
    assert directory == ["done/000000.json", "done/000001.json"]
    prefix = [os.path.relpath(i.path, tmp_path) for i in store.list(f"{tmp_path}/done")]
    assert prefix == ["done-extra/x.json", "done/000000.json", "done/000001.json", "doneish.json"]
    assert store.list(f"{tmp_path}/absent/") == []


def test_delete_prefix_removes_everything_under_it(tmp_path):
    store = LocalStore()
    for i in range(5):
        store.write(str(tmp_path / "data" / "exact" / f"{i:06d}.bin"), b"x")
    store.write(str(tmp_path / "data" / "reps" / "000000.bin"), b"keep")
    assert delete_prefix(store, f"{tmp_path}/data/exact/") == 5
    assert store.list(f"{tmp_path}/data/exact/") == []
    assert store.read(str(tmp_path / "data" / "reps" / "000000.bin")) == b"keep"
    store.delete(str(tmp_path / "data" / "exact" / "000000.bin"))


def test_fsspec_store_round_trip_and_listing():
    store = FsspecStore("memory")
    store.write("memory://a4test/run/data/x.bin", b"abcdef")
    store.write("memory://a4test/run/data/y.bin", [b"12", b"34"])
    assert store.read("memory://a4test/run/data/x.bin", 1, 4) == b"bcd"
    assert store.read("memory://a4test/run/data/y.bin") == b"1234"
    with pytest.raises(PreconditionFailed):
        store.create("memory://a4test/run/data/x.bin", b"again")
    listed = [i.path.rsplit("/", 1)[1] for i in store.list("memory://a4test/run/data/")]
    assert listed == ["x.bin", "y.bin"]
    assert store.stat("memory://a4test/run/data/z.bin") is None
    store.delete("memory://a4test/run/data/x.bin")
    assert [i.size for i in store.list("memory://a4test/run/")] == [4]


def test_store_for_picks_the_backend_by_scheme():
    assert isinstance(store_for("/tmp/run"), LocalStore)
    assert isinstance(store_for("memory://bucket/run"), FsspecStore)
