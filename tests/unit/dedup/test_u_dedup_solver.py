# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""U-DEDUP rules 5 and 6: the c075 containment solver, its solve loop and the oversized split (B section 6(a))."""

import json
from pathlib import Path

import pyarrow as pa
import pytest

from atlantic_cc.dedup import solver
from atlantic_cc.dedup.params import C075_PARAMS
from atlantic_cc.dedup.runtime import tie_order_platform
from atlantic_cc.dedup.solver import ClusterDocument, find_duplicates, ngram_hashes, prepare
from tests.unit.dedup import synthetic

GOLDEN = Path(__file__).parents[2] / "golden" / "dedup"


@pytest.fixture(scope="module")
def rules():
    return json.loads((GOLDEN / "rules.json").read_text())


@pytest.fixture(scope="module")
def asrun():
    return json.loads((GOLDEN / "asrun_goldens.json").read_text())


def _words(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{i}" for i in range(n)]


def _solve(members: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """``(member id, representative id)`` removals of one group, members ordered by id as the solve loop orders them."""
    members = sorted(members)
    documents = [ClusterDocument(id=i, text=t, file_idx=0, source_tag="source_000") for i, t in members]
    removals = find_duplicates(documents, C075_PARAMS)
    return [(members[r.member_index][0], members[r.representative_index][0]) for r in removals]


def _containment(member: str, representative: str) -> float:
    m, r = ngram_hashes(member, 3), ngram_hashes(representative, 3)
    return solver._overlap(m, r) / m.size


def test_micro_cluster_matches_the_as_run_removals(rules):
    base = " ".join(_words("w", 100))
    members = [
        ("1" * 32, base),
        ("2" * 32, " ".join(_words("w", 80))),
        ("3" * 32, " ".join(_words("z", 50))),
        ("0" * 32, base),
    ]
    got = [{"member": m[:4], "rep": r[:4]} for m, r in _solve(members)]
    assert got == [{"member": g["member"], "rep": g["rep"]} for g in rules["c075_micro"]]


def test_containment_of_exactly_three_quarters_removes_and_just_below_keeps():
    member = " ".join(_words("t", 102))  # 100 word 3-grams
    filler = _words("zz", 60)
    holds_75 = " ".join(_words("t", 77) + filler)  # 75 of the member's 3-grams
    holds_74 = " ".join(_words("t", 76) + filler)
    assert _containment(member, holds_75) == 0.75 and _containment(member, holds_74) == 0.74
    assert _solve([("a", member), ("b", holds_75)]) == [("a", "b")]
    assert _solve([("a", member), ("b", holds_74)]) == []


def test_the_shorter_document_is_removed_and_equal_lengths_keep_the_smaller_id():
    long = " ".join(_words("w", 120))
    short = " ".join(_words("w", 100))
    assert _solve([("0", short), ("1", long)]) == [("0", "1")]
    assert _solve([("9", long), ("4", long)]) == [("9", "4")]


def test_a_removed_document_never_represents_another():
    c = _words("c", 10)
    y = _words("y", 100)
    a = " ".join(y + _words("z", 60))  # holds most of b, nothing of c
    b = " ".join(c + y)  # holds all of c
    assert _containment(" ".join(c), b) == 1.0 and _containment(" ".join(c), a) == 0.0
    assert _containment(b, a) >= 0.75
    assert _solve([("a", a), ("b", b), ("c", " ".join(c))]) == [("b", "a")]


def test_short_and_empty_token_lists():
    assert ngram_hashes("Hello WORLD", 3).size == 1  # fewer than 3 tokens: one shingle of all of them
    assert ngram_hashes(" \t ", 3).size == 0
    assert _solve([("0", "hello world"), ("1", "Hello   World")]) == [("0", "1")]
    assert _solve([("0", "hello world"), ("1", "hello world again")]) == []
    # A document without tokens is never removed and never holds anything.
    assert _solve([("0", "   "), ("1", "      ")]) == []


def test_casefold_matches_sharp_s_to_ss():
    assert _solve([("0", "STRASSE x y"), ("1", "straße x y z")]) == [("0", "1")]


def test_batches_over_256_members_use_the_inverted_index():
    text = "a b c d e"
    for n, last_candidates in ((256, 255), (257, C075_PARAMS.maximum_candidates)):
        prepared = prepare([ClusterDocument(f"{i:04d}", text, 0, "") for i in range(n)], C075_PARAMS)
        candidates = solver._candidate_pairs(prepared, C075_PARAMS)
        assert len(candidates[n - 1]) == last_candidates


def test_probes_held_only_by_the_member_give_no_candidates():
    documents = [ClusterDocument(f"{i:04d}", "a b c d e f", 0, "") for i in range(256)]
    documents.append(ClusterDocument("9999", "p q r s", 0, ""))
    candidates = solver._candidate_pairs(prepare(documents, C075_PARAMS), C075_PARAMS)
    assert candidates[256] == []


def test_ngrams_over_512_postings_are_used_when_nothing_rarer_exists():
    # 600 documents with the same three 3-grams (each held 600 > 512 times), longer by trailing spaces.
    documents = [ClusterDocument(f"{i:04d}", "a b c d e" + " " * i, 0, "") for i in range(600)]
    candidates = solver._candidate_pairs(prepare(documents, C075_PARAMS), C075_PARAMS)
    # Ranked by length, document 594 is sixth: the five longer documents are its candidates.
    assert sorted(candidates[594]) == [595, 596, 597, 598, 599]
    removed = {r.member_index for r in find_duplicates(documents, C075_PARAMS)}
    assert {594, 595, 596, 597, 598} <= removed and 599 not in removed


def test_the_32_candidate_cap_applies_before_removed_documents_are_skipped():
    """M's 32 best-sharing candidates are all removed by then, so M survives although S holds 78% of it."""
    m = _words("m", 34)  # 32 word 3-grams
    members = [("M", " ".join(m))]
    for i in range(32):
        unique = _words(f"u{i:02d}x", 202)
        members.append((f"T{i:02d}", " ".join(m[:32] + unique)))  # 30 of M's 3-grams; removed by K
        members.append((f"K{i:02d}", " ".join(unique + _words(f"k{i:02d}x", 50))))
    s = " ".join(m[:27] + _words("v", 32))
    members.append(("S", s))
    members += [(f"F{j:03d}", " ".join(_words(f"f{j:03d}x", 6))) for j in range(200)]
    assert len(members) > C075_PARAMS.exact_scan_maximum
    assert _containment(" ".join(m), s) >= C075_PARAMS.minimum_containment
    removed = {member for member, _ in _solve(members)}
    assert removed == {f"T{i:02d}" for i in range(32)}


def test_split_hash_and_cluster_keys_match_the_as_run_values(rules):
    assert solver.split_hash("the quick brown fox jumps over the lazy dog") == rules["split_hash"]["fox"]
    assert solver.split_hash("A b C") == rules["split_hash"]["three_tokens"]
    assert solver.split_hash("") == rules["split_hash"]["empty"]
    fox = "the quick brown fox jumps over the lazy dog"
    assert solver.cluster_key(123456789, fox, "0" * 32, 1) == rules["cluster_key"]["small"]
    # :04d is a minimum width: 45,573 splits give five-digit split indices.
    assert solver.cluster_key(123456789, fox, "0" * 32, 45573) == rules["cluster_key"]["split"]
    assert solver.cluster_key(7, fox, "0" * 32, 2) == f"7:{rules['split_hash']['fox'] % 2:04d}:03"


@pytest.mark.parametrize("cap", ["268435456", "2500"])
def test_solve_loop_matches_the_as_run_solve_text_shard(monkeypatch, asrun, cap):
    """Port of the as-run test_solve: the per-document cut, byte-bounded batches and mid-cluster flushes."""
    golden = asrun["solver_exact_scan"][cap]
    monkeypatch.setattr(solver, "SOLVE_MAX_CLUSTER_CHARS", int(cap))
    monkeypatch.setattr(solver, "SOLVE_MAX_DOCUMENT_CHARS", 400)
    rows = synthetic.solve_rows(int(cap), big_cluster=200)
    id_of = {r["d"]: r["id"] for r in rows}
    stats = solver.SolveStats()
    removed = sorted(id_of[d] for d in solver.solve_sorted_rows([dict(r) for r in rows], stats))
    assert removed == golden["removed_ids"]
    assert vars(stats) == golden["stats"]


@pytest.mark.parametrize("cap", ["268435456", "2500"])
def test_index_path_matches_the_as_run_solver_on_this_platform(monkeypatch, asrun, cap):
    """Batches over 256 members break argpartition ties in the dispatched SIMD kernel's order (see tie-order test)."""
    tag = tie_order_platform()
    if tag not in asrun["solver_index_path"]:
        pytest.skip(f"no as-run golden for {tag}; generate one with generate_asrun.py --platform-only")
    golden = asrun["solver_index_path"][tag][cap]
    monkeypatch.setattr(solver, "SOLVE_MAX_CLUSTER_CHARS", int(cap))
    monkeypatch.setattr(solver, "SOLVE_MAX_DOCUMENT_CHARS", 400)
    rows = synthetic.solve_rows(int(cap), big_cluster=300)
    id_of = {r["d"]: r["id"] for r in rows}
    removed = sorted(id_of[d] for d in solver.solve_sorted_rows([dict(r) for r in rows], solver.SolveStats()))
    assert removed == golden["removed_ids"]


def test_sorted_rows_order_by_cluster_key_then_id():
    table = pa.table(
        {
            "cluster_key": ["b", "a", "a", "b"],
            "id": ["2", "9", "1", "0"],
            "text": pa.array(["x", "y", "z", "w"], type=pa.large_string()),
            "d": pa.array([1, 2, 3, 4], type=pa.uint64()),
        }
    )
    assert [r["d"] for r in solver.sorted_rows(table)] == [3, 2, 4, 1]
