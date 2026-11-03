# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The c075 containment rule, vendored from the as-run code.

* :func:`find_duplicates` and its helpers are the fork's ``cluster_dedup.py`` at f47919338b (the version the
  ``lpv11-8m-as-run`` tag carries), with ``ClusterDedupParams`` as a dataclass instead of a pydantic model.
* :func:`split_hash` is ``fuzzy_cluster_text._split_hash`` and :func:`cluster_key` is ``rdedup/stages.py:797-807``.
* :func:`solve_sorted_rows` is ``rdedup/stages.py:887-975``, itself a line-for-line port of the fork's
  ``cluster_verify.solve_text_shard`` loop.

A document is removed when a longer surviving document of its group already holds at least 0.75 of its hashed,
casefolded word 3-grams. In batches of more than 256 members the candidates come from an inverted index, and two
``np.argpartition`` calls there break ties in the order of the SIMD kernel NumPy dispatches at runtime
(``ArgQSelect_AVX2`` vs ``ArgQSelect_AVX512_SKX`` vs scalar): the release was built on AVX2 hosts, so reproduce on
AVX2 or with AVX-512 masked (see :mod:`atlantic_cc.dedup.runtime`).
"""

import dataclasses
from collections.abc import Iterable, Iterator, Sequence

import dupekit_native as dupekit
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from atlantic_cc.dedup.params import (
    C075_PARAMS,
    CC_SOURCE_TAG,
    SOLVE_MAX_CLUSTER_CHARS,
    SOLVE_MAX_DOCUMENT_CHARS,
    SPLIT_NGRAM_SIZE,
    SPLIT_SUBDIVISIONS,
    ClusterDedupParams,
)

# 64-bit hashes over a cluster of at most 100,000 documents collide with negligible probability, so a hash set
# stands in for the n-gram set itself.
_EMPTY = np.empty(0, dtype=np.uint64)
# Rows are materialized from the sorted table this many at a time.
_ROW_SLICE = 8192


@dataclasses.dataclass(frozen=True)
class ClusterDocument:
    """One cluster member as the materialized dataset stores it."""

    id: str
    text: str
    file_idx: int
    source_tag: str


@dataclasses.dataclass(frozen=True)
class PreparedDocument:
    """N-gram hashes and the sizes the rule compares."""

    index: int
    chars: int
    tokens: int
    ngrams: np.ndarray
    text: str


@dataclasses.dataclass(frozen=True)
class Removal:
    """One duplicate decision."""

    member_index: int
    representative_index: int
    containment: float
    jaccard: float
    novel_tokens: int
    comparisons: int


def ngram_hashes(text: str, ngram_size: int) -> np.ndarray:
    """Sorted unique 64-bit hashes of the case-folded word n-grams."""
    tokens = text.casefold().split()
    if not tokens:
        return _EMPTY
    if len(tokens) < ngram_size:
        shingles = [" ".join(tokens).encode("utf-8", "surrogatepass")]
    else:
        shingles = [
            " ".join(tokens[start : start + ngram_size]).encode("utf-8", "surrogatepass")
            for start in range(len(tokens) - ngram_size + 1)
        ]
    return np.unique(np.asarray(dupekit.hash_xxh3_64_batch(shingles), dtype=np.uint64))


def prepare(documents: Sequence[ClusterDocument], params: ClusterDedupParams) -> list[PreparedDocument]:
    """Prepare every member of one cluster."""
    prepared = []
    for index, document in enumerate(documents):
        prepared.append(
            PreparedDocument(
                index=index,
                chars=len(document.text),
                tokens=document.text.count(" ") + 1,
                ngrams=ngram_hashes(document.text, params.ngram_size),
                text=document.text,
            )
        )
    return prepared


def novel_token_count(
    member: PreparedDocument,
    representative: PreparedDocument,
    cache: dict[int, frozenset[str]] | None = None,
) -> int:
    """Words of the member that the representative does not hold."""
    if cache is None:
        cache = {}
    for document in (member, representative):
        if document.index not in cache:
            cache[document.index] = frozenset(document.text.casefold().split())
    return len(cache[member.index] - cache[representative.index])


def _overlap(left: np.ndarray, right: np.ndarray) -> int:
    """Size of the intersection of two sorted unique hash arrays."""
    if left.size == 0 or right.size == 0:
        return 0
    if left.size > right.size:
        left, right = right, left
    position = np.searchsorted(right, left)
    position[position >= right.size] = right.size - 1
    return int(np.count_nonzero(right[position] == left))


@dataclasses.dataclass(frozen=True)
class _NgramIndex:
    """Inverted index from n-gram hash to the documents that hold it."""

    values: np.ndarray  # distinct n-gram hashes, ascending
    counts: np.ndarray  # how many documents hold each hash
    starts: np.ndarray  # where each hash's document list begins in ``owners``
    owners: np.ndarray


def _build_index(prepared: list[PreparedDocument]) -> _NgramIndex:
    sizes = np.fromiter((document.ngrams.size for document in prepared), dtype=np.int64, count=len(prepared))
    values = np.concatenate([document.ngrams for document in prepared]) if sizes.sum() else _EMPTY
    owners = np.repeat(np.arange(len(prepared), dtype=np.int32), sizes)
    order = np.argsort(values, kind="stable")
    values = values[order]
    owners = owners[order]
    if values.size == 0:
        return _NgramIndex(values=values, counts=_EMPTY.astype(np.int64), starts=_EMPTY.astype(np.int64), owners=owners)
    # The array is already sorted, so the run boundaries are one comparison per element.
    boundary = np.empty(values.size, dtype=bool)
    boundary[0] = True
    np.not_equal(values[1:], values[:-1], out=boundary[1:])
    starts = np.flatnonzero(boundary)
    counts = np.diff(np.append(starts, values.size))
    return _NgramIndex(values=values[starts], counts=counts, starts=starts, owners=owners)


def _index_candidates(
    member: PreparedDocument,
    index: _NgramIndex,
    rank: np.ndarray,
    params: ClusterDedupParams,
) -> np.ndarray:
    """Documents that share the member's rarest n-grams, best first."""
    if member.ngrams.size == 0:
        return np.empty(0, dtype=np.int32)
    position = np.searchsorted(index.values, member.ngrams)
    position[position >= index.values.size] = index.values.size - 1
    present = index.values[position] == member.ngrams
    position = position[present]
    if position.size == 0:
        return np.empty(0, dtype=np.int32)

    counts = index.counts[position]
    usable = counts <= params.maximum_posting_length
    if np.any(usable):
        position, counts = position[usable], counts[usable]
    if position.size > params.probe_ngrams:
        # Tie order among equal counts is the dispatched SIMD kernel's (see the module docstring).
        rarest = np.argpartition(counts, params.probe_ngrams)[: params.probe_ngrams]
        position, counts = position[rarest], counts[rarest]

    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int32)
    offsets = np.repeat(index.starts[position], counts)
    within = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    owners = index.owners[offsets + within]

    distinct, shared = np.unique(owners, return_counts=True)
    # Only a document that ranks ahead of the member -- longer, or equal and earlier -- can be its representative.
    ahead = rank[distinct] < rank[member.index]
    distinct, shared = distinct[ahead], shared[ahead]
    if distinct.size == 0:
        return np.empty(0, dtype=np.int32)
    if distinct.size > params.maximum_candidates:
        # Tie-dependent as above.
        best = np.argpartition(-shared, params.maximum_candidates - 1)[: params.maximum_candidates]
        distinct, shared = distinct[best], shared[best]
    return distinct[np.argsort(-shared)].astype(np.int32)


def _candidate_pairs(prepared: list[PreparedDocument], params: ClusterDedupParams) -> dict[int, list[int]]:
    """Map each member to the representatives worth an exact comparison: only earlier-ranked documents."""
    order = sorted(range(len(prepared)), key=lambda index: (-prepared[index].chars, index))
    if len(prepared) <= params.exact_scan_maximum:
        rank = {index: position for position, index in enumerate(order)}
        return {index: [other for other in order if rank[other] < rank[index]] for index in order}

    rank = np.empty(len(prepared), dtype=np.int64)
    for position, index in enumerate(order):
        rank[index] = position
    ngram_index = _build_index(prepared)
    return {index: _index_candidates(prepared[index], ngram_index, rank, params).tolist() for index in order}


def find_duplicates(
    documents: Sequence[ClusterDocument],
    params: ClusterDedupParams,
    prepared: list[PreparedDocument] | None = None,
) -> list[Removal]:
    """Solve one cluster: every member that a longer survivor already holds.

    Walks members from the longest, so a representative is always decided before the documents it can absorb, and
    a removed document never becomes the representative of another.
    """
    prepared = prepared if prepared is not None else prepare(documents, params)
    candidates = _candidate_pairs(prepared, params)
    order = sorted(range(len(prepared)), key=lambda index: (-prepared[index].chars, index))

    removed: set[int] = set()
    removals: list[Removal] = []
    token_cache: dict[int, frozenset[str]] = {}
    for member in order:
        best: Removal | None = None
        comparisons = 0
        member_prepared = prepared[member]
        if member_prepared.ngrams.size == 0:
            continue
        for representative in candidates.get(member, ()):
            if representative in removed:
                continue
            other = prepared[representative]
            if member_prepared.chars > other.chars:
                continue
            comparisons += 1
            shared = _overlap(member_prepared.ngrams, other.ngrams)
            containment = shared / member_prepared.ngrams.size
            if containment < params.minimum_containment:
                continue
            novel_tokens = novel_token_count(member_prepared, other, token_cache)
            union = member_prepared.ngrams.size + other.ngrams.size - shared
            best = Removal(
                member_index=member,
                representative_index=representative,
                containment=containment,
                jaccard=shared / union if union else 1.0,
                novel_tokens=novel_tokens,
                comparisons=comparisons,
            )
            break
        if best is not None:
            removed.add(member)
            removals.append(best)
    return removals


# ------------------------------------------------------------------------------------------- grouping


def split_hash(text: str) -> int:
    """One MinHash permutation over the document's word 5-grams: the minimum ``xxh3_64`` of the casefolded
    5-grams (of the whole joined text below 5 words). Near-duplicates share it with probability = their Jaccard."""
    tokens = text.casefold().split()
    if len(tokens) < SPLIT_NGRAM_SIZE:
        return dupekit.hash_xxh3_64(" ".join(tokens).encode("utf-8", "ignore"))
    shingles = [
        " ".join(tokens[start : start + SPLIT_NGRAM_SIZE]).encode("utf-8", "ignore")
        for start in range(len(tokens) - SPLIT_NGRAM_SIZE + 1)
    ]
    return min(dupekit.hash_xxh3_64_batch(shingles))


def cluster_key(root: int, text: str, doc_id: str, splits: int) -> str:
    """The solve group of a member: its component, or for an oversized component a split of it.

    An oversized component splits by :func:`split_hash` modulo ``splits``, then uniformly by content id into 16
    sub-groups. ``:04d`` is a minimum width, so more than 9,999 splits give 5-digit indices.
    """
    if splits <= 1:
        return str(root)
    split_index = split_hash(text) % splits
    sub_index = dupekit.hash_xxh3_64(doc_id.encode()) % SPLIT_SUBDIVISIONS
    return f"{root}:{split_index:04d}:{sub_index:02d}"


# ------------------------------------------------------------------------------------------- solve loop


@dataclasses.dataclass
class SolveStats:
    documents: int = 0
    text_chars: int = 0
    clusters: int = 0
    singleton_groups: int = 0
    oversized_clusters: int = 0
    mid_cluster_flushes: int = 0
    truncated_documents: int = 0
    duplicates: int = 0


def solve_sorted_rows(rows: Iterable[dict], stats: SolveStats) -> Iterator[int]:
    """Doc keys the containment rule removes, from rows ``{cluster_key, id, text, d}`` sorted by
    ``(cluster_key, id)``.

    Each document is cut to 8 MiB. Inside a group, once the buffered text exceeds 256 MiB *before* the next row is
    appended, the buffer is solved as a batch and reset (documents in different flush batches are never compared).
    A group's final buffer over 256 MiB is sorted longest first (stable) and packed greedily into batches of at
    most 256 MiB.
    """
    params = C075_PARAMS

    def solve_batch(members: list[dict]) -> Iterator[int]:
        if len(members) < 2:
            return
        cluster = [ClusterDocument(id=m["id"], text=m["text"], file_idx=0, source_tag=CC_SOURCE_TAG) for m in members]
        for removal in find_duplicates(cluster, params):
            stats.duplicates += 1
            yield members[removal.member_index]["d"]

    def batches(members: list[dict]) -> Iterator[list[dict]]:
        if sum(len(m["text"]) for m in members) <= SOLVE_MAX_CLUSTER_CHARS:
            yield members
            return
        stats.oversized_clusters += 1
        ordered = sorted(members, key=lambda m: -len(m["text"]))
        batch: list[dict] = []
        size = 0
        for m in ordered:
            if batch and size + len(m["text"]) > SOLVE_MAX_CLUSTER_CHARS:
                yield batch
                batch, size = [], 0
            batch.append(m)
            size += len(m["text"])
        if batch:
            yield batch

    def solve(members: list[dict]) -> Iterator[int]:
        stats.clusters += 1
        if len(members) < 2:
            stats.singleton_groups += 1
            return
        for chunk in batches(members):
            yield from solve_batch(chunk)

    pending: list[dict] = []
    pending_chars = 0
    current: str | None = None
    for row in rows:
        stats.documents += 1
        stats.text_chars += len(row["text"])
        if len(row["text"]) > SOLVE_MAX_DOCUMENT_CHARS:
            stats.truncated_documents += 1
            row["text"] = row["text"][:SOLVE_MAX_DOCUMENT_CHARS]
        if row["cluster_key"] != current:
            if pending:
                yield from solve(pending)
            pending = []
            pending_chars = 0
            current = row["cluster_key"]
        elif pending_chars > SOLVE_MAX_CLUSTER_CHARS:
            stats.mid_cluster_flushes += 1
            yield from solve_batch(pending)
            pending = []
            pending_chars = 0
        pending.append(row)
        pending_chars += len(row["text"])
    if pending:
        yield from solve(pending)


def sorted_rows(table: pa.Table) -> Iterator[dict]:
    """Rows in ``(cluster_key, id)`` order, materialized a slice at a time."""
    order = pc.sort_indices(
        table.select(["cluster_key", "id"]), sort_keys=[("cluster_key", "ascending"), ("id", "ascending")]
    )
    for start in range(0, len(order), _ROW_SLICE):
        yield from table.take(order.slice(start, _ROW_SLICE)).to_pylist()
