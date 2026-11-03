# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A slow, pure-Python port of the dupekit MinHash/LSH pipeline, as a readable reference and test oracle.

Ported from the Rust sources of ``marin-dupekit-native`` 0.1.6 (``minhash_ops.rs``) and the crates it pins:

* ``CleanText``: whole-string lowercase, drop the 32 ASCII punctuation characters, collapse runs of Unicode
  ``White_Space`` to one space and strip both ends.
* ``MinHash``: character 5-grams of the cleaned text (one hash of the whole string when it has fewer than 5
  characters), each hashed with ``xxh3_64``; 286 odd u128 multipliers drawn from ``rand_pcg::Pcg64`` seeded with
  ``seed_from_u64(42)`` (rand 0.8.6, rand_core 0.6.4, rand_pcg 0.3.1); permutation ``i`` of a hash ``h`` is
  ``(h * a_i mod 2**128) >> 64`` and the signature keeps the minimum per permutation.
* ``MinHashLSH``: bucket ``b`` is ``xxh3_64`` of the little-endian bytes of signature values ``[11b, 11b + 11)``.
  The band index is not mixed in.

The only known divergence: Python 3.12's ``str.lower`` uses Unicode 15.0 tables while the Rust standard library
in the wheel is newer (for example it lowercases U+1C89), so ``clean_text`` can differ on such characters.
"""

import struct

import xxhash

from atlantic_cc.dedup.params import NGRAM_SIZE, NUM_BANDS, NUM_PERMS, SEED

_MASK64 = (1 << 64) - 1
_MASK128 = (1 << 128) - 1
_PCG32_MULTIPLIER = 6364136223846793005
_PCG32_INCREMENT = 11634580027462260723
_PCG128_MULTIPLIER = 0x2360ED051FC65DA44385DF649FCCF645

# Rust char::is_whitespace: the Unicode White_Space property.
_WHITE_SPACE = frozenset(
    [chr(c) for c in range(0x09, 0x0E)]
    + [" ", "\u0085", "\u00a0", "\u1680"]
    + [chr(c) for c in range(0x2000, 0x200B)]
    + ["\u2028", "\u2029", "\u202f", "\u205f", "\u3000"]
)
# Rust char::is_ascii_punctuation.
_ASCII_PUNCTUATION = frozenset("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def _rotate_right32(x: int, r: int) -> int:
    r &= 31
    return ((x >> r) | (x << (32 - r))) & 0xFFFFFFFF


def _rotate_right64(x: int, r: int) -> int:
    r &= 63
    return ((x >> r) | (x << (64 - r))) & _MASK64


def seed_from_u64(state: int) -> bytes:
    """rand_core 0.6.4 ``SeedableRng::seed_from_u64``: a PCG32 stream expanded into a 32-byte seed."""
    seed = b""
    for _ in range(8):
        state = (state * _PCG32_MULTIPLIER + _PCG32_INCREMENT) & _MASK64
        xorshifted = (((state >> 18) ^ state) >> 27) & 0xFFFFFFFF
        seed += struct.pack("<I", _rotate_right32(xorshifted, state >> 59))
    return seed


class Pcg64:
    """rand_pcg 0.3.1 ``Lcg128Xsl64``: a 128-bit LCG with XSL-RR output of the post-step state."""

    def __init__(self, seed: bytes) -> None:
        s0, s1, s2, s3 = struct.unpack("<4Q", seed)
        self.increment = (s2 | (s3 << 64)) | 1
        self.state = ((s0 | (s1 << 64)) + self.increment) & _MASK128
        self._step()

    def _step(self) -> None:
        self.state = (self.state * _PCG128_MULTIPLIER + self.increment) & _MASK128

    def next_u64(self) -> int:
        self._step()
        return _rotate_right64(((self.state >> 64) ^ self.state) & _MASK64, self.state >> 122)


def coefficients(num_perms: int = NUM_PERMS, seed: int = SEED) -> list[int]:
    """The odd u128 multipliers: rand 0.8.6 ``gen::<u128>()`` is the low u64 then the high u64."""
    rng = Pcg64(seed_from_u64(seed))
    out = []
    for _ in range(num_perms):
        low = rng.next_u64()
        coefficient = low | (rng.next_u64() << 64)
        if coefficient % 2 == 0:
            coefficient = (coefficient + 1) & _MASK128
        out.append(coefficient)
    return out


def clean_text(text: str) -> str:
    out = []
    pending_space = False
    for character in text.lower():
        if character in _ASCII_PUNCTUATION:
            continue
        if character in _WHITE_SPACE:
            pending_space = bool(out)
            continue
        if pending_space:
            out.append(" ")
            pending_space = False
        out.append(character)
    return "".join(out)


def shingle_hashes(cleaned: str, ngram_size: int = NGRAM_SIZE) -> list[int]:
    if len(cleaned) < ngram_size:
        return [xxhash.xxh3_64_intdigest(cleaned.encode("utf-8"))]
    return [
        xxhash.xxh3_64_intdigest(cleaned[start : start + ngram_size].encode("utf-8"))
        for start in range(len(cleaned) - ngram_size + 1)
    ]


def signature(cleaned: str, multipliers: list[int], ngram_size: int = NGRAM_SIZE) -> list[int]:
    hashes = shingle_hashes(cleaned, ngram_size)
    return [min(((h * a) & _MASK128) >> 64 for h in hashes) for a in multipliers]


def lsh_buckets(values: list[int], num_bands: int = NUM_BANDS) -> list[int]:
    rows = len(values) // num_bands
    return [
        xxhash.xxh3_64_intdigest(struct.pack(f"<{rows}Q", *values[b * rows : (b + 1) * rows])) for b in range(num_bands)
    ]


def buckets(text: str, multipliers: list[int] | None = None) -> list[int]:
    """The 26 LSH buckets of one document, as :func:`atlantic_cc.dedup.minhash.minhash_buckets` computes them."""
    return lsh_buckets(signature(clean_text(text), multipliers if multipliers is not None else coefficients()))
