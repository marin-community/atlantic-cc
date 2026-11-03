# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""U-NORM: the classifier input normalization."""

import re

import pytest

from atlantic_cc.classify.text import normalize_text

# Every code point Python 3.12 (Unicode 15.0) treats as whitespace, i.e. what ``\s`` collapses.
PY312_WHITESPACE = (
    "\t\n\x0b\x0c\r\x1c\x1d\x1e\x1f \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006"
    "\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000"
)


def test_whitespace_set_is_python_3_12s():
    every = "".join(map(chr, range(0x110000)))
    candidates = set(re.findall(r"\s", every)) | {c for c in every if c.isspace()}
    assert "".join(sorted(candidates)) == PY312_WHITESPACE
    assert all(normalize_text(f"a{c}b") == "a b" for c in candidates)


@pytest.mark.parametrize(
    ("text", "normalized"),
    [
        ("  Hello,\n\tWORLD  ", "hello, world"),
        ("a\xa0\u3000\u2028b", "a b"),
        ("zero\u200bwidth\ufeffjoiners", "zero\u200bwidth\ufeffjoiners"),  # not whitespace: kept
        ("\u0130STANBUL", "i\u0307stanbul"),  # dotted capital I lowers to two code points
        (
            "\u039f\u0394\u039f\u03a3 \u039f\u0394\u039f\u03a3",
            "\u03bf\u03b4\u03bf\u03c2 \u03bf\u03b4\u03bf\u03c2",
        ),  # final sigma
        ("\u1e9e", "\xdf"),
        ("", ""),
        (" \n ", ""),
    ],
)
def test_normalize_text(text, normalized):
    assert normalize_text(text) == normalized
