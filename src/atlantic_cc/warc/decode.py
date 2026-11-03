# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Strict charset cascade that turns an HTML payload into text without ever inventing U+FFFD.

Order (as-run ``decode_warcs_clean.decode_payload``): byte-order mark, HTTP ``Content-Type``
charset, ``<meta charset>`` in the first 1024 bytes, strict UTF-8, ``charset_normalizer`` (its
guess, when multi-byte or coherent), then cp1252 and finally latin-1. Every step decodes strictly
and is rejected if its output contains U+FFFD. Python's own codec registry resolves every label;
the only WHATWG mapping applied is latin-1/ASCII -> cp1252.

Two consequences are part of the released data and are kept on purpose:

* A UTF-8 page that contains a literal U+FFFD fails the BOM, label and UTF-8 steps. It is then
  decoded with charset_normalizer's guess when that guess is accepted (any multi-byte guess is: a
  short page can come out as ``shift_jis_2004`` CJK), otherwise as cp1252 mojibake (``ï¿½``), or as
  latin-1 when cp1252 hits one of its five undefined bytes.
* A declared bytes-to-bytes codec (``hex``, ``base64``, ``zlib``, ...) passes ``codecs.lookup`` and
  then makes ``bytes.decode`` raise ``LookupError``, which escapes this module; the record loop
  skips such records.
"""

import codecs
import re

import charset_normalizer

REPLACEMENT_CHAR = "\ufffd"

_HTTP_CHARSET_RE = re.compile(r"charset\s*=\s*[\"']?([a-zA-Z0-9_\-:.]+)", re.IGNORECASE)
_META_CHARSET_RE = re.compile(rb"""<meta[^>]+charset[\s"'=]*([a-zA-Z0-9_\-:.]+)""", re.IGNORECASE)
META_SCAN_BYTES = 1024

# Labels decoded as windows-1252, as browsers do: pages declaring latin-1 routinely use the cp1252
# punctuation range 0x80-0x9F (curly quotes, dashes).
_LATIN1_FAMILY = frozenset(
    {
        "iso-8859-1",
        "iso8859-1",
        "iso_8859-1",
        "iso-8859-1:1987",
        "latin-1",
        "latin1",
        "l1",
        "ascii",
        "us-ascii",
        "cp819",
        "ibm819",
        "8859-1",
        "windows-1252",
        "cp1252",
    }
)
# charset_normalizer's guess is trusted only for a multi-byte encoding or a single-byte one with
# real language coherence; anything weaker falls through to the cp1252 default.
_COHERENCE_MIN = 0.50


def _normalize_label(label: str | None) -> str | None:
    if not label:
        return None
    if label.strip().lower() in _LATIN1_FAMILY:
        return "cp1252"
    return label


def _charset_from_header(content_type: str | None) -> str | None:
    if not content_type:
        return None
    m = _HTTP_CHARSET_RE.search(content_type)
    return m.group(1) if m else None


def _bom_charset(raw: bytes) -> str | None:
    if raw.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if raw.startswith(codecs.BOM_UTF16_LE) or raw.startswith(codecs.BOM_UTF16_BE):
        return "utf-16"
    return None


def _try_strict(raw: bytes, encoding: str | None) -> str | None:
    """Strict decode, or None for an unknown label or invalid bytes.

    ``LookupError`` from ``bytes.decode`` (a registered non-text codec) is deliberately not caught.
    """
    if not encoding:
        return None
    try:
        codecs.lookup(encoding)
    except LookupError:
        return None
    try:
        return raw.decode(encoding)
    except (UnicodeDecodeError, ValueError):
        return None


def decode_payload(raw: bytes, content_type: str | None) -> str:
    """Decode HTML payload bytes with the as-run cascade (see the module docstring)."""
    bom = _bom_charset(raw)
    if bom:
        s = _try_strict(raw, bom)
        if s is not None and REPLACEMENT_CHAR not in s:
            return s

    s = _try_strict(raw, _normalize_label(_charset_from_header(content_type)))
    if s is not None and REPLACEMENT_CHAR not in s:
        return s

    m = _META_CHARSET_RE.search(raw[:META_SCAN_BYTES])
    if m:
        s = _try_strict(raw, _normalize_label(m.group(1).decode("ascii", "ignore")))
        if s is not None and REPLACEMENT_CHAR not in s:
            return s

    s = _try_strict(raw, "utf-8")
    if s is not None and REPLACEMENT_CHAR not in s:
        return s

    best = charset_normalizer.from_bytes(raw).best()
    if best is not None and (best.multi_byte_usage > 0 or best.coherence >= _COHERENCE_MIN):
        s = str(best)
        if REPLACEMENT_CHAR not in s:
            return s

    try:
        return raw.decode("cp1252")
    except (UnicodeDecodeError, ValueError):
        return raw.decode("latin-1")
