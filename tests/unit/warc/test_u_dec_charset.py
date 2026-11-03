# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""U-DEC: the strict charset cascade, with expected strings produced by the as-run ``decode_payload``
(``decode_warcs_clean.py`` at tag ``lpv11-8m-as-run``) under charset-normalizer 3.5.0."""

import codecs

import pytest

from atlantic_cc.warc.decode import decode_payload

RU = "Привет, как дела? Это проверка кодировки страницы."
JP = "日本語のテキストです。これは文字コードの判定テストです。漢字とひらがなとカタカナ。"
ENGLISH_PAGE = "<html><body><p>" + "Some ordinary English words in a sentence. " * 20


@pytest.mark.parametrize(
    ("raw", "content_type", "expected"),
    [
        pytest.param(
            codecs.BOM_UTF8 + "héllo wörld".encode(),
            "text/html; charset=windows-1252",
            "héllo wörld",
            id="bom-beats-header",
        ),
        pytest.param(codecs.BOM_UTF16_LE + "hi ü".encode("utf-16-le"), None, "hi ü", id="bom-utf16"),
        pytest.param(
            b'<meta charset="koi8-r">' + RU.encode("cp1251"),
            "text/html; charset=windows-1251",
            '<meta charset="koi8-r">' + RU,
            id="header-beats-meta",
        ),
        pytest.param(
            b'<meta http-equiv="Content-Type" content="text/html; charset=windows-1251">' + RU.encode("cp1251"),
            "text/html",
            '<meta http-equiv="Content-Type" content="text/html; charset=windows-1251">' + RU,
            id="meta-without-header-charset",
        ),
        pytest.param(
            b"<meta charset='windows-1251'>" + RU.encode("cp1251"),
            None,
            "<meta charset='windows-1251'>" + RU,
            id="meta-single-quotes",
        ),
        pytest.param(
            b" " * 900 + b'<meta charset="koi8-r">\xf0\xd2\xc9',
            "text/html",
            " " * 900 + '<meta charset="koi8-r">При',
            id="meta-in-first-1024",
        ),
        pytest.param(
            b" " * 1024 + b'<meta charset="koi8-r">\xf0\xd2\xc9',
            "text/html",
            " " * 1024 + '<meta charset="koi8-r">ðÒÉ',
            id="meta-past-1024-ignored",
        ),
        pytest.param(
            b'<meta charset="cp1251">\xcf\xf0\xe8',
            "text/html; charset=shift_jis",
            '<meta charset="cp1251">При',
            id="invalid-header-then-meta",
        ),
        pytest.param(
            b"\x93quoted\x94 caf\xe9", "text/html; charset=ISO-8859-1", "“quoted” café", id="latin1-is-cp1252"
        ),
        pytest.param(b"caf\xe9", "text/html; charset=us-ascii", "café", id="ascii-is-cp1252"),
        pytest.param(b"\x93x\x94", 'text/html; charset = "Windows-1252"', "“x”", id="header-spacing-quotes"),
        pytest.param("naïve".encode(), "text/html; charset=utf8mb4", "naïve", id="unknown-label-falls-through"),
        pytest.param("naïve café".encode(), None, "naïve café", id="undeclared-utf8"),
        pytest.param(JP.encode("shift_jis"), "text/html", JP, id="charset-normalizer-multibyte"),
        pytest.param(
            b"caf\xe9 \x93x\x94 and more plain text here",
            "text/html; charset=utf-8",
            "café “x” and more plain text here",
            id="cp1252-default",
        ),
    ],
)
def test_cascade(raw, content_type, expected):
    assert decode_payload(raw, content_type) == expected


def test_charset_normalizer_acceptance_rule():
    """Undeclared pages: a coherent single-byte guess is accepted, and so is any multi-byte guess."""
    assert decode_payload(RU.encode("cp1251"), "text/html") == RU  # cp1251, coherence 0.56
    # KOI8-R is guessed as shift_jis_2004 (multi-byte usage 0.04, coherence 0) and accepted.
    assert decode_payload(RU.encode("koi8-r"), "text/html") == (
        "\u5d39\uff89\uff97\uff85\uff94, \uff8b\uff81\uff8b \uff84\uff85\uff8c\uff81? \u9e9b\uff8f \uff90"
        "\uff92\uff8f\uff97\uff85\uff92\uff8b\uff81 \uff8b\uff8f\uff84\uff89\uff92\uff8f\uff97\uff8b"
        "\uff89 \uff93\uff94\uff92\uff81\uff8e\uff89\uff83\uff99."
    )


def test_literal_replacement_character_never_survives():
    """Valid UTF-8 holding a literal U+FFFD fails every strict step. charset_normalizer's guess is
    then used if it is multi-byte or coherent, otherwise cp1252, otherwise latin-1."""
    assert decode_payload("x \ufffd y".encode(), "text/html") == "x 魹ｽ y"  # shift_jis_2004 guess
    page = (ENGLISH_PAGE + "\ufffd</p></body></html>").encode()
    assert decode_payload(page, "text/html; charset=utf-8") == ENGLISH_PAGE + "ï¿½</p></body></html>"
    assert decode_payload("a \ufffd b \x81".encode(), "text/html; charset=utf-8") == "a ï¿½ b Â\x81"  # 0x81: latin-1


@pytest.mark.parametrize("label", ["hex", "base64", "zlib", "rot13"])
def test_bytes_codec_label_raises_lookup_error(label):
    with pytest.raises(LookupError):
        decode_payload(b"abc", f"text/html; charset={label}")
