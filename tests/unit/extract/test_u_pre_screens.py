# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""U-PRE (in-process): population prefilter and the per-page screens in front of the engine."""

import sys
import types

import pytest

from atlantic_cc import spec
from atlantic_cc.extract.engine import extract_text, screen_batch
from atlantic_cc.extract.prefilter import body_strip, in_population

CAP = spec.MAX_HTML_CHARS


@pytest.mark.parametrize(
    ("html", "stripped"),
    [
        ("<html><head><script>var x;</script></head><body>a</body></html>", "a"),
        ("<SCRIPT type='t'>\n<body>x</body></script > <body> y </body>", " y "),
        ("<body class=x>a</body><p>between</p><BODY>b</Body >", "ab"),
        ("<p>no body tag <b>kept</b></p><script>gone</script>", "<p>no body tag <b>kept</b></p>"),
        ("<body>unclosed", "<body>unclosed"),
        ("<html><body></body>outside</html>", ""),
    ],
)
def test_body_strip(html, stripped):
    assert body_strip(html) == stripped


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        ("<html><body></body>content outside the body</html>", False),
        ("<body> \xa0\u3000\n\t</body>", False),  # Unicode whitespace only
        ("<body><script>only a script</script></body>", False),
        ("<body><br></body>", True),  # tags are not removed, so markup counts
        ("<body>\u200b</body>", True),  # ZERO WIDTH SPACE is not whitespace
        ("plain text without any tags", True),
        ("", False),
    ],
)
def test_in_population(html, expected):
    assert in_population(html) is expected


class _RecordingEngine:
    def __init__(self, raises: BaseException | None = None):
        self.calls: list[str] = []
        self._raises = raises

    def extract_plain_text(self, html, main_content=False, preserve_formatting=False):
        assert main_content is True and preserve_formatting == "markdown"
        self.calls.append(html)
        if self._raises is not None:
            raise self._raises
        return "md:" + html


@pytest.fixture
def engine(monkeypatch):
    def install(raises: BaseException | None = None) -> _RecordingEngine:
        fake = _RecordingEngine(raises)
        module = types.ModuleType("resiliparse._extract_rs")
        module.extract_plain_text = fake.extract_plain_text
        monkeypatch.setitem(sys.modules, "resiliparse", types.ModuleType("resiliparse"))
        monkeypatch.setitem(sys.modules, "resiliparse._extract_rs", module)
        return fake

    return install


@pytest.mark.parametrize("html", ["<html><frameset cols='20%,80%'></frameset></html>", "<HTML><FRAMESET ROWS='*'>"])
def test_frameset_pages_never_reach_the_engine(engine, html):
    fake = engine()
    assert extract_text(html, CAP) == ""
    assert fake.calls == []


def test_mixed_case_frameset_reaches_the_engine(engine):
    fake = engine()
    assert extract_text("<FrameSet>", CAP) == "md:<FrameSet>"
    assert fake.calls == ["<FrameSet>"]


def test_size_cap_is_in_code_points(engine):
    fake = engine()
    at_cap = "é" * CAP  # 2 bytes per code point in UTF-8: only code points count
    assert extract_text(at_cap + "x", CAP) == ""
    assert fake.calls == []
    assert extract_text(at_cap, CAP) == "md:" + at_cap


def test_engine_failures_become_empty_text(engine):
    engine(raises=type("PanicException", (BaseException,), {})("boom"))
    assert extract_text("<p>x</p>", CAP) == ""
    engine(raises=UnicodeEncodeError("utf-8", "\ud800", 0, 1, "surrogates not allowed"))
    assert extract_text("<p>\ud800</p>", CAP) == ""


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), SystemExit(1)])
def test_interrupts_propagate(engine, exc):
    engine(raises=exc)
    with pytest.raises(type(exc)):
        extract_text("<p>x</p>", CAP)


def test_missing_engine_raises_instead_of_returning_empty(monkeypatch):
    monkeypatch.setitem(sys.modules, "resiliparse", types.ModuleType("resiliparse"))
    monkeypatch.setitem(sys.modules, "resiliparse._extract_rs", None)  # import of it raises ImportError
    with pytest.raises(ImportError):
        extract_text("<p>x</p>", CAP)
    assert extract_text("<frameset>", CAP) == ""  # the screens run before the import


def test_screen_batch_routes_each_page(engine):
    fake = engine()
    pages = ["<body> </body>", "<body>text</body><frameset>", "<body>text</body>"]
    assert screen_batch((pages, CAP)) == [("", False), ("", True), ("md:<body>text</body>", True)]
    assert fake.calls == ["<body>text</body>"]
