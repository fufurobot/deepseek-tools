"""Tests for deepseek_tools.tmux_manager: HTML length / style hint parsing
and the HTML cell attribute extractor."""

from __future__ import annotations

import pytest

from deepseek_tools import tmux_manager as tm


class TestParseLength:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("10", ("px", 10.0)),
            ("10px", ("px", 10.0)),
            ("50%", ("%", 50.0)),
            (" 2.5 em ", ("em", 2.5)),
            ("3rem", ("rem", 3.0)),
            ("12pt", ("pt", 12.0)),
            ("-4px", ("px", -4.0)),
        ],
    )
    def test_valid(self, value, expected):
        assert tm.parse_length(value) == expected

    @pytest.mark.parametrize("value", ["", None, 10, "abc", "10 vh", "auto"])
    def test_invalid(self, value):
        assert tm.parse_length(value) is None


class TestStyleHelpers:
    def test_style_decls(self):
        assert tm._style_decls("Width: 10px; color:red;;junk") == {
            "width": "10px",
            "color": "red",
        }

    @pytest.mark.parametrize("style", [None, "", 5])
    def test_style_decls_falsy(self, style):
        assert tm._style_decls(style) == {}

    def test_extract_size_from_attrs(self):
        attrs = {"width": "30%", "style": "height: 4em; width: 99px"}
        assert tm._extract_size_from_attrs(attrs) == (("%", 30.0), ("em", 4.0))

    def test_extract_size_falls_back_to_style(self):
        assert tm._extract_size_from_attrs({"style": "width: 5px"}) == (
            ("px", 5.0),
            None,
        )

    def test_extract_size_none(self):
        assert tm._extract_size_from_attrs({}) == (None, None)


class TestHtmlCellAttrsExtractor:
    def test_records_td_and_th(self):
        parser = tm._HTMLCellAttrsExtractor()
        parser.feed(
            '<table><tr><td rowspan="2">a</td><th colspan="3" '
            'width="20%">b</th></tr></table>'
        )
        parser.close()
        assert parser.cells == [
            {"rowspan": 2, "colspan": 1, "w_hint": None, "h_hint": None},
            {"rowspan": 1, "colspan": 3,
             "w_hint": ("%", 20.0), "h_hint": None},
        ]

    def test_invalid_spans_default_to_one(self):
        parser = tm._HTMLCellAttrsExtractor()
        parser.feed('<td rowspan="x" colspan="">t</td>')
        parser.close()
        assert parser.cells[0]["rowspan"] == 1
        assert parser.cells[0]["colspan"] == 1

    def test_self_closing_tag_is_recorded(self):
        parser = tm._HTMLCellAttrsExtractor()
        parser.feed('<td style="height: 2em"/>')
        parser.close()
        assert parser.cells == [
            {"rowspan": 1, "colspan": 1, "w_hint": None,
             "h_hint": ("em", 2.0)}
        ]
