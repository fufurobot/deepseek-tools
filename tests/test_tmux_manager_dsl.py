"""Tests for deepseek_tools.tmux_manager: Table Renderer DSL parsing, Base64
word encoding and DSL weight computation."""

from __future__ import annotations

import base64

import pytest

from deepseek_tools import tmux_manager as tm


class TestBase64Helpers:
    @pytest.mark.parametrize("text", ["", "hello", "héllo wörld", "a"])
    def test_roundtrip(self, text):
        assert tm.decode_word(tm.encode_word(text)) == text

    def test_encode_is_padded_urlsafe(self):
        assert tm.encode_word("hi?") == base64.urlsafe_b64encode(b"hi?").decode()

    def test_decode_adds_padding(self):
        assert tm.decode_word("aGk") == "hi"

    def test_decode_utf8(self):
        assert tm.decode_word("w6k") == "é"

    def test_decode_non_utf8_returns_input(self):
        # b"\xff" is valid Base64 but not valid UTF-8.
        assert base64.urlsafe_b64decode("_w==") == b"\xff"
        assert tm.decode_word("_w==") == "_w=="

    def test_decode_non_ascii_keeps_word_plus_padding(self):
        # Base64 padding is added before the ASCII encode fails, and the
        # padded word is what gets returned (a known quirk).
        assert tm.decode_word("héllo") == "héllo==="


class TestDslUnitToHint:
    def test_pixels(self):
        assert tm._dsl_unit_to_hint("pixels", "12") == ("px", 12.0)

    def test_percentage(self):
        assert tm._dsl_unit_to_hint("percentage", 50) == ("%", 50.0)

    def test_unknown_unit(self):
        with pytest.raises(ValueError):
            tm._dsl_unit_to_hint("metres", 1)


DSL_TABLE = """\
800 600
begin table
begin tr
begin td
prop width 50 percentage
text YQ==
prop height 40 percentage
end td
begin td
end td
end tr
begin tr
begin td
prop height 100 pixels
end td
end tr
end table
"""


class TestParseDsl:
    def test_parses_cells_and_hints(self):
        info = tm.parse_dsl(DSL_TABLE)
        assert info["rows"] == [
            [(1, 1, "a", ("%", 50.0), ("%", 40.0)), (1, 1, "", None, None)],
            [(1, 1, "", None, ("px", 100.0))],
        ]
        assert info["col_widths"] is None
        assert info["_W"] == 800
        assert info["_H"] == 600
        assert info["col_weights"] == [400.0, 400.0]
        # The later row height hint overwrites the earlier 40% one.
        assert info["row_weights"] == [240.0, 100.0]

    def test_leading_blank_lines_and_inline_statements(self):
        # Two statements on one line: only the first token is used as command.
        text = "\n  \n" + DSL_TABLE
        assert tm.parse_dsl(text)["_W"] == 800
        # "begin tr begin td end td end tr" is one "begin tr" statement, so
        # the row stays empty.
        assert tm.parse_dsl("10 10\nbegin tr begin td end td end tr\n")["rows"] == [
            []
        ]

    def test_default_props(self):
        info = tm.parse_dsl("10 10\nbegin tr\nbegin td\ntext eA==\nend td\nend tr\n")
        assert info["rows"] == [[(1, 1, "x", None, None)]]

    def test_end_table_and_extra_tokens(self):
        info = tm.parse_dsl(
            "10 10\nbegin table\nbegin tr\nbegin td\nprop colspan 2 extra\n"
            "end td\nend tr\nend table\n"
        )
        assert info["rows"] == [[(1, 2, "", None, None)]]
