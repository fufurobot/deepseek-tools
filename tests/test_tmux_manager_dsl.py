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

    def test_end_without_tag_is_unknown(self):
        with pytest.raises(ValueError, match="Unknown end tag"):
            tm.parse_dsl("10 10\nend\n")

    @pytest.mark.parametrize(
        "text,message",
        [
            ("", "Empty DSL input"),
            ("\n \n", "Empty DSL input"),
            ("800\n", "Expected 'W H'"),
            ("800 600 1\n", "Expected 'W H'"),
            ("wx 600\n", "two integers"),
            ("800 600\nbegin\n", "Malformed begin"),
            ("800 600\nbegin body\n", "Unknown tag"),
            ("800 600\nprop width 1 pixels\n", "prop outside cell"),
            ("800 600\nbegin tr\nbegin td\nprop width\n", "Malformed prop"),
            ("800 600\nbegin tr\nbegin td\nprop nope 1 pixels\n", "Unknown prop"),
            ("800 600\nbegin tr\nbegin td\ntext\n", "Malformed text"),
            ("800 600\nbegin tr\ntext eA==\n", "text outside cell"),
            ("800 600\nbegin tr\nbegin td\nend td\nend td\n",
             "end td without open cell"),
            ("800 600\nbegin tr\nbegin td\nbogus\n", "Unknown command"),
            ("800 600\nbegin td\nprop width 5 metres\nend td\n",
             "Unknown DSL unit"),
            ("800 600\nbegin td\nprop width x pixels\nend td\n",
             "invalid literal"),
        ],
    )
    def test_errors(self, text, message):
        with pytest.raises(ValueError, match=message):
            tm.parse_dsl(text)


class TestComputeDslWeights:
    def _info(self, rows):
        return {"rows": rows}

    def test_unconstrained_split_evenly(self):
        rows = [[(1, 1, "", None, None), (1, 1, "", None, None)]]
        assert tm.compute_dsl_weights(self._info(rows), 100, 50) == (
            [50.0], [50.0, 50.0]
        )

    def test_percentage_hints_are_of_total(self):
        rows = [[(1, 1, "", ("%", 25.0), None), (1, 1, "", None, None)]]
        row_w, col_w = tm.compute_dsl_weights(self._info(rows), 400, 10)
        assert col_w == [100.0, 300.0]

    def test_colspan_hint_divided_across_tracks(self):
        rows = [[(1, 2, "", ("px", 100.0), None), (1, 1, "", None, None)],
                [(1, 1, "", None, None), (1, 1, "", None, None),
                 (1, 1, "", None, None)]]
        _, col_w = tm.compute_dsl_weights(self._info(rows), 400, 10)
        assert col_w == [50.0, 50.0, 300.0]

    def test_rowspan_hint_divided_across_tracks(self):
        rows = [[(1, 1, "", None, None), (2, 1, "", None, ("px", 100.0))],
                [(1, 1, "", None, None), (1, 1, "", None, None)],
                [(1, 1, "", None, None), (1, 1, "", None, None)]]
        row_w, _ = tm.compute_dsl_weights(self._info(rows), 10, 400)
        assert row_w == [50.0, 50.0, 300.0]

    def test_all_tracks_constrained(self):
        rows = [[(1, 1, "", ("px", 10.0), ("px", 5.0)),
                 (1, 1, "", ("px", 30.0), None)]]
        assert tm.compute_dsl_weights(self._info(rows), 100, 100) == (
            [5.0], [10.0, 30.0]
        )


def test_rowspan_and_colspan_props_are_applied():
    info = tm.parse_dsl(
        "10 10\nbegin tr\nbegin td\nprop rowspan 2 pixels\n"
        "prop colspan 3 pixels\n"
        "end td\nbegin td\nend td\nbegin td\nend td\nbegin td\nend td\n"
        "end tr\nend table\n"
    )
    assert info["rows"] == [[(2, 3, "", None, None), (1, 1, "", None, None),
                             (1, 1, "", None, None), (1, 1, "", None, None)]]
    assert info["col_weights"] == [10.0, 10.0, 10.0, 10.0]
    assert info["row_weights"] == [5.0, 5.0]
