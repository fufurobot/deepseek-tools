"""Tests for the DSL serializers: table_info -> DSL text -> table_info."""

from __future__ import annotations

from deepseek_tools import tmux_manager as tm

TABLE = {
    "rows": [
        [(1, 1, "a", ("%", 50.0), None), (1, 1, "", None, None)],
        [(2, 1, "it's", ("px", 12.0), ("px", "-")), (1, 2, "ü", None, None)],
    ],
    "_W": 80,
    "_H": 24,
}


def test_dsl_serialize_round_trips():
    text = tm.dsl_serialize(TABLE)
    back = tm.parse_dsl(text, resolve=False)
    assert back["rows"] == TABLE["rows"]
    assert (back["_W"], back["_H"]) == (80, 24)


def test_dsl_serialize_is_line_based_and_encodes_text():
    lines = tm.dsl_serialize(TABLE).splitlines()
    assert lines[0] == "80 24"
    assert "begin table" in lines
    assert f"text {tm.encode_word('it\'s')}" in lines
    assert "prop width 12 pixels" in lines
    assert "prop height - pixels" in lines
    assert any(l.startswith("prop rowspan 2") for l in lines)


def test_parse_dsl_resolve_flag_skips_weights():
    text = "10 10\nbegin tr\nbegin td\nend td\nend tr\n"
    assert tm.parse_dsl(text)["col_weights"] == [10.0]
    assert "col_sizes" not in tm.parse_dsl(text, resolve=False)
