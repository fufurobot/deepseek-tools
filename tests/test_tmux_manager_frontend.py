"""Tests for to_dsl: every frontend becomes DSL text before layout."""

from __future__ import annotations

import pytest

from deepseek_tools import tmux_manager as tm

HTML = (
    '<table>'
    '<tr><td width="20%">a</td><td width="12em">b</td></tr>'
    '<tr><td colspan="2" height="3px">c</td></tr>'
    '</table>'
)


def test_html_becomes_dsl_first(monkeypatch):
    ast = {"t": "Pandoc", "c": {"blocks": [
        {"t": "Table", "c": tm._table_body([
            [("a", 1, 1), ("b", 1, 1)], [("c", 1, 2)]])}]}}
    monkeypatch.setattr(tm, "pandoc_json", lambda text, fmt: ast)
    text = tm.to_dsl(HTML, "html")
    assert text.startswith("80 24\nbegin table\n")
    assert "prop width 20 percentage" in text
    assert "prop width 12 pixels" in text
    assert "prop colspan 2" in text
    back = tm.parse_dsl(text, resolve=False)
    assert [c[2] for row in back["rows"] for c in row] == ["a", "b", "c"]


def test_dsl_input_passes_through_unchanged():
    text = "80 24\nbegin tr\nbegin td\nend td\nend tr\nend table\n"
    assert tm.to_dsl(text, "dsl") == text


def test_rst_colspec_widths_become_percentages(monkeypatch):
    ast = {"t": "Pandoc", "c": {"blocks": [
        {"t": "Table", "c": tm._table_body([[("a", 1, 1), ("b", 1, 1)]],
                                           widths=[0.25, 0.75])}]}}
    monkeypatch.setattr(tm, "pandoc_json", lambda text, fmt: ast)
    text = tm.to_dsl("ignored", "rst")
    assert "prop width 25 percentage" in text
    assert "prop width 75 percentage" in text


def test_unknown_format_raises():
    with pytest.raises(ValueError):
        tm.to_dsl("x", "docx")
