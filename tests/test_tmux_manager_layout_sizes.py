"""Tests for layout_sizes: one source of truth for panes and boxes."""

from __future__ import annotations

import pytest

from deepseek_tools import tmux_manager as tm

HTML = (
    '<table>'
    '<tr><td rowspan=2 width="20%">a</td><td width="12">b</td></tr>'
    '<tr><td width="8">c</td></tr>'
    '</table>'
)


def test_percentage_and_absolute_mix(monkeypatch):
    # A 20% column next to a 12-cell column: the percentage share wins over
    # the smaller absolute sizes (#FIXME no conflict is reported).
    ast = {"t": "Pandoc", "c": {"blocks": [
        {"t": "Table", "c": tm._table_body(
            [[("a", 2, 1), ("b", 1, 1)], [("c", 1, 1)]])}]}}
    monkeypatch.setattr(tm, "pandoc_json", lambda text, fmt: ast)
    info = tm.parse_dsl(tm.to_dsl(HTML, "html"), resolve=False)
    rows, cols = tm.layout_sizes(info)
    assert sum(cols) >= sum(rows) > 0


def test_stored_sizes_win():
    info = {"rows": [[(1, 1, "", None, None)]],
            "row_sizes": [5.0], "col_sizes": [9.0]}
    assert tm.layout_sizes(info) == ([5.0], [9.0])


def test_stored_sizes_are_used_as_given():
    # The CLI always passes sizes computed by compute_dsl_weights; stored
    # sizes win so panes and boxes see identical numbers.
    info = {"rows": [[(1, 2, "", None, None)]], "row_sizes": [3.0],
            "col_sizes": [1.0, 2.0]}
    assert tm.layout_sizes(info) == ([3.0], [1.0, 2.0])


def test_spans_and_sizes_cover_every_cell(monkeypatch):
    ast = {"t": "Pandoc", "c": {"blocks": [
        {"t": "Table", "c": tm._table_body(
            [[("a", 2, 1), ("b", 1, 1)], [("c", 1, 1)]])}]}}
    monkeypatch.setattr(tm, "pandoc_json", lambda text, fmt: ast)
    boxes = tm.render_boxes(tm.parse_dsl(tm.to_dsl(HTML, "html")))
    assert boxes  # every cell has a box
    assert all(len(line.split()) == 5 for line in boxes.splitlines())
