"""Tests for deepseek_tools.tmux_manager: whole-table extraction from pandoc
ASTs (3.x and legacy shapes) and merging HTML width/height hints."""

from __future__ import annotations

import pytest

from deepseek_tools import tmux_manager as tm


def _str(text):
    return {"t": "Str", "c": text}


def _para(*inlines):
    return {"t": "Para", "c": list(inlines)}


def _cell(inlines, rowspan=1, colspan=1):
    return {"t": "Cell", "c": [["", [], []], {"t": "AlignDefault"}, rowspan,
                               colspan, [_para(*inlines)]]}


def _row(*cells):
    return {"t": "Row", "c": ["", list(cells)]}


def _table(head_rows=(), body_rows=(), foot_rows=()):
    """Build a pandoc 3.x Table constructor payload."""
    return [
        ["", [], []],
        ["", []],
        [],
        {"t": "TableHead", "c": ["", list(head_rows)]},
        [{"t": "TableBody", "c": ["", 0, list(body_rows), []]}],
        {"t": "TableHead", "c": ["", list(foot_rows)]},
    ]


class TestExtractTableFull:
    def test_pandoc3_table(self):
        table_c = _table(
            head_rows=[_row(_cell([_str("H")]))],
            body_rows=[_row(_cell([_str("b1")]), _cell([_str("b2")]))],
            foot_rows=[_row(_cell([_str("F")]))],
        )
        info = tm.extract_table_full(table_c)
        assert info["rows"] == [
            [(1, 1, "H", None, None)],
            [(1, 1, "b1", None, None), (1, 1, "b2", None, None)],
            [(1, 1, "F", None, None)],
        ]
        assert info["col_widths"] is None

    def test_pandoc3_table_skips_malformed_body(self):
        table_c = _table(body_rows=[_row(_cell([_str("b")]))])
        table_c[4] = [{"t": "TableBody", "c": ["", 0]},
                      {"t": "TableBody", "c": ["", 0, table_c[4][0]["c"][2], []]}]
        # The short TableBody is skipped, the well-formed one is kept.
        assert tm.extract_table_full(table_c)["rows"] == [
            [(1, 1, "b", None, None)]
        ]

    def test_pandoc3_table_without_bodies(self):
        table_c = _table()
        table_c[4] = None
        assert tm.extract_table_full(table_c) is None

    def test_pandoc3_table_without_rows_returns_none(self):
        assert tm.extract_table_full(_table()) is None

    def test_legacy_table(self):
        table_c = [
            {"t": "Attr", "c": ["", [], []]},
            {"t": "Caption", "c": [None, []]},
            {"t": "ColSpec", "c": []},
            [_cell([_str("h1")])["c"], _cell([_str("h2")])["c"]],
            [[_cell([_str("a")])["c"]], "junk-row"],
        ]
        info = tm.extract_table_full(table_c)
        # The trailing "junk-row" string is iterated character by character,
        # so each character is handed to extract_cell_info and ignored.
        rows = info["rows"]
        assert rows[:2] == [
            [(1, 1, "h1", None, None), (1, 1, "h2", None, None)],
            [(1, 1, "a", None, None)],
        ]
        assert len(rows) == 3
        assert rows[2] == [(1, 1, "", None, None)] * len("junk-row")

    def test_legacy_table_without_headers(self):
        table_c = [0, 0, 0, None, [[_cell([_str("a")])["c"]]]]
        assert tm.extract_table_full(table_c)["rows"] == [
            [(1, 1, "a", None, None)]
        ]

    def test_not_a_list(self):
        assert tm.extract_table_full("nope") is None


class TestExtractTables:
    def test_walks_nested_ast(self):
        table = {"t": "Table", "c": _table(body_rows=[_row(_cell([_str("a")]))])}
        ast = {"t": "Pandoc", "c": {"blocks": [{"t": "Div", "c": [[], [table]]}]}}
        tables = tm.extract_tables(ast)
        assert len(tables) == 1
        assert tables[0]["rows"][0][0][2] == "a"

    def test_empty_table_is_skipped(self):
        assert tm.extract_tables({"c": [{"t": "Table", "c": _table()}]}) == []


class TestMergeHtmlHints:
    def _tables(self):
        return [{"rows": [[(1, 1, "a", None, None), (1, 2, "b", None, None)]]}]

    def test_merges_hints(self):
        tables = self._tables()
        html = ('<table><tr><td width="10%">a</td>'
                '<td colspan="2" style="height: 3em">b</td></tr></table>')
        out = tm.merge_html_hints(tables, html)
        assert out[0]["rows"][0] == [
            (1, 1, "a", ("%", 10.0), None),
            (1, 2, "b", None, ("em", 3.0)),
        ]

    def test_existing_hints_win(self):
        tables = [{"rows": [[(1, 1, "a", ("px", 1.0), ("px", 2.0))]]}]
        out = tm.merge_html_hints(tables, '<td width="10%">a</td>')
        assert out[0]["rows"][0][0] == (1, 1, "a", ("px", 1.0), ("px", 2.0))

    @pytest.mark.parametrize("html", [None, "", "<p>no cells</p>"])
    def test_no_source_or_no_cells(self, html):
        tables = self._tables()
        assert tm.merge_html_hints(tables, html) is tables

    def test_span_mismatch_aborts_merge(self):
        tables = self._tables()
        out = tm.merge_html_hints(tables, "<tr><td>x</td><td colspan=5>y</td></tr>")
        assert out[0]["rows"][0][0][3] is None

    def test_fewer_html_cells_than_table_cells(self):
        tables = self._tables()
        html = '<td width="1px">a</td><td colspan=2>b</td>'
        out = tm.merge_html_hints(tables, html)
        assert out[0]["rows"][0] == [
            (1, 1, "a", ("px", 1.0), None),
            (1, 2, "b", None, None),
        ]

    def test_parser_error_returns_tables(self, monkeypatch):
        def boom(self, html_source):
            raise RuntimeError("nope")

        monkeypatch.setattr(tm._HTMLCellAttrsExtractor, "feed", boom)
        tables = self._tables()
        assert tm.merge_html_hints(tables, "<td>x</td>") is tables
