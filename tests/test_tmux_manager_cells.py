"""Tests for deepseek_tools.tmux_manager: pandoc Cell / Row / ColSpec
extraction."""

from __future__ import annotations

import pytest

from deepseek_tools import tmux_manager as tm


def _str(text):
    return {"t": "Str", "c": text}


def _para(*inlines):
    return {"t": "Para", "c": list(inlines)}


def _cell(inlines, rowspan=1, colspan=1, attr=None):
    """Build a pandoc 3.x Cell: [attr, alignment, rowspan, colspan, blocks]."""
    if attr is None:
        attr = ["", [], []]
    return {"t": "Cell", "c": [attr, {"t": "AlignDefault"}, rowspan, colspan,
                               [_para(*inlines)]]}


def _raw(cell):
    """Return the constructor payload of a Cell built by :func:`_cell`."""
    return cell["c"]


class TestUnwrapHelpers:
    def test_unwrap_ctor(self):
        assert tm.unwrap_ctor({"t": "Cell", "c": [1, 2]}, "Cell") == [1, 2]

    def test_unwrap_ctor_other(self):
        node = [1, 2]
        assert tm.unwrap_ctor(node, "Cell") is node
        other = {"t": "Row", "c": []}
        assert tm.unwrap_ctor(other, "Cell") is other

    def test_cell_attrs_dict(self):
        attr = ["id", ["cls"], [["width", "5px"], ["bad"], "x", ["h", "1em"]]]
        assert tm._cell_attrs_dict(attr) == {"width": "5px", "h": "1em"}

    @pytest.mark.parametrize("attr", [None, [], ["", []], ["", [], "nope"]])
    def test_cell_attrs_dict_invalid(self, attr):
        assert tm._cell_attrs_dict(attr) == {}


class TestCellExtraction:
    def test_extract_cell_info(self):
        cell = _cell([_str("hi")], rowspan=2, colspan=3,
                     attr=["", [], [["width", "10%"], ["height", "20px"]]])
        assert tm.extract_cell_info(cell) == (
            2, 3, "hi", ("%", 10.0), ("px", 20.0)
        )

    def test_extract_cell_info_unwraps_constructor(self):
        cell = {"t": "Cell", "c": _raw(_cell([_str("hi")]))}
        assert tm.extract_cell_info(cell)[2] == "hi"

    @pytest.mark.parametrize("cell", [None, [], [1, 2]])
    def test_extract_cell_info_invalid(self, cell):
        assert tm.extract_cell_info(cell) == (1, 1, "", None, None)

    @pytest.mark.parametrize("rs,cs", [(0, 1), (1, -2), ("2", 1), (None, None)])
    def test_extract_cell_info_invalid_spans(self, rs, cs):
        cell = ["", {}, rs, cs, []]
        assert tm.extract_cell_info(cell)[:2] == (1, 1)


def _row(*cells):
    return {"t": "Row", "c": ["", list(cells)]}


class TestRowExtraction:
    def test_extract_row_with_spans(self):
        row = _row(_cell([_str("a")]), _cell([_str("b")], colspan=2))
        assert tm.extract_row_with_spans(row) == [
            (1, 1, "a", None, None),
            (1, 2, "b", None, None),
        ]

    def test_extract_row_with_spans_unwrap(self):
        row = {"t": "Row", "c": _row(_cell([_str("a")]))["c"]}
        assert tm.extract_row_with_spans(row)[0][2] == "a"

    @pytest.mark.parametrize("row", [None, [], [""], ["", "notalist"]])
    def test_extract_row_with_spans_invalid(self, row):
        assert tm.extract_row_with_spans(row) == []

    def test_head_rows_raw(self):
        head = {"t": "TableHead", "c": ["", ["r1", "r2"]]}
        assert tm._head_rows_raw(head) == ["r1", "r2"]

    @pytest.mark.parametrize("head", [None, [], [""], ["", "nope"]])
    def test_head_rows_raw_invalid(self, head):
        assert tm._head_rows_raw(head) == []


class TestColspecWidths:
    def test_explicit_widths(self):
        cw = {"t": "ColWidth", "c": 0.25}
        default = {"t": "ColWidthDefault"}
        table_c = [
            ["", [], []],
            ["", []],
            [["AlignDefault", cw], ["AlignDefault", default]],
        ]
        assert tm.extract_colspec_widths(table_c) == [0.25, None]

    def test_no_explicit_width_returns_none(self):
        table_c = [
            ["", [], []],
            ["", []],
            [["AlignDefault", {"t": "ColWidthDefault"}]],
        ]
        assert tm.extract_colspec_widths(table_c) is None

    def test_colwidth_non_numeric(self):
        table_c = [0, 0, [["AlignDefault", {"t": "ColWidth", "c": "x"}]]]
        assert tm.extract_colspec_widths(table_c) is None

    @pytest.mark.parametrize("table_c", [None, [], [1, 2], [1, 2, "bad"]])
    def test_invalid(self, table_c):
        assert tm.extract_colspec_widths(table_c) is None

    def test_short_spec_entry(self):
        table_c = [0, 0, [["AlignDefault"]], 0]
        assert tm.extract_colspec_widths(table_c) is None
