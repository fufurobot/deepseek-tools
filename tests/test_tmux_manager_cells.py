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
