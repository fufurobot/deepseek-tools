"""Tests for deepseek_tools.tmux_manager: render_boxes and is_dsl."""

from __future__ import annotations

import pytest

from deepseek_tools import tmux_manager as tm


class TestRenderBoxes:
    def test_uniform_grid(self):
        info = {
            "rows": [
                [(1, 1, "a", None, None), (1, 1, "b", None, None)],
                [(1, 1, "c", None, None), (1, 1, "d", None, None)],
            ],
            "row_weights": [10.0, 10.0],
            "col_weights": [10.0, 10.0],
        }
        # Boxes are sorted by (x, y, w, h), so the left column comes first.
        assert tm.render_boxes(info).splitlines() == [
            "0 0 10 10 " + tm.encode_word("a"),
            "0 10 10 10 " + tm.encode_word("c"),
            "10 0 10 10 " + tm.encode_word("b"),
            "10 10 10 10 " + tm.encode_word("d"),
        ]

    def test_spans_and_rounding(self):
        info = {
            "rows": [
                [(1, 1, "a", None, None), (2, 1, "tall", None, None)],
                [(1, 2, "wide", None, None)],
            ],
            "row_weights": [3.3, 6.7],
            "col_weights": [4.4, 5.6],
        }
        lines = tm.render_boxes(info).splitlines()
        # Edge coordinates are rounded (4.4 -> 4, 3.3 -> 3, 10.0 -> 10).
        boxes = {line.split()[-1]: line for line in lines}
        assert boxes[tm.encode_word("tall")] == (
            f"4 0 6 10 {tm.encode_word('tall')}"
        )
        assert boxes[tm.encode_word("wide")] == (
            f"0 3 10 7 {tm.encode_word('wide')}"
        )

    def test_non_integer_weights_are_rounded(self):
        info = {
            "rows": [[(1, 1, "", None, None), (1, 1, "", None, None)]],
            "row_weights": [1.0],
            "col_weights": [1.4, 1.4],
        }
        lines = tm.render_boxes(info).splitlines()
        assert lines[0].split()[:4] == ["0", "0", "1", "1"]
        assert lines[1].split()[:4] == ["1", "0", "2", "1"]

    def test_uses_computed_weights_without_precomputed(self):
        info = {"rows": [[(1, 1, "", None, None), (1, 1, "", None, None)]]}
        assert tm.render_boxes(info).splitlines() == ["0 0 1 1 ", "1 0 1 1 "]


class TestIsDsl:
    @pytest.mark.parametrize(
        "text",
        [
            "800 600\nbegin table\nend table\n",
            "\n\n  \n800 600\nbegin table",
            "1 2\n\nbegin table",
        ],
    )
    def test_true(self, text):
        assert tm.is_dsl(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "\n \n",
            "800\nbegin table\n",
            "800 600 700\nbegin table\n",
            "wx 600\nbegin table\n",
            "800 600\n",
            "800 600\n\n",
        ],
    )
    def test_false(self, text):
        assert tm.is_dsl(text) is False

    @pytest.mark.parametrize(
        "text",
        [
            # The first statement after "W H" must be "begin" with any tag;
            # the tag is not required to be "table".
            "800 600\nbegin tr\n",
            "800 600\nbegin table\n",
        ],
    )
    def test_true_for_any_begin_after_header(self, text):
        assert tm.is_dsl(text) is True
