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
            "col_sizes": [10.0, 10.0],
            "row_sizes": [10.0, 10.0],
        }
        # Boxes are sorted by (x, y, w, h), so the left column comes first.
        assert tm.render_boxes(info).splitlines() == [
            "0 0 10 10 " + tm.encode_word("a"),
            "0 10 10 10 " + tm.encode_word("c"),
            "10 0 10 10 " + tm.encode_word("b"),
            "10 10 10 10 " + tm.encode_word("d"),
        ]

    @pytest.mark.xfail(reason="render_boxes sizing rework", strict=False)
    def test_percentages_fill_the_declared_extent(self):
        info = {
            "rows": [[(1, 1, "a", ("%", 25.0), ("%", 20.0)),
                      (1, 1, "b", None, None)]],
            "col_sizes": [1.0, 1.0],
            "row_sizes": [1.0],
            "_W": 400,
            "_H": 50,
        }
        # 25% of 400 = 100 cells, 20% of 50 = 10 cells; the rest is split.
        assert tm.render_boxes(info).splitlines() == [
            "0 0 100 10 " + tm.encode_word("a"),
            "100 0 300 10 " + tm.encode_word("b"),
        ]

    def test_spans_cover_their_tracks(self):
        info = {
            "rows": [
                [(1, 1, "a", None, None), (2, 1, "tall", None, None)],
                [(1, 2, "wide", None, None)],
            ],
            "col_sizes": [4.0, 6.0],
            "row_sizes": [3.0, 7.0],
        }
        boxes = {line.split()[-1]: line for line in
                 tm.render_boxes(info).splitlines()}
        assert boxes[tm.encode_word("tall")] == (
            f"4 0 6 10 {tm.encode_word('tall')}"
        )
        assert boxes[tm.encode_word("wide")] == (
            f"0 3 10 7 {tm.encode_word('wide')}"
        )

    @pytest.mark.xfail(reason="render_boxes sizing rework", strict=False)
    def test_no_declared_extent_uses_provisional_sizes(self):
        info = {"rows": [[(1, 1, "", ("px", 12.0), ("px", 3.0)),
                          (1, 1, "", None, None)]]}
        assert tm.render_boxes(info).splitlines() == [
            "0 0 12 3 ", "12 0 1 1 ",
        ]


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

    def test_requires_begin_table(self):
        # Whatever follows "begin" must literally be "table".
        assert tm.is_dsl("800 600\nbegin tr\n") is False
        assert tm.is_dsl("800 600\nbegin table\n") is True
