"""Tests for deepseek_tools.tmux_manager: grid construction and weight
computation for pandoc-derived tables."""

from __future__ import annotations

from deepseek_tools import tmux_manager as tm


class TestBuildGrid:
    def test_simple_grid(self):
        rows = [
            [(1, 1, "a", None, None), (1, 1, "b", None, None)],
            [(1, 1, "c", None, None), (1, 1, "d", None, None)],
        ]
        grid, cells, max_row, max_col = tm._build_grid(rows)
        assert grid == {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
        assert cells[3]["text"] == "d"
        assert (max_row, max_col) == (2, 2)

    def test_spans_occupy_grid_and_push_cells(self):
        rows = [
            [(2, 1, "tall", None, None), (1, 2, "wide", None, None)],
            [(1, 1, "x", None, None)],
        ]
        grid, cells, max_row, max_col = tm._build_grid(rows)
        assert grid[(0, 0)] == 0
        assert grid[(1, 0)] == 0
        assert grid[(0, 1)] == 1
        assert grid[(0, 2)] == 1
        assert grid[(1, 1)] == 2
        assert cells[2] == {
            "row": 1, "col": 1, "rowspan": 1, "colspan": 1,
            "text": "x", "w_hint": None, "h_hint": None,
        }
        # max_col is the column cursor after the last cell; because the row-1
        # cell starts at column 1, the cursor ends at 3.
        assert (max_row, max_col) == (2, 3)

    def test_empty(self):
        assert tm._build_grid([]) == ({}, {}, 0, 0)


class TestFillWeights:
    def test_empty(self):
        assert tm.fill_weights([]) == []

    def test_none_entries_get_default(self):
        assert tm.fill_weights([None, None], default=2.5) == [2.5, 2.5]

    def test_all_present(self):
        assert tm.fill_weights([("px", 3.0), ("px", 7.0)]) == [3.0, 7.0]

    def test_missing_px_filled_with_average(self):
        assert tm.fill_weights([("px", 10.0), None, ("px", 20.0)]) == [
            10.0, 15.0, 20.0
        ]

    def test_missing_percent_filled_remainder(self):
        assert tm.fill_weights([("%", 20.0), None, None]) == [20.0, 40.0, 40.0]

    def test_percent_fill_average_when_no_room_left(self):
        # total (100) is not below 100, so the fill falls back to the average
        # of the present hints rather than sharing the remainder.
        assert tm.fill_weights([("%", 100.0), None]) == [100.0, 100.0]
        assert tm.fill_weights([("%", 100.0), None, None]) == [
            100.0, 100.0, 100.0
        ]

    def test_over_budget_percent_falls_back_to_average(self):
        # 120% is already over budget, so the remainder branch is not used.
        assert tm.fill_weights([("%", 120.0), None, None]) == [
            120.0, 120.0, 120.0
        ]

    def test_mixed_units_fall_back_to_default(self):
        assert tm.fill_weights([("px", 10.0), ("%", 20.0), None]) == [
            1.0, 1.0, 1.0
        ]

    def test_custom_default_applies_to_mixed_units(self):
        assert tm.fill_weights([("px", 10.0), ("%", 20.0)], default=7.0) == [
            7.0, 7.0
        ]


class TestComputeWeights:
    def test_colspec_widths_take_priority(self):
        info = {
            "col_widths": [0.25, 0.75],
            "rows": [[(1, 1, "a", ("px", 5.0), ("px", 5.0)),
                      (1, 1, "b", None, None)]],
        }
        row_w, col_w = tm.compute_weights(info)
        assert col_w == [25.0, 75.0]
        assert row_w == [5.0]

    def test_cell_hints_used_without_colspec(self):
        info = {
            "col_widths": None,
            "rows": [
                [(1, 1, "a", ("px", 10.0), ("px", 2.0)),
                 (1, 1, "b", None, None)],
                [(1, 1, "c", ("px", 30.0), None),
                 (1, 1, "d", ("px", 50.0), None)],
            ],
        }
        row_w, col_w = tm.compute_weights(info)
        # The first span-1 hint for a track wins; later ones are ignored, so
        # column 1 keeps "b"'s hint rather than "c"'s.
        assert col_w == [10.0, 50.0]
        # Only "a" constrains its row; the missing row-2 hint is filled with
        # the average of the present hints (2).
        assert row_w == [2.0, 2.0]

    def test_spanning_cells_do_not_donate_hints(self):
        info = {
            "col_widths": None,
            "rows": [[(2, 1, "tall", None, ("px", 9.0)),
                      (1, 2, "wide", ("px", 40.0), None)],
                     [(1, 1, "x", None, None)]],
        }
        row_w, col_w = tm.compute_weights(info)
        assert row_w == [1.0, 1.0]
        assert col_w == [1.0, 1.0, 1.0]

    def test_colspec_shorter_than_grid(self):
        info = {
            "col_widths": [0.5],
            "rows": [[(1, 1, "a", None, None), (1, 1, "b", None, None)]],
        }
        _, col_w = tm.compute_weights(info)
        assert col_w == [50.0, 50.0]
