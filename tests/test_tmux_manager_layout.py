"""Tests for deepseek_tools.tmux_manager: weight selection and the recursive
tmux split layout built from a table grid."""

from __future__ import annotations

import pytest

from deepseek_tools import tmux_manager as tm


class TestPickWeights:
    def test_precomputed_weights_win(self):
        info = {"rows": [], "row_weights": [1], "col_weights": [2]}
        assert tm._pick_weights(info) == ([1], [2])

    def test_computes_when_missing(self):
        info = {"rows": [[(1, 1, "", None, None)]]}
        assert tm._pick_weights(info) == ([1.0], [1.0])


class TestBuildTableLayout:
    def test_empty_table(self):
        assert tm.build_table_layout({"rows": []}) == ({}, {}, [])

    def test_single_cell_has_no_splits(self):
        _, pane_map, cmds = tm.build_table_layout(
            {"rows": [[(1, 1, "", None, None)]], "row_weights": [1.0],
             "col_weights": [1.0]}
        )
        assert cmds == []
        assert pane_map == {0: "P0"}

    def test_two_by_two_splits_rows_then_columns(self):
        cells_info, pane_map, cmds = tm.build_table_layout({
            "rows": [
                [(1, 1, "", None, None), (1, 1, "", None, None)],
                [(1, 1, "", None, None), (1, 1, "", None, None)],
            ],
            "row_weights": [3.0, 1.0],
            "col_weights": [1.0, 1.0],
        })
        # Rows are split first; P1 is the new bottom half. The top half has
        # no clean vertical cut, so its two cells share P0; the bottom half
        # splits into P3 (bottom-left, 1:3 weights) and a further column.
        assert cmds == [
            'P1=$(tmux split-window -v -p 25 -t "${P0}" -P -F "#{pane_id}")',
            'P2=$(tmux split-window -h -p 50 -t "${P0}" -P -F "#{pane_id}")',
            'P3=$(tmux split-window -h -p 50 -t "${P1}" -P -F "#{pane_id}")',
        ]
        assert sorted(cells_info) == [0, 1, 2, 3]
        assert pane_map == {0: "P0", 1: "P2", 2: "P1", 3: "P3"}

    def test_spanning_cell_forces_later_split(self):
        # firefox spans two rows on the right; the left column splits in two.
        _, pane_map, cmds = tm.build_table_layout({
            "rows": [
                [(1, 1, "watch", None, None), (2, 1, "firefox", None, None)],
                [(1, 1, "btop", None, None)],
            ],
            "row_weights": [1.0, 1.0],
            "col_weights": [1.0, 1.0],
        })
        assert cmds == [
            'P1=$(tmux split-window -h -p 50 -t "${P0}" -P -F "#{pane_id}")',
            'P2=$(tmux split-window -v -p 50 -t "${P0}" -P -F "#{pane_id}")',
        ]
        assert pane_map == {0: "P0", 1: "P1", 2: "P2"}

    def test_unsplittable_interleaving_shares_top_pane(self):
        # Row 0: A(colspan 2) B; row 1: C D E. The top row can be cut after
        # column 2 (B|E in P1), but the bottom row has overlapping spans left
        # (C D in P0), so A/C share P0 and D splits into P3/P4.
        _, pane_map, cmds = tm.build_table_layout({
            "rows": [
                [(1, 2, "A", None, None), (1, 1, "B", None, None)],
                [(1, 1, "C", None, None), (1, 1, "D", None, None),
                 (1, 1, "E", None, None)],
            ],
            "row_weights": [1.0, 1.0],
            "col_weights": [1.0, 1.0, 1.0],
        })
        assert cmds[0] == (
            'P1=$(tmux split-window -v -p 50 -t "${P0}" -P -F "#{pane_id}")'
        )
        # The bottom-thirds split recurses on the left half first (P2), then
        # splits the remaining right side (P3) once more.
        assert cmds[1].startswith('P2=$(tmux split-window -h -p 33 -t "${P0}"')
        assert cmds[2].startswith('P3=$(tmux split-window -h -p 67 -t "${P1}"')
        assert cmds[3].startswith('P4=$(tmux split-window -h -p 50 -t "${P3}"')
        assert pane_map == {0: "P0", 1: "P2", 2: "P1", 3: "P3", 4: "P4"}

    def test_zero_sized_side_is_not_split(self):
        # A 0-cell track has nothing to show: splitting it would create a
        # 1% pane, so the split is skipped (#FIXME cells share the pane).
        _, _, cmds = tm.build_table_layout({
            "rows": [[(1, 1, "", None, None), (1, 1, "", None, None)]],
            "row_weights": [1.0],
            "col_weights": [1000.0, 0.0],
        })
        assert cmds == []

    def test_zero_weights_are_not_split(self):
        _, pane_map, cmds = tm.build_table_layout({
            "rows": [
                [(1, 1, "", None, None), (1, 1, "", None, None)],
                [(1, 1, "", None, None), (1, 1, "", None, None)],
            ],
            "row_weights": [0.0, 0.0],
            "col_weights": [0.0, 0.0],
        })
        assert cmds == []
        assert set(pane_map.values()) == {"P0"}

    def test_split_with_an_empty_side_is_skipped(self):
        # Rows 0-1 are one rowspan=2 cell and rows 2-3 are rowspan=3 cells,
        # so the cut after row 1 leaves nothing above/below another cut.
        cells_info = tm._build_grid([
            [(1, 1, "a", None, None), (4, 1, "b", None, None)],
            [(1, 1, "c", None, None)],
            [(1, 1, "d", None, None)],
            [(1, 1, "e", None, None)],
        ])[1]
        pane_map = {}
        cmds = tm._split_layout(
            "P0", (0, 0, 4, 2), list(cells_info), cells_info,
            [1.0] * 4, [1.0, 1.0], [0], pane_map,
        )
        # Left column splits first, then its lower part again.
        assert cmds[0].startswith('P1=$(tmux split-window -h -p 50 -t "${P0}"')
        assert sorted(pane_map) == [0, 1, 2, 3, 4]
        assert set(pane_map.values()) == {"P0", "P1", "P2", "P3", "P4"}

    def test_no_cut_cells_share_the_pane(self):
        # A single L-shaped arrangement that has no clean row or column cut.
        cells_info = tm._build_grid([
            [(1, 2, "a", None, None)],
            [(1, 1, "b", None, None), (1, 1, "c", None, None)],
            [(1, 1, "d", None, None)],
        ])[1]
        # Force the "no cut" path with a single-cell id list per branch.
        pane_map = {}
        assert tm._split_layout(
            "P9", (0, 0, 1, 1), [999], cells_info, [1.0], [1.0], [0], pane_map,
        ) == []
        assert pane_map == {999: "P9"}


class TestSplitPercent:
    def test_share_of_resolved_sizes(self):
        assert tm._split_percent(30.0, 70.0, "col", 1) == (70, True)
        assert tm._split_percent(1.0, 2.0, "row", 1) == (67, True)

    def test_clamped_to_one_and_ninety_nine(self):
        assert tm._split_percent(1000.0, 1.0, "col", 1) == (1, True)
        assert tm._split_percent(1.0, 1000.0, "col", 1) == (99, True)

    def test_empty_side_skips_the_split(self):
        assert tm._split_percent(0.0, 0.0, "col", 1) == (50, False)
        assert tm._split_percent(10.0, 0.0, "row", 1) == (50, False)
