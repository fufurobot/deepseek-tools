"""Tests for deepseek_tools.tmux_manager: weight selection and the recursive
tmux split layout built from a table grid."""

from __future__ import annotations

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

    def test_percentage_clamped(self):
        _, _, cmds = tm.build_table_layout({
            "rows": [[(1, 1, "", None, None), (1, 1, "", None, None)]],
            "row_weights": [1.0],
            "col_weights": [1000.0, 0.0],
        })
        # A zero-weighted right pane still keeps at least 1%.
        assert cmds[-1].endswith('-p 1 -t "${P0}" -P -F "#{pane_id}")')
