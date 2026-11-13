"""Tests for deepseek_tools.tmux_manager.resolve_sizes.

Covers the relative (percentage) rules; absolute and mixed sizes come next."""

from __future__ import annotations

import pytest

from deepseek_tools import tmux_manager as tm


def _h(*hints):
    """One track per hint."""
    return [(i, 1, hint) for i, hint in enumerate(hints)]


class TestRelativeMode:
    def test_no_hints_split_evenly(self):
        assert tm.resolve_sizes(_h(None, None, None, None), 800, "width") == [
            200.0, 200.0, 200.0, 200.0
        ]

    def test_rounding_keeps_the_total(self):
        sizes = tm.resolve_sizes(_h(None, None, None), 100, "width")
        assert sizes == [33.0, 33.0, 34.0]
        assert sum(sizes) == 100

    def test_percentages_are_fractions_of_the_total(self):
        assert tm.resolve_sizes(_h(("%", 50), ("%", 25), None), 800, "w") == [
            400.0, 200.0, 200.0
        ]

    def test_spanning_percentage_counts_once(self):
        # A colspan-2 40% hint is 400 cells: the same 40% sits on both of its
        # tracks, so the fixed part is 800 and the free track takes 200.
        hint = [(0, 2, ("%", 40.0)), (1, 1, None), (2, 1, None)]
        assert tm.resolve_sizes(hint, 1000, "w") == [400.0, 400.0, 200.0]

    def test_over_committed_percentages_raise(self):
        with pytest.raises(tm.LayoutError, match="percentages need 120"):
            tm.resolve_sizes(_h(("%", 60), ("%", 60)), 100, "width")

    def test_unknown_extent_is_provisional(self):
        assert tm.resolve_sizes(_h(("%", 50), ("%", 50)), None, "w") == [
            0.0, 0.0
        ]

    def test_zero_percent_track_is_allowed(self):
        assert tm.resolve_sizes(_h(("%", 0), None), 10, "w") == [0.0, 10.0]


class TestAbsoluteMode:
    def test_absolute_sizes_are_character_counts(self):
        # 30 and 50 cells are literal counts; the flexible track takes the
        # declared 100 - 80 = 20 cells that are left.
        hint = _h(("px", 30.0), ("px", 50.0), ("px", "-"))
        assert tm.resolve_sizes(hint, 100, "w") == [30.0, 50.0, 20.0]

    def test_spanning_absolute_divides_across_tracks(self):
        hint = [(0, 2, ("px", 20.0)), (2, 1, ("px", "-"))]
        assert tm.resolve_sizes(hint, 30, "w") == [10.0, 10.0, 10.0]

    def test_flexible_track_takes_the_remainder(self):
        assert tm.resolve_sizes(_h(("px", 30.0), ("px", "-")), 100, "w") == [
            30.0, 70.0
        ]

    def test_conflicting_absolute_hints_raise(self):
        hint = [(0, 2, ("px", 20.0)), (1, 1, ("px", 5.0))]
        with pytest.raises(tm.LayoutError, match="disagree"):
            tm.resolve_sizes(hint, 100, "height")

    def test_overfull_absolute_raises(self):
        hint = _h(("px", 10.0), ("px", 10.0), ("px", "-"))
        with pytest.raises(tm.LayoutError, match="need 20 cells"):
            tm.resolve_sizes(hint, 5, "w")

    def test_only_one_flexible_track(self):
        with pytest.raises(tm.LayoutError, match="only one flexible"):
            tm.resolve_sizes(_h(("px", "-"), ("px", "-")), 10, "width")

    def test_unassigned_absolute_track_raises(self):
        with pytest.raises(tm.LayoutError, match="flexible track"):
            tm.resolve_sizes(_h(("px", 10.0), None), 100, "width")

    def test_absolute_hints_must_cover_the_table(self):
        # 10 + 10 cells in a table declared 100 cells wide leaves 80 cells
        # unassigned: reported instead of silently stretched.
        with pytest.raises(tm.LayoutError, match="provides 100"):
            tm.resolve_sizes(_h(("px", 10.0), ("px", 10.0)), 100, "width")
