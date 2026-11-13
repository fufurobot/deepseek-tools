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

    def test_spanning_percentage_keeps_one_share_per_track(self):
        # A colspan-2 percentage is one hint over two tracks: its share is
        # identical on both tracks; the unconstrained track takes the rest.
        hint = [(0, 2, ("%", 40.0)), (1, 1, None), (2, 1, None)]
        assert tm.resolve_sizes(hint, 1000, "w") == [200.0, 200.0, 600.0]

    def test_over_committed_percentages_raise(self):
        with pytest.raises(tm.LayoutError, match="percentages need 120"):
            tm.resolve_sizes(_h(("%", 60), ("%", 60)), 100, "width")

    def test_unknown_extent_is_provisional(self):
        assert tm.resolve_sizes(_h(("%", 50), ("%", 50)), None, "w") == [
            0.0, 0.0
        ]

    def test_zero_percent_track_is_allowed(self):
        assert tm.resolve_sizes(_h(("%", 0), None), 10, "w") == [0.0, 10.0]
