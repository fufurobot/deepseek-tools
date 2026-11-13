"""Tests for deepseek_tools.tmux_manager_web helpers (no Streamlit needed)."""

from __future__ import annotations

import pytest

from deepseek_tools import tmux_manager as tm
from deepseek_tools import tmux_manager_web as web


def test_default_dsl_parses_and_renders():
    tables = web.docs(web.DEFAULT_DSL)
    assert [c[2] for c in tables[0]["rows"][0]] == ["btop", "watch -n 2 nvidia-smi"]
    script = web.render(tables, session="s", attach=False)
    assert "tmux send-keys -t \"${P0}\" btop C-m" in script
    assert "watch -n 2 nvidia-smi" in script


def test_dsl_input_passes_through():
    assert web.dsl_from("dsl", web.DEFAULT_DSL) == web.DEFAULT_DSL


def test_preview_has_one_line_per_cell():
    lines = web.preview(web.docs(web.DEFAULT_DSL)).splitlines()
    assert len(lines) == 2
    assert all(len(line.split()) == 5 for line in lines)


def test_unknown_frontend_raises():
    with pytest.raises(ValueError):
        web.dsl_from("docx", "x")
