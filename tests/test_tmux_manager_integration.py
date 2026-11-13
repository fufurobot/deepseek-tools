"""End-to-end tests for deepseek_tools.tmux_manager that run a real pandoc on
an HTML table document."""

from __future__ import annotations

import shutil

import pytest

from deepseek_tools import tmux_manager as tm

# The HTML table that motivated the tool (a 2x2 grid with spans and width
# hints); kept inline because the repository ignores "data" directories.
HTML = """\
<!DOCTYPE html>
<html>
<head>
<title>Page Title</title>
</head>
<body>

<table>
<tr><td rowspan=2>firefox</td><td>btop</td></tr>
<tr><td>watch -n 0.2 nvidia-smi</td></tr>
<tr><td width=5 colspan=2>pnpx @deepseek-ai/dsh web</td></tr>
<tr><td width=5 colspan=2>uvx --from-requirements requirements.txt jupyterlab --ip 0.0.0.0</td></tr>
</table>

</body>
</html>
"""

pytestmark = pytest.mark.skipif(
    shutil.which("pandoc") is None, reason="pandoc is not installed"
)


def _tables():
    html = HTML
    tables = tm.extract_tables(tm.pandoc_json(html, "html"))
    assert len(tables) == 1
    return tm.merge_html_hints(tables, html)


def test_extracts_all_cell_texts():
    rows = _tables()[0]["rows"]
    assert [[cell[2] for cell in row] for row in rows] == [
        ["firefox", "btop"],
        ["watch -n 0.2 nvidia-smi"],
        ["pnpx @deepseek-ai/dsh web"],
        ["uvx --from-requirements requirements.txt jupyterlab --ip 0.0.0.0"],
    ]


def test_preserves_spans_and_width_hints():
    rows = _tables()[0]["rows"]
    assert rows[0][0][:2] == (2, 1)
    assert rows[2][0][1] == 2
    assert rows[3][0][1] == 2
    assert rows[2][0][3] == ("px", 5.0)


def test_generated_script_contains_every_command():
    script = tm.generate_tmux_script(_tables(), session_name="my-tmux")
    for text in ("firefox", "btop", "watch -n 0.2 nvidia-smi",
                 "pnpx @deepseek-ai/dsh web"):
        assert "send-keys" in script
        assert text in script
    assert "tmux attach-session -t my-tmux" in script
