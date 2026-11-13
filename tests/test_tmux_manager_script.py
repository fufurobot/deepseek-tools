"""Tests for deepseek_tools.tmux_manager: generate_tmux_script."""

from __future__ import annotations

from deepseek_tools import tmux_manager as tm


def _one_cell_table(text):
    return {"rows": [[(1, 1, text, None, None)]],
            "row_weights": [1.0], "col_weights": [1.0]}


class TestGenerateTmuxScript:
    def test_single_table(self):
        script = tm.generate_tmux_script([_one_cell_table("ls")],
                                         session_name="mysess")
        assert script.splitlines() == [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            "tmux new-session -d -s mysess",
            "",
            "tmux rename-window -t mysess:0 table_1",
            'P0=$(tmux list-panes -t mysess:table_1 -F "#{pane_id}" | head -n 1)',
            "# (0,0) 1x1",
            'tmux send-keys -t "${P0}" ls C-m',
            "",
            "tmux attach-session -t mysess",
        ]

    def test_no_attach(self):
        script = tm.generate_tmux_script([], session_name="s", attach=False)
        assert "attach-session" not in script
        assert script.splitlines()[:4] == [
            "#!/usr/bin/env bash", "set -euo pipefail", "",
            "tmux new-session -d -s s",
        ]

    def test_multiple_tables_use_new_window_and_labels(self):
        tables = [
            _one_cell_table("one"),
            {"rows": [], "row_weights": [], "col_weights": []},
        ]
        lines = tm.generate_tmux_script(tables, attach=False).splitlines()
        assert "tmux rename-window -t table_session:0 table_1" in lines
        assert "tmux new-window -t table_session -n table_2" in lines
        assert "# (empty table)" in lines
        assert sum("P0=$(tmux list-panes" in line for line in lines) == 1

    def test_skips_empty_text_and_quotes_arguments(self):
        tables = [{
            "rows": [[(1, 1, "   ", None, None), (1, 1, "it's here", None, None)]],
            "row_weights": [1.0],
            "col_weights": [1.0, 1.0],
        }]
        script = tm.generate_tmux_script(tables, session_name="a b", attach=False)
        assert script.count("send-keys") == 1
        assert """'it'"'"'s here' C-m""" in script
        assert "tmux new-session -d -s 'a b'" in script
        # The whole "session:window" target is quoted as one word.
        assert "tmux list-panes -t 'a b:table_1'" in script

    def test_split_commands_before_send_keys(self):
        tables = [{
            "rows": [[(1, 1, "top", None, None), (1, 1, "bottom", None, None)]],
            "row_weights": [1.0],
            "col_weights": [1.0, 1.0],
        }]
        lines = tm.generate_tmux_script(tables, attach=False).splitlines()
        pane_idx = next(i for i, l in enumerate(lines) if "list-panes" in l)
        split_idx = next(i for i, l in enumerate(lines) if "split-window" in l)
        send_idx = next(i for i, l in enumerate(lines) if "send-keys" in l)
        assert pane_idx < split_idx < send_idx
        assert lines[split_idx] == (
            'P1=$(tmux split-window -h -p 50 -t "${P0}" -P -F "#{pane_id}")'
        )
        assert lines[send_idx + 1] == "# (0,1) 1x1"

    def test_multiline_text_sent_as_single_key(self):
        script = tm.generate_tmux_script([_one_cell_table("a\nb")], attach=False)
        assert "'a\nb' C-m" in script
