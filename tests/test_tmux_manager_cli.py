"""Tests for deepseek_tools.tmux_manager.main (the command line interface)."""

from __future__ import annotations

import io

import pytest

from deepseek_tools import tmux_manager as tm

DSL = "10 10\nbegin table\nbegin tr\nbegin td\ntext eA==\nend td\nend tr\n"


def _run(monkeypatch, argv, stdin=None):
    monkeypatch.setattr(tm.sys, "argv", ["tmux_manager", *argv])
    if stdin is not None:
        monkeypatch.setattr(tm.sys, "stdin", io.StringIO(stdin))
    tm.main()


class TestMainDslInput:
    def test_dsl_file_to_stdout(self, monkeypatch, tmp_path, capsys):
        path = tmp_path / "t.dsl"
        path.write_text(DSL, encoding="utf-8")
        _run(monkeypatch, [str(path), "--session", "s", "--no-attach"])
        out = capsys.readouterr().out
        assert out.startswith("#!/usr/bin/env bash")
        assert 'tmux send-keys -t "${P0}" x C-m' in out
        assert out.endswith("\n")
        assert "attach-session" not in out

    def test_dsl_detected_by_content_for_unknown_extension(
        self, monkeypatch, tmp_path, capsys
    ):
        path = tmp_path / "t.txt"
        path.write_text(DSL, encoding="utf-8")
        _run(monkeypatch, [str(path)])
        assert "tmux attach-session -t table_session" in capsys.readouterr().out

    def test_stdin_dsl_autodetect_boxes(self, monkeypatch, capsys):
        _run(monkeypatch, ["-", "--emit", "boxes"], stdin=DSL)
        out = capsys.readouterr().out
        assert out.splitlines()[0] == "0 0 10 10 " + tm.encode_word("x")

    def test_stdin_requires_format(self, monkeypatch):
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, ["-"], stdin="<p>hi</p>")
        assert exc.value.code == 2

    def test_output_file_written_and_executable(
        self, monkeypatch, tmp_path, capsys
    ):
        src = tmp_path / "t.dsl"
        src.write_text(DSL, encoding="utf-8")
        dst = tmp_path / "out.sh"
        _run(monkeypatch, [str(src), "-o", str(dst)])
        assert capsys.readouterr().out == ""
        assert dst.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash")
        assert dst.stat().st_mode & 0o777 == 0o755

    def test_cannot_infer_format(self, monkeypatch, tmp_path):
        path = tmp_path / "t.xyz"
        path.write_text("just text", encoding="utf-8")
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, [str(path)])
        assert exc.value.code == 2

    def test_empty_dsl_exits(self, monkeypatch, tmp_path):
        path = tmp_path / "t.dsl"
        path.write_text("", encoding="utf-8")
        # No table survives in the pipeline: main reports and exits.
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, [str(path)])
        assert "No tables found" in str(exc.value.code)


class TestMainPandocInput:
    def _table_ast(self):
        cell = {"t": "Cell", "c": [["", [], []], {"t": "AlignDefault"}, 1, 1,
                                  [{"t": "Para", "c": [{"t": "Str", "c": "x"}]}]]}
        row = {"t": "Row", "c": ["", [cell]]}
        table = {"t": "Table", "c": [
            ["", [], []], ["", []], [],
            {"t": "TableHead", "c": ["", []]},
            [{"t": "TableBody", "c": ["", 0, [row], []]}],
            {"t": "TableHead", "c": ["", []]},
        ]}
        return {"t": "Pandoc", "c": {"blocks": [table]}}

    def test_html_merges_hints(self, monkeypatch, tmp_path, capsys):
        path = tmp_path / "t.html"
        path.write_text('<table><tr><td width="10%">x</td></tr></table>',
                        encoding="utf-8")
        monkeypatch.setattr(tm, "pandoc_json",
                            lambda text, fmt: self._table_ast())
        _run(monkeypatch, [str(path), "--emit", "boxes"])
        # The HTML width hint is merged in. #FIXME a relative hint without
        # a declared character width resolves to a provisional 0 cells, so
        # the box only has its height.
        assert capsys.readouterr().out.splitlines()[0] == (
            "0 0 0 1 " + tm.encode_word("x"))

    def test_no_debug_dump_by_default(self, monkeypatch, tmp_path, capsys):
        path = tmp_path / "t.rst"
        path.write_text("ignored", encoding="utf-8")
        monkeypatch.setattr(tm, "pandoc_json", lambda text, fmt: self._table_ast())
        _run(monkeypatch, [str(path)])
        assert capsys.readouterr().err == ""

    def test_html_without_tables_exits(self, monkeypatch, tmp_path):
        path = tmp_path / "t.html"
        path.write_text("<p>no tables</p>", encoding="utf-8")
        monkeypatch.setattr(tm, "pandoc_json", lambda text, fmt: {"blocks": []})
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, [str(path)])
        assert "No tables found in the input." in str(exc.value.code)

    def test_multiple_tables_boxes_headers(self, monkeypatch, tmp_path, capsys):
        ast = self._table_ast()
        table = ast["c"]["blocks"][0]
        monkeypatch.setattr(
            tm, "pandoc_json", lambda text, fmt: {"c": [table, table]}
        )
        path = tmp_path / "t.rst"
        path.write_text("ignored", encoding="utf-8")
        _run(monkeypatch, [str(path), "--emit", "boxes"])
        out = capsys.readouterr().out
        assert out.startswith("# Table 1\n")
        assert "\n# Table 2\n" in out

    def test_single_table_has_no_boxes_header(self, monkeypatch, tmp_path, capsys):
        path = tmp_path / "t.html"
        path.write_text("ignored", encoding="utf-8")
        monkeypatch.setattr(tm, "pandoc_json", lambda text, fmt: self._table_ast())
        _run(monkeypatch, [str(path), "--emit", "boxes"])
        assert "# Table" not in capsys.readouterr().out

    def test_from_format_override(self, monkeypatch, tmp_path):
        path = tmp_path / "t.dsl"
        path.write_text(DSL, encoding="utf-8")
        seen = {}

        def fake_pandoc_json(text, from_format):
            seen["fmt"] = from_format
            return {"c": []}

        monkeypatch.setattr(tm, "pandoc_json", fake_pandoc_json)
        with pytest.raises(SystemExit):
            _run(monkeypatch, [str(path), "--from", "html"])
        assert seen["fmt"] == "html"

    def test_rst_extension_uses_pandoc(self, monkeypatch, tmp_path):
        path = tmp_path / "t.rest"
        path.write_text("ignored", encoding="utf-8")
        seen = {}

        def fake_pandoc_json(text, from_format):
            seen["fmt"] = from_format
            return self._table_ast()

        monkeypatch.setattr(tm, "pandoc_json", fake_pandoc_json)
        _run(monkeypatch, [str(path), "--no-attach"])
        assert seen["fmt"] == "rst"


class TestMainStdin:
    def test_stdin_with_explicit_from(self, monkeypatch, capsys):
        _run(monkeypatch, ["-", "--from", "dsl", "-o", "-"],
             stdin=DSL.replace("begin table", "begin table "))  # not auto-detected
        assert 'tmux send-keys -t "${P0}" x C-m' in capsys.readouterr().out


class TestWebCommand:
    def test_web_launches_streamlit_with_port(self, monkeypatch):
        calls = {}
        monkeypatch.setattr(tm.subprocess, "call",
                            lambda cmd: calls.setdefault("cmd", cmd) and 0)
        _run(monkeypatch, ["web", "--port", "9000"])
        cmd = calls["cmd"]
        assert cmd[1:3] == ["-m", "streamlit"]
        assert cmd[3] == "run"
        assert cmd[4].endswith("tmux_manager_web.py")
        assert cmd[cmd.index("--server.port") + 1] == "9000"

    def test_missing_input_is_usage_error(self, monkeypatch):
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, [])
        assert exc.value.code == 2

    def test_recipe_command_exits(self, monkeypatch, capsys):
        import io

        _run(monkeypatch, ["--recipe"], stdin="")
        assert "begin table" in capsys.readouterr().out
