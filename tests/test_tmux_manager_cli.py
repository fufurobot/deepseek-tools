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

    def test_empty_dsl_raises(self, monkeypatch, tmp_path):
        path = tmp_path / "t.dsl"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="Empty DSL input"):
            _run(monkeypatch, [str(path)])
