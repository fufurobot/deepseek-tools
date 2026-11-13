"""Tests for deepseek_tools.tmux_manager: the pandoc subprocess wrapper and
Pandoc inline/block text extraction."""

from __future__ import annotations

import subprocess

import pytest

from deepseek_tools import tmux_manager as tm


def _str(text):
    return {"t": "Str", "c": text}


def _space():
    return {"t": "Space"}


def _para(*inlines):
    return {"t": "Para", "c": list(inlines)}


class TestPandocJson:
    def test_runs_pandoc_and_returns_ast(self, monkeypatch):
        calls = {}

        class Proc:
            stdout = b'{"pandoc-api-version": [1], "blocks": []}'

        def fake_run(cmd, **kwargs):
            calls["cmd"] = cmd
            calls["input"] = kwargs["input"]
            return Proc()

        monkeypatch.setattr(tm.subprocess, "run", fake_run)
        ast = tm.pandoc_json("| a |", "rst")

        assert ast == {"pandoc-api-version": [1], "blocks": []}
        assert calls["cmd"] == [
            "pandoc", "--from", "rst", "--to", "json", "--output", "-", "-",
        ]
        assert calls["input"] == b"| a |"

    def test_propagates_pandoc_failure(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            raise subprocess.CalledProcessError(1, cmd)

        monkeypatch.setattr(tm.subprocess, "run", fake_run)
        with pytest.raises(subprocess.CalledProcessError):
            tm.pandoc_json("bad", "html")


class TestCollectInlines:
    def test_non_list_returns_empty(self):
        assert tm.collect_inlines(None) == ""
        assert tm.collect_inlines("Str") == ""

    def test_skips_non_dict_items(self):
        assert tm.collect_inlines(["x", _str("ok")]) == "ok"

    def test_simple_kinds(self):
        inlines = [
            _str("Hello"),
            _space(),
            {"t": "Code", "c": ["", "ls", []]},
            {"t": "SoftBreak"},
            {"t": "LineBreak"},
            _str("world"),
        ]
        assert tm.collect_inlines(inlines) == "Hello ls\n\nworld"

    def test_code_with_short_payload(self):
        assert tm.collect_inlines([{"t": "Code", "c": ["only"]}]) == ""

    @pytest.mark.parametrize(
        "kind", ["Emph", "Strong", "Strikeout", "Superscript",
                 "Subscript", "SmallCaps"]
    )
    def test_formatting_wrappers(self, kind):
        node = {"t": kind, "c": [_str("bold")]}
        assert tm.collect_inlines([node]) == "bold"

    @pytest.mark.parametrize("kind", ["Link", "Quoted", "Cite", "Span"])
    def test_second_element_wrappers(self, kind):
        node = {"t": kind, "c": [["", [], []], [_str("inner")]]}
        assert tm.collect_inlines([node]) == "inner"

    def test_image_is_dropped(self):
        img = {"t": "Image", "c": [["", [], []], [], ["", "alt"]]}
        assert tm.collect_inlines([img, _str("x")]) == "x"

    def test_math(self):
        node = {"t": "Math", "c": [{"t": "InlineMath"}, "x^2"]}
        assert tm.collect_inlines([node]) == "x^2"

    def test_math_with_short_payload(self):
        assert tm.collect_inlines([{"t": "Math", "c": []}]) == ""

    def test_unknown_kind_is_ignored(self):
        node = {"t": "RawInline", "c": ["html", "<br>"]}
        assert tm.collect_inlines([node]) == ""


class TestCellTextFromBlocks:
    def test_non_list_returns_empty(self):
        assert tm.cell_text_from_blocks(None) == ""

    def test_joins_plain_and_para(self):
        blocks = [
            _para(_str("one")),
            {"t": "CodeBlock", "c": ["", "x"]},
            _para(_str("two")),
            "junk",
        ]
        assert tm.cell_text_from_blocks(blocks) == "one\ntwo"

    def test_strips_surrounding_whitespace(self):
        blocks = [_para(_str("  padded  "))]
        assert tm.cell_text_from_blocks(blocks) == "padded"

    def test_empty_list(self):
        assert tm.cell_text_from_blocks([]) == ""


def test_pandoc_json_rejects_invalid_json(monkeypatch):
    class Proc:
        stdout = b"not json"

    monkeypatch.setattr(tm.subprocess, "run", lambda cmd, **kw: Proc())
    with pytest.raises(ValueError):
        tm.pandoc_json("x", "html")
