"""Tests for deepseek_tools.tmux_manager: Table Renderer DSL parsing, Base64
word encoding and DSL weight computation."""

from __future__ import annotations

import base64

import pytest

from deepseek_tools import tmux_manager as tm


class TestBase64Helpers:
    @pytest.mark.parametrize("text", ["", "hello", "héllo wörld", "a"])
    def test_roundtrip(self, text):
        assert tm.decode_word(tm.encode_word(text)) == text

    def test_encode_is_padded_urlsafe(self):
        assert tm.encode_word("hi?") == base64.urlsafe_b64encode(b"hi?").decode()

    def test_decode_adds_padding(self):
        assert tm.decode_word("aGk") == "hi"

    def test_decode_utf8(self):
        assert tm.decode_word("w6k") == "é"

    def test_decode_non_utf8_returns_input(self):
        # b"\xff" is valid Base64 but not valid UTF-8.
        assert base64.urlsafe_b64decode("_w==") == b"\xff"
        assert tm.decode_word("_w==") == "_w=="

    def test_decode_non_ascii_keeps_word_plus_padding(self):
        # Base64 padding is added before the ASCII encode fails, and the
        # padded word is what gets returned (a known quirk).
        assert tm.decode_word("héllo") == "héllo==="


class TestDslUnitToHint:
    def test_pixels(self):
        assert tm._dsl_unit_to_hint("pixels", "12") == ("px", 12.0)

    def test_percentage(self):
        assert tm._dsl_unit_to_hint("percentage", 50) == ("%", 50.0)

    def test_unknown_unit(self):
        with pytest.raises(ValueError):
            tm._dsl_unit_to_hint("metres", 1)
