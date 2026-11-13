"""Smoke test for the Streamlit UI (needs Streamlit's AppTest)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "src" / "deepseek_tools"
SCRIPT = f"""
import sys
sys.path.insert(0, {str(APP.parent)!r})
from deepseek_tools import tmux_manager_web as web
web.run()
"""


def _app():
    app = AppTest.from_string(SCRIPT)
    app.run(timeout=30)
    return app


def test_page_renders():
    app = _app()
    assert not app.exception
    assert app.title[0].value == "tmux_manager"
    assert app.radio[0].value == "DSL"


def test_convert_shows_dsl_script_and_boxes():
    app = _app()
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert "begin table" in app.code[0].value
    assert "tmux send-keys" in app.code[1].value
    assert app.toggle[0].value is False
    app.toggle[0].set_value(True).run(timeout=30)
    assert " YQ==" in app.code[2].value


def test_bad_dsl_shows_error():
    app = _app()
    app.text_area[0].set_value("nonsense").run(timeout=30)
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert "ValueError" in app.error[0].value
