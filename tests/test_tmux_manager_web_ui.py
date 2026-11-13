"""Smoke test for the Streamlit UI (needs Streamlit's AppTest)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from deepseek_tools import tmux_manager_web as web  # noqa: E402

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
    # The first render has no boxes: only the DSL and the script.
    assert len(app.code) == 2
    app.checkbox[0].set_value(True).run(timeout=30)
    app.button[0].click().run(timeout=30)
    assert " YnRvcA==" in app.code[2].value


def test_bad_dsl_shows_error():
    app = _app()
    app.text_area[0].set_value("nonsense").run(timeout=30)
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert "ValueError" in app.error[0].value


def test_default_page_offers_every_input():
    app = _app()
    assert [r.options for r in app.radio][0][:3] == ["DSL", "HTML", "RST"]
    assert len(app.button) == 1


def test_size_editor_round_trips_sizes():
    class Stub:
        """Minimal Streamlit recorder: answers like a fresh widget."""
        def __init__(self):
            self.widgets, self.elements = [], []

        def markdown(self, *a):
            self.elements.append(a)

        def caption(self, *a):
            self.elements.append(a)

        def columns(self, n):
            return [self] * n

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def selectbox(self, label, options, index=0, key=None):
            self.elements.append(("select", label, options[index]))
            return options[index]

        def number_input(self, label, min_value=0, max_value=None,
                         value=0, step=1, key=None):
            self.elements.append(("number", label, value))
            return value

    stub = Stub()
    info = web.docs(web.DEFAULT_DSL)[0]
    rows = web.size_editor(stub, info)
    assert rows == [tuple(r) for r in info["rows"]]
    labels = [e[1] for e in stub.elements if e[0] == "select"]
    assert "width 0,1" in labels


def test_size_editor_reports_edits():
    from deepseek_tools import tmux_manager_web as w

    class Stub:
        def markdown(self, *a):
            pass

        def caption(self, *a):
            pass

        def columns(self, n):
            return [self] * n

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def selectbox(self, label, options, index=0, key=None):
            return "flexible (-)" if label.startswith("width") else "percentage"

        def number_input(self, label, min_value=0, max_value=None,
                         value=0, step=1, key=None):
            return 12 if label == "value" else value

    rows = w.size_editor(Stub(), web.docs(web.DEFAULT_DSL)[0])
    assert rows == [(1, 1, "btop", ("px", "-"), ("%", 12.0)),
                    (1, 1, "watch -n 2 nvidia-smi", ("px", "-"),
                     ("%", 12.0))]
