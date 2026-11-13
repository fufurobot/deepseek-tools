"""Streamlit front end for :mod:`deepseek_tools.tmux_manager`.

Run it with ``python -m deepseek_tools.tmux_manager web`` (or
``streamlit run -m ...`` is not supported: use the CLI). Everything is built
on the same DSL pipeline as the CLI: every input becomes DSL text, the DSL is
parsed, then rendered into a tmux script or pixel boxes.

Only :func:`build` and :func:`run` need Streamlit; the helpers below are
plain functions so they can be unit tested without it.
"""

from __future__ import annotations

from deepseek_tools import tmux_manager as tm

DEFAULT_DSL = """\
80 24
begin table
begin tr
begin td
prop width 30 percentage
text {editor}
end td
begin td
prop height - pixels
text {terminal}
end td
end tr
end table
""".format(editor=tm.encode_word("btop"), terminal=tm.encode_word("watch -n 2 nvidia-smi"))


def dsl_from(name, text, from_format=None):
    """Convert any UI input into DSL text (``dsl`` passes through)."""
    if from_format is None:
        from_format = "dsl" if name == "dsl" else name
    return tm.to_dsl(text, from_format)


def docs(text):
    """Parse DSL text into table info dicts (no layout side effects)."""
    return [tm.parse_dsl(chunk, resolve=False)
            for chunk in text.split("\n# ---\n") if chunk.strip()]


def preview(tables):
    """The pixel boxes of every table (from the resolved cell sizes)."""
    chunks = []
    for i, info in enumerate(tables, start=1):
        if len(tables) > 1:
            chunks.append(f"# Table {i}")
        chunks.append(tm.render_boxes(info))
    return "\n".join(chunks)


def render(tables, session="table_session", attach=True):
    """The tmux Bash script for the parsed tables."""
    return tm.generate_tmux_script(tables, session_name=session, attach=attach)


def run(port=8501, **kwargs):
    """Start the Streamlit app."""
    import streamlit as st
    st.set_page_config(page_title="tmux_manager", layout="wide")
    build(st)
    st.stop()
