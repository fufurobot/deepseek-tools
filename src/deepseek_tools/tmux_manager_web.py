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




def build(st):
    """The whole Streamlit page. ``st`` is injected for testability."""
    st.title("tmux_manager")
    st.caption("HTML / RST / DSL tables -> DSL -> tmux panes. Absolute sizes "
               "are character cells; percentages scale with the table.")
    frontend = st.radio("Input", ["DSL", "HTML", "RST"], horizontal=True)
    text = st.text_area("Table source", DEFAULT_DSL, height=180)
    with st.expander("Quick start"):
        st.markdown("`begin tr` starts a row, `begin td` a cell; "
                    "`prop width 30 percentage`, `prop height 8 pixels`, "
                    "`prop width - pixels` (flexible), `text <base64>`.")
    if st.button("Convert", type="primary"):
        run_pipeline(st, frontend.lower(), text)


def run_pipeline(st, frontend, source, session="table_session"):
    """Convert + parse + render, then show every output once."""
    try:
        dsl = dsl_from(frontend, source, "dsl" if frontend == "dsl" else None)
        tables = docs(dsl)
        for info in tables:
            info.update(zip(("row_sizes", "col_sizes"),
                            tm.compute_dsl_weights(info, info["_W"], info["_H"])))
        boxes = preview(tables)
        script = render(tables, session=session)
    except (ValueError, tm.LayoutError) as exc:
        st.error(f"{type(exc).__name__}: {exc}")
        return
    st.subheader("DSL")
    st.code(dsl, language="text")
    st.subheader("Panes")
    st.code(script, language="bash")
    if st.toggle("Pixel boxes"):
        st.code(boxes or "(empty)", language="text")
    st.download_button("Download setup.sh", script, "setup.sh",
                       "text/x-shellscript")
