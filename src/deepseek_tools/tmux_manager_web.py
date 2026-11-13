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


SECTIONS = "\n# ---\n"


def _frontend(name):
    """Lower-case the UI frontend name (``dsl`` passes through)."""
    return name.lower()


def dsl_from(name, text, from_format=None):
    """Convert any UI input into DSL text (``dsl`` passes through)."""
    if from_format is None:
        from_format = "dsl" if name == "dsl" else name
    return tm.to_dsl(text, from_format)


def dsl_sections(text):
    """Split DSL text into the per-table sections joined by to_dsl."""
    return [chunk for chunk in text.split(SECTIONS) if chunk.strip()]


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
        run_pipeline(st, _frontend(frontend), text)


def run_pipeline(st, frontend, source, session="table_session",
                 edits=None):
    """Convert + parse + render, then show every output once.

    ``edits`` is a list of replacement rows per table (from the size
    editor) that is applied before the sizes are resolved again."""
    try:
        dsl = dsl_from(frontend, source, "dsl" if frontend == "dsl" else None)
        tables = dsl_sections(dsl)
        infos = docs(dsl)
        for i, rows in (edits or {}).items():
            infos[i]["rows"] = rows
            tables[i] = tm.dsl_serialize(infos[i])
        dsl = SECTIONS.join(tables)
        tables = [tm.parse_dsl(chunk) for chunk in tables]
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
    if st.checkbox("Pixel boxes"):
        st.code(boxes or "(empty)", language="text")
    st.download_button("Download setup.sh", script, "setup.sh",
                       "text/x-shellscript")
    if st.checkbox("Edit sizes (px/em = cells, % = relative, '-' = rest)"):
        rows = size_editor(st, tables[0])
        st.caption("Edited table (feed this back as the DSL input or rerun "
                   "Convert with the edits applied):")
        st.code(tm.dsl_serialize({**tables[0], "rows": rows}), "text")


UNITS = ["none", "percentage", "pixels", "flexible (-)"]


def _hint(st, key, hint, label):
    """One size editor: unit select + number ('-' disables the number)."""
    st.markdown(f"**{label}**")
    current = "none" if hint is None else ("flexible (-)" if hint[1] == "-"
                                           else "percentage" if hint[0] == "%"
                                           else "pixels")
    unit = st.selectbox(label, UNITS, index=UNITS.index(current),
                        key=f"{key}_unit")
    number = st.number_input("value", min_value=0, step=1,
                             value=0 if hint is None or hint[1] == "-"
                             else int(hint[1]), key=f"{key}_num")
    if unit == "none":
        return None
    if unit.startswith("flexible"):
        return ("px", "-")
    return tm._dsl_unit_to_hint(unit, int(number))


def size_editor(st, info):
    """Edit every cell's span and sizes; returns the new rows."""
    rows = []
    for r, row in enumerate(info["rows"]):
        for c, (rs, cs, text, w_hint, h_hint) in enumerate(row):
            st.caption(f"cell ({r},{c}) {text!r}")
            w_col, h_col, r_col, c_col = st.columns(4)
            with w_col:
                w = _hint(st, f"w_{r}_{c}", w_hint, f"width {r},{c}")
            with h_col:
                h = _hint(st, f"h_{r}_{c}", h_hint, f"height {r},{c}")
            with r_col:
                rs = st.number_input("rowspan", 1, 20, rs, key=f"rs_{r}_{c}")
            with c_col:
                cs = st.number_input("colspan", 1, 20, cs, key=f"cs_{r}_{c}")
            rows.append((rs, cs, text, w, h))
    # Cells are edited in row-major order and the table here is one row of
    # cells on the common path; #TODO handle multi-row tables (needs a row
    # breakbox in the editor).
    return [tuple(rows)]
