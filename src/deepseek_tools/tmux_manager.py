#!/usr/bin/env python3
"""
tmux_manager.py - Turn HTML, reStructuredText or Table Renderer DSL tables
into a tmux Bash script (and a pixel-box rendering for validation).

Pipeline: every frontend is parsed into Table Renderer DSL text first
(:func:`to_dsl`), then ``parse_dsl`` -> :func:`layout_sizes` ->
``_split_layout``. Panes and pixel boxes both draw from the same integer
sizes returned by :func:`layout_sizes`, so they always agree.

Size units (``prop width``/``prop height`` in DSL, HTML width/height
attributes, pandoc column widths). Two kinds, both deterministic:

  * Relative - ``30 percentage`` / HTML ``width="30%"`` / pandoc column
    widths: a fraction of the whole axis (30 = 30% of W for a width, of H
    for a height). A percentage on a spanning cell is one hint: the same
    share sits on every track it covers. Unconstrained tracks split the
    remainder; rounding slack goes to the largest fractional parts, so the
    sizes always add up exactly. Totals over 100% raise ``LayoutError``.
  * Absolute - ``12 pixels``, ``3 em`` or HTML ``width="12"``/``"12px"``:
    12 character cells (1em == 1px == 1 cell), never scaled to the axis.
    ``-`` (or a negative value) marks the one *flexible* track per axis
    that absorbs the rest. Absolute tables shrink to fit their sizes; the
    declared W x H is an upper bound. Conflicting sizes on one track raise
    ``LayoutError``.

Mixing: a percentage next to absolute sizes wins: the table grows (or
shrinks) so the percentage track can hold its share, unless that would
drop an absolute track below its size. A percentage of the free space
next to absolute sizes caps the table's width instead.

HTML tables assume an 80x24 pane unless they carry only absolute sizes
(#TODO: a --width/--height option). Cells of a colspan cell are given the
full row width instead of a legacy ``width=5`` attribute (#TODO: override
per cell). A pane whose side has zero cells is not split: its cells share
the parent pane (#FIXME should become a report instead).

Usage:
    python -m deepseek_tools.tmux_manager input.html
    python -m deepseek_tools.tmux_manager input.rst -o setup.sh
    python -m deepseek_tools.tmux_manager input.dsl --emit boxes
    python -m deepseek_tools.tmux_manager web --port 8501   # Streamlit UI
    python -m deepseek_tools.tmux_manager --recipe           # sample DSL
    cat table.dsl | python -m deepseek_tools.tmux_manager - --from dsl
"""

import argparse
import base64
import json
import math
import re
import shlex
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path


# ======================================================================
# 1. Pandoc JSON helpers
# ======================================================================
def pandoc_json(text: str, from_format: str) -> dict:
    cmd = [
        "pandoc",
        "--from", from_format,
        "--to", "json",
        "--output", "-",
        "-",
    ]
    proc = subprocess.run(
        cmd,
        input=text.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return json.loads(proc.stdout.decode())


def collect_inlines(inlines: list) -> str:
    if not isinstance(inlines, list):
        return ""
    parts = []
    for inline in inlines:
        if not isinstance(inline, dict):
            continue
        t = inline.get("t")
        c = inline.get("c")
        if t == "Str":
            parts.append(c)
        elif t == "Code":
            parts.append(c[1] if isinstance(c, list) and len(c) > 1 else "")
        elif t == "Space":
            parts.append(" ")
        elif t in ("SoftBreak", "LineBreak"):
            parts.append("\n")
        elif t in ("Emph", "Strong", "Strikeout", "Superscript",
                   "Subscript", "SmallCaps"):
            parts.append(collect_inlines(c))
        elif t == "Link":
            parts.append(collect_inlines(c[1]))
        elif t == "Image":
            pass
        elif t == "Quoted":
            parts.append(collect_inlines(c[1]))
        elif t == "Cite":
            parts.append(collect_inlines(c[1]))
        elif t == "Span":
            parts.append(collect_inlines(c[1]))
        elif t == "Math":
            parts.append(c[1] if isinstance(c, list) and len(c) > 1 else "")
    return "".join(parts)


def cell_text_from_blocks(blocks: list) -> str:
    if not isinstance(blocks, list):
        return ""
    texts = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        bt = block.get("t")
        bc = block.get("c")
        if bt in ("Plain", "Para"):
            texts.append(collect_inlines(bc))
    return "\n".join(texts).strip()


# ======================================================================
# 2. Length / style parsing (HTML)
# ======================================================================
_LEN_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*(px|%|em|rem|pt)?\s*$")


def parse_length(value):
    if not value or not isinstance(value, str):
        return None
    m = _LEN_RE.match(value)
    if not m:
        return None
    try:
        num = float(m.group(1))
    except ValueError:
        return None
    unit = m.group(2) or "px"
    return (unit, num)


def _style_decls(style):
    if not style or not isinstance(style, str):
        return {}
    out = {}
    for decl in style.split(";"):
        if ":" not in decl:
            continue
        k, v = decl.split(":", 1)
        out[k.strip().lower()] = v.strip()
    return out


def _extract_size_from_attrs(attrs):
    w = parse_length(attrs.get("width"))
    h = parse_length(attrs.get("height"))
    style = _style_decls(attrs.get("style", ""))
    if w is None and "width" in style:
        w = parse_length(style["width"])
    if h is None and "height" in style:
        h = parse_length(style["height"])
    return (w, h)


class _HTMLCellAttrsExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.cells = []

    def _record(self, tag, attrs):
        if tag not in ("td", "th"):
            return
        d = dict(attrs)

        def _int(v, default=1):
            try:
                return int(v)
            except (TypeError, ValueError):
                return default

        w_hint, h_hint = _extract_size_from_attrs(d)
        # HTML width/height attributes are legacy pixels: a unitless number
        # in them is an absolute character count, while a unitless CSS
        # "width: 5" value is invalid and ignored. See README "Size units".
        for attr, get_hint in (("width", 0), ("height", 1)):
            value = (d.get(attr) or "").strip()
            hint = (w_hint, h_hint)[get_hint]
            if value.isdigit() and hint is not None:
                hint = ("px", float(int(value)))
            if attr == "width":
                w_hint = hint
            else:
                h_hint = hint
        self.cells.append({
            "rowspan": _int(d.get("rowspan"), 1),
            "colspan": _int(d.get("colspan"), 1),
            "w_hint": w_hint,
            "h_hint": h_hint,
        })

    def handle_starttag(self, tag, attrs):
        self._record(tag, attrs)

    def handle_startendtag(self, tag, attrs):
        self._record(tag, attrs)


# ======================================================================
# 2b. Size resolution: absolute pixels/em vs relative percentages
# ======================================================================
class LayoutError(ValueError):
    """Raised for conflicting or impossible size constraints."""


def _fmt(value):
    """Render a cell count for messages (12.0 -> '12', 2.5 -> '2.5')."""
    return f"{value:g}"


def layout_sizes(info):
    """Resolved integer sizes of both axes for a parsed table.

    Returns ``(row_sizes, col_sizes)``; stored sizes win (they already
    account for hints and the declared extent). Panes and pixel boxes both
    work from these numbers, so they always agree."""
    cols, rws = info.get("col_sizes"), info.get("row_sizes")
    if cols is None or rws is None:
        # The extent of an axis is the larger of the declared size and the
        # grid (a colspan can start past the last sized column).
        _, _, max_row, max_col = _build_grid(info["rows"])
        hints_c, hints_r = grid_size_hints(info["rows"])
        # An undeclared axis has no extent (None): absolute sizes then fix
        # it, and relative hints stay provisional.
        cols = resolve_sizes(hints_c, info.get("_W"), "width")
        rws = resolve_sizes(hints_r, info.get("_H"), "height")
    return rws, cols


def grid_size_hints(rows):
    """Turn table rows into per-axis ``(track, span, hint)`` lists.
    Percentages are divided across their span; absolute sizes are not.
    Empty tracks after the last cell are included so the axis covers them."""
    _, cells, max_row, max_col = _build_grid(rows)
    # Tracks are owned by the position after the last cell on the axis.
    widths = tuple("colspan" for _ in range(max_col))
    heights = tuple("rowspan" for _ in range(max_row))
    cols, rws = [], []
    for cell in cells.values():
        for out, index, key, axis in (
                (cols, "col", "w_hint", widths),
                (rws, "row", "h_hint", heights)):
            hint, span = cell[key], cell[axis[cell[index]]]
            if hint is not None and hint[0] == "%":
                hint = ("%", hint[1] / span)
            out.append((cell[index], span, hint))
    return cols, rws


def resolve_sizes(hints, total, axis):
    """Resolve one axis into absolute integer cell sizes summing to *total*.

    ``hints`` is ``[(track, span, hint)]`` for every cell of the grid,
    ``total`` the extent in cells (``None`` = provisional) and ``axis``
    names the axis for messages. Rules, in order:

    1. With no absolute hints the axis is *relative*: percentage tracks
       take ``total * pct / 100`` (a spanning percentage is already
       divided by its span) and unconstrained tracks share the remainder.
    2. An absolute ``px`` hint owns its span with ``value / span`` per
       track, fixing the extent; only a ``-`` hint absorbs the rest.
    3. A percentage next to absolute sizes is a ceiling on its share of the
       axis, so it can only shrink the table.
    4. Percentages that over-commit the table raise ``LayoutError``.
    5. Rounding slack goes to the largest fractional parts, so the sizes
       always add up to *total* exactly.

    WIP: mixed absolute/percentage bounds (rule 3) are not verified yet.
    """
    if not hints:
        return []
    tracks = 1 + max((t + span - 1 for t, span, _ in hints), default=-1)
    unknown = total is None or int(total) <= 0
    extent = 0 if unknown else int(total)
    claimed = [None] * tracks
    pct = [None] * tracks
    flex = []
    absolute = {}
    for track, span, hint in hints:
        if hint is None:
            continue
        if hint[0] == "%":
            # A spanning percentage is one hint: the same share on every
            # track it covers (no double division).
            for t in range(track, min(tracks, track + span)):
                pct[t] = float(hint[1])
        elif hint[1] == "-":
            flex.append(track)
        else:
            absolute[track] = (span, float(hint[1]))
    if len(flex) > 1:
        raise LayoutError(
            f"only one flexible '-' {axis} track is allowed, found {len(flex)}")
    if not absolute:
        used = 0.0
        for t, p in enumerate(pct):
            if p is not None:
                claimed[t] = extent * p / 100.0
                used += claimed[t]
        free = [t for t in range(tracks) if claimed[t] is None]
        if unknown:
            # The extent is not known yet (the web UI fills it in later):
            # percentages stay proportional, free tracks are 1 unit.
            return [v if v is not None else 1.0 for v in claimed]
        share = (extent - used) / len(free) if free else extent - used
        if share < -1:
            raise LayoutError(
                f"{axis} percentages need {_fmt(used)} cells but only "
                f"{_fmt(extent)} are available")
        for t in free:
            claimed[t] = share
        return _round_to(claimed, extent)

    if unknown:
        extent = sum(value for _, value in absolute.values())
    for track, (span, value) in absolute.items():
        share = value / span
        for t in range(track, min(tracks, track + span)):
            if claimed[t] is not None and abs(claimed[t] - share) > 1e-9:
                raise LayoutError(
                    f"{axis} track {t}: absolute sizes disagree "
                    f"({_fmt(claimed[t])} and {_fmt(share)} cells)")
            claimed[t] = share
    used = sum(v for v in claimed if v is not None)
    free = [t for t in range(tracks) if claimed[t] is None and t not in flex]
    fixed = [t for t in free if pct[t] is None]
    percent = sum(pct[t] or 0.0 for t in free)
    if percent >= 100.0:
        raise LayoutError(
            f"{axis} percentages leave no room for the absolute sizes")
    if not flex:
        # No flexible track: an absolute table shrinks to what its hints
        # cover (the declared extent is only an upper bound), and the
        # percentage tracks take their share of that smaller table.
        if percent > 0:
            # The declared extent is honoured when the percentage track can
            # hold its share of it; otherwise the percentage share wins.
            if extent - used - extent * percent / 100.0 < -1e-9:
                extent = max(extent, math.ceil(
                    used / (1.0 - percent / 100.0)))
            for t in free:
                claimed[t] = extent * (pct[t] or 0.0) / 100.0
            used = sum(v for v in claimed if v is not None)
            while extent - used < -1e-9:
                extent += 1
        for t in fixed:
            claimed[t] = 0.0
        total = sum(claimed)
        if abs(extent - total) >= 1:
            if total > extent + 1:
                raise LayoutError(
                    f"{axis} absolute sizes need {_fmt(total)} cells but "
                    f"the table provides {_fmt(extent)}; add a '-' track")
            extent = total          # the table shrinks to fit its sizes
        return _round_to(claimed, extent)

    # A flexible '-' track absorbs the rest. The table may grow past the
    # declared extent so a percentage track can hold its share next to the
    # absolute sizes; it never shrinks below them.
    if percent > 0:
        # The declared extent is used when the percentage track can hold its
        # share of it; otherwise the percentage share wins over the declared
        # size (used / (1 - pct)), and the table never drops below its
        # absolute sizes.
        share = extent * percent / 100.0
        need = math.ceil(used / (1.0 - percent / 100.0))
        if need > extent:
            extent = need
        # Rounding may leave the share a cell short; grow until it fits.
        while extent - used - extent * percent / 100.0 < -1e-9:
            extent += 1
        extent = max(extent, used)
        for t in free:
            claimed[t] = extent * (pct[t] or 0.0) / 100.0
    for t in fixed:
        claimed[t] = 0.0
    rest = extent - sum(v for v in claimed if v is not None)
    if rest < -1:
        raise LayoutError(
            f"{axis} absolute sizes need "
            f"{_fmt(sum(v for v in claimed if v is not None))} cells but the "
            f"table provides {_fmt(extent)}")
    claimed[flex[0]] = rest
    return _round_to(claimed, extent)


def _round_to(values, total):
    """Round sizes to whole cells so they add up to *total* exactly."""
    base = [int(math.floor(v + 1e-9)) for v in values]
    missing = int(round(total)) - sum(base)
    order = sorted(range(len(values)), key=lambda i: (values[i] - base[i], i),
                   reverse=True)
    for i in order[:max(0, missing)]:
        base[i] += 1
    return [float(v) for v in base]


# ======================================================================
# 3. Pandoc table extraction
# ======================================================================
def unwrap_ctor(node, tag):
    if isinstance(node, dict) and node.get("t") == tag:
        return node.get("c", [])
    return node


def _cell_attrs_dict(cell_attr):
    if not isinstance(cell_attr, list) or len(cell_attr) < 3:
        return {}
    kvs = cell_attr[2]
    out = {}
    if isinstance(kvs, list):
        for kv in kvs:
            if isinstance(kv, list) and len(kv) == 2:
                out[kv[0]] = kv[1]
    return out


def extract_cell_info(cell):
    cell = unwrap_ctor(cell, "Cell")
    if not isinstance(cell, list) or len(cell) < 5:
        return (1, 1, "", None, None)
    rowspan = cell[2] if isinstance(cell[2], int) and cell[2] > 0 else 1
    colspan = cell[3] if isinstance(cell[3], int) and cell[3] > 0 else 1
    text = cell_text_from_blocks(cell[4])
    attrs = _cell_attrs_dict(cell[0]) if len(cell) > 0 else {}
    w_hint, h_hint = _extract_size_from_attrs(attrs)
    return (rowspan, colspan, text, w_hint, h_hint)


def extract_row_with_spans(row):
    row = unwrap_ctor(row, "Row")
    if not isinstance(row, list) or len(row) < 2:
        return []
    cells = row[1]
    if not isinstance(cells, list):
        return []
    return [extract_cell_info(c) for c in cells]


def _head_rows_raw(head):
    head = unwrap_ctor(head, "TableHead")
    if not isinstance(head, list) or len(head) < 2:
        return []
    rows = head[1]
    return rows if isinstance(rows, list) else []


def extract_colspec_widths(table_c):
    if not isinstance(table_c, list) or len(table_c) < 3:
        return None
    colspecs = table_c[2]
    if not isinstance(colspecs, list):
        return None
    widths = []
    has_explicit = False
    for spec in colspecs:
        if not isinstance(spec, list) or len(spec) < 2:
            widths.append(None)
            continue
        cw = spec[1]
        if isinstance(cw, dict) and cw.get("t") == "ColWidth":
            val = cw.get("c")
            if isinstance(val, (int, float)):
                widths.append(float(val))
                has_explicit = True
            else:
                widths.append(None)
        else:
            widths.append(None)
    return widths if has_explicit else None


def extract_table_full(table_c):
    if not isinstance(table_c, list):
        return None
    col_widths = extract_colspec_widths(table_c)
    rows = []

    if len(table_c) >= 6:
        for r in _head_rows_raw(table_c[3]):
            rows.append(extract_row_with_spans(r))
        bodies = table_c[4]
        if isinstance(bodies, list):
            for body in bodies:
                body = unwrap_ctor(body, "TableBody")
                if not isinstance(body, list) or len(body) < 4:
                    continue
                for r in body[2]:
                    rows.append(extract_row_with_spans(r))
                for r in body[3]:
                    rows.append(extract_row_with_spans(r))
        for r in _head_rows_raw(table_c[5]):
            rows.append(extract_row_with_spans(r))
    elif len(table_c) >= 5:
        headers = table_c[3]
        if headers:
            rows.append([extract_cell_info(c) for c in headers])
        for r in table_c[4]:
            rows.append([extract_cell_info(c) for c in r])

    if not rows:
        return None
    return {"rows": rows, "col_widths": col_widths}


def extract_tables(ast: dict):
    tables = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("t") == "Table":
                info = extract_table_full(node.get("c", []))
                if info:
                    tables.append(info)
                return
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(ast)
    return tables


def merge_html_hints(tables, html_source):
    if not html_source:
        return tables
    try:
        parser = _HTMLCellAttrsExtractor()
        parser.feed(html_source)
        parser.close()
    except Exception:
        return tables
    html_cells = parser.cells
    if not html_cells:
        return tables

    idx = 0
    for tbl in tables:
        for r_i, row in enumerate(tbl["rows"]):
            for c_i, cell in enumerate(row):
                if idx >= len(html_cells):
                    return tables
                hc = html_cells[idx]
                rs, cs, text, w_hint, h_hint = cell
                if hc["rowspan"] != rs or hc["colspan"] != cs:
                    return tables
                if w_hint is None:
                    w_hint = hc["w_hint"]
                if h_hint is None:
                    h_hint = hc["h_hint"]
                tbl["rows"][r_i][c_i] = (rs, cs, text, w_hint, h_hint)
                idx += 1
    return tables


# ======================================================================
# 4. Grid + weight computation (shared)
# ======================================================================
def _build_grid(rows):
    grid = {}
    cells_info = {}
    cell_id = 0
    max_col = 0
    for r, row in enumerate(rows):
        c = 0
        for (rs, cs, text, wh, hh) in row:
            while (r, c) in grid:
                c += 1
            cid = cell_id
            cell_id += 1
            cells_info[cid] = {
                "row": r, "col": c, "rowspan": rs, "colspan": cs,
                "text": text, "w_hint": wh, "h_hint": hh,
            }
            for dr in range(rs):
                for dc in range(cs):
                    grid[(r + dr, c + dc)] = cid
            c += cs
            if c > max_col:
                max_col = c
    max_row = (max(r for (r, _) in grid) + 1) if grid else 0
    return grid, cells_info, max_row, max_col


def fill_weights(hints, default=1.0):
    if not hints:
        return []
    present = [h for h in hints if h is not None]
    if not present:
        return [default] * len(hints)
    units = set(h[0] for h in present)
    if len(units) > 1:
        return [default] * len(hints)
    unit = units.pop()
    total = sum(h[1] for h in present)
    n_missing = len(hints) - len(present)
    if n_missing == 0:
        return [h[1] for h in hints]
    if unit == "%" and total < 100:
        fill = max(1.0, (100.0 - total) / n_missing)
    else:
        fill = max(1.0, total / len(present))
    return [h[1] if h is not None else fill for h in hints]


def compute_weights(info):
    """Fallback weighting for pandoc-derived tables (uses col_widths or
    colspan=1 hints; spanning hints are ignored, matching prior behaviour)."""
    rows = info["rows"]
    colspec = info.get("col_widths")
    _, cells_info, max_row, max_col = _build_grid(rows)

    col_hints = [None] * max_col
    colspec_used = False
    if colspec:
        for i, w in enumerate(colspec[:max_col]):
            if w is not None:
                col_hints[i] = ("%", w * 100.0)
                colspec_used = True
    if not colspec_used:
        for cid, info_c in cells_info.items():
            if info_c["colspan"] == 1 and info_c["w_hint"] is not None:
                c = info_c["col"]
                if c < max_col and col_hints[c] is None:
                    col_hints[c] = info_c["w_hint"]

    row_hints = [None] * max_row
    for cid, info_c in cells_info.items():
        if info_c["rowspan"] == 1 and info_c["h_hint"] is not None:
            r = info_c["row"]
            if r < max_row and row_hints[r] is None:
                row_hints[r] = info_c["h_hint"]

    return fill_weights(row_hints, 1.0), fill_weights(col_hints, 1.0)


# ======================================================================
# 5. DSL parser ("Table Renderer" spec)
# ======================================================================
def decode_word(word):
    """Decode filename-safe padded Base64."""
    pad = (-len(word)) % 4
    if pad:
        word = word + "=" * pad
    try:
        return base64.urlsafe_b64decode(word.encode("ascii")).decode("utf-8")
    except Exception:
        return word


def encode_word(text):
    """Encode as filename-safe padded Base64."""
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii")


ABSOLUTE_UNITS = frozenset({"pixels", "px", "em"})
RELATIVE_UNITS = frozenset({"percentage", "%"})


def _hint_unit(value):
    """Normalise a unit token; absolute ``px``/``em`` collapse to ``px``
    because both mean character cells."""
    unit = str(value).strip().lower()
    if unit in RELATIVE_UNITS:
        return "%"
    if unit in ABSOLUTE_UNITS:
        return "px"
    raise ValueError(f"Unknown DSL unit: {value!r}")


def dsl_hint_text(hint):
    """Canonical DSL rendering of a (unit, number) hint."""
    if hint is None:
        return ""
    number = hint[1] if hint[1] == "-" else f"{hint[1]:g}"
    unit = "percentage" if hint[0] == "%" else "pixels"
    return f"{number} {unit}"


def _dsl_unit_to_hint(unit, value):
    """Build a size hint. ``value`` is a number, a numeric string (pixels)
    or ``"-"`` (flexible absolute track); percentages must be numeric."""
    unit = _hint_unit(unit)
    text = str(value).strip()
    if text == "-" and unit == "px":
        return ("px", "-")
    if text == "-":
        raise ValueError(f"'-' needs pixels/em, not {unit!r}")
    number = int(text)
    return (unit, float(number))


def parse_dsl(text, resolve=True):
    """Parse Table Renderer DSL. Returns an info dict in internal format
    with ``row_sizes``/``col_sizes``; the legacy
    ``row_weights``/``col_weights`` keys are the same numbers. With
    ``resolve=False`` no layout is computed (handy for round trips)."""
    lines = text.splitlines()
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines):
        raise ValueError("Empty DSL input")
    parts = lines[idx].split()
    if len(parts) != 2:
        raise ValueError(f"Expected 'W H' first line, got: {lines[idx]!r}")
    try:
        W, H = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(f"Expected two integers first line, got: {lines[idx]!r}")
    idx += 1

    rows = []
    current_row = None
    current_cell = None

    for line in lines[idx:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cmd = parts[0]
        if cmd == "begin":
            if len(parts) < 2:
                raise ValueError(f"Malformed begin: {line!r}")
            tag = parts[1]
            if tag == "table":
                pass
            elif tag == "tr":
                current_row = []
                rows.append(current_row)
            elif tag == "td":
                current_cell = {
                    "rowspan": 1, "colspan": 1, "text": "",
                    "w_hint": None, "h_hint": None,
                }
            else:
                raise ValueError(f"Unknown tag: {tag}")
        elif cmd == "end":
            tag = parts[1] if len(parts) > 1 else ""
            if tag == "td":
                if current_row is None or current_cell is None:
                    raise ValueError("end td without open cell")
                current_row.append(current_cell)
                current_cell = None
            elif tag == "tr":
                current_row = None
            elif tag == "table":
                pass
            else:
                raise ValueError(f"Unknown end tag: {tag}")
        elif cmd == "prop":
            if len(parts) < 4:
                raise ValueError(f"Malformed prop: {line!r}")
            name, val, unit = parts[1], parts[2], parts[3]
            if current_cell is None:
                raise ValueError(f"prop outside cell: {line!r}")
            if name == "rowspan":
                current_cell["rowspan"] = int(val)
            elif name == "colspan":
                current_cell["colspan"] = int(val)
            elif name in ("width", "height"):
                key = "w_hint" if name == "width" else "h_hint"
                current_cell[key] = _dsl_unit_to_hint(unit, val)
            else:
                raise ValueError(f"Unknown prop: {name}")
        elif cmd == "text":
            if len(parts) < 2:
                raise ValueError(f"Malformed text: {line!r}")
            if current_cell is None:
                raise ValueError("text outside cell")
            current_cell["text"] = decode_word(parts[1])
        else:
            raise ValueError(f"Unknown command: {cmd!r}")

    converted = []
    for row in rows:
        converted.append([
            (c["rowspan"], c["colspan"], c["text"], c["w_hint"], c["h_hint"])
            for c in row
        ])

    info = {"rows": converted, "col_widths": None, "_W": W, "_H": H}
    if resolve:
        row_sizes, col_sizes = compute_dsl_weights(info, W, H)
        info["row_sizes"] = row_sizes
        info["col_sizes"] = col_sizes
    return info


def dsl_serialize(info):
    """Render a table dict (``{"rows": ..., "_W": ..., "_H": ...}``) as
    Table Renderer DSL text. Texts are Base64-encoded so they stay safe."""
    lines = [f"{int(info.get('_W') or 0)} {int(info.get('_H') or 0)}",
             "begin table"]
    for row in info["rows"]:
        lines.append("begin tr")
        for rowspan, colspan, text, w_hint, h_hint in row:
            lines.append("begin td")
            for prop, value in (("rowspan", rowspan), ("colspan", colspan)):
                if value != 1:
                    lines.append(f"prop {prop} {value} pixels")
            for prop, hint in (("width", w_hint), ("height", h_hint)):
                if hint is not None:
                    lines.append(f"prop {prop} {dsl_hint_text(hint)}")
            if text:
                lines.append(f"text {encode_word(text)}")
            lines.append("end td")
        lines.append("end tr")
    lines.append("end table")
    return "\n".join(lines) + "\n"


def compute_dsl_weights(info, W, H):
    """Resolve the DSL table's hints into absolute cell sizes per track.

    Returns ``(row_sizes, col_sizes)``; see :func:`resolve_sizes` and the
    module docstring for the absolute/relative rules. ``W``/``H`` are the
    declared extent in character cells (``None``/absent when unknown)."""
    cols, rws = grid_size_hints(info["rows"])
    col_sizes = resolve_sizes(cols, W, "width")
    row_sizes = resolve_sizes(rws, H, "height")
    return row_sizes, col_sizes


# ======================================================================
# 6. Layout generation
# ======================================================================
def _pick_weights(info):
    if "row_sizes" in info and "col_sizes" in info:
        return info["row_sizes"], info["col_sizes"]
    if "row_weights" in info and "col_weights" in info:
        return info["row_weights"], info["col_weights"]
    return layout_sizes(info)
    return compute_weights(info)


def build_table_layout(info):
    rows = info["rows"]
    _, cells_info, max_row, max_col = _build_grid(rows)
    if not cells_info:
        return {}, {}, []

    row_weights, col_weights = _pick_weights(info)

    var_counter = [0]
    pane_map = {}
    split_cmds = _split_layout(
        "P0",
        (0, 0, max_row, max_col),
        list(cells_info.keys()),
        cells_info,
        row_weights,
        col_weights,
        var_counter,
        pane_map,
    )
    return cells_info, pane_map, split_cmds


def _split_layout(pane_var, rect, cell_ids, cells_info,
                  row_weights, col_weights, var_counter, pane_map):
    if not cell_ids:
        return []
    if len(cell_ids) == 1:
        pane_map[cell_ids[0]] = pane_var
        return []

    # ---- Horizontal (top / bottom) first -------------------------------
    for split_row in range(rect[0] + 1, rect[0] + rect[2]):
        if any(
            cells_info[c]["row"] < split_row
            < cells_info[c]["row"] + cells_info[c]["rowspan"]
            for c in cell_ids
        ):
            continue
        top = [c for c in cell_ids if cells_info[c]["row"] < split_row]
        bot = [c for c in cell_ids if cells_info[c]["row"] >= split_row]
        if not top or not bot:
            continue

        top_w = sum(row_weights[rect[0]:split_row])
        bot_w = sum(row_weights[split_row:rect[0] + rect[2]])
        bot_pct, keep = _split_percent(top_w, bot_w, "row", split_row)
        if not keep:
            continue

        var_counter[0] += 1
        new_var = f"P{var_counter[0]}"
        cmds = [
            f'{new_var}=$(tmux split-window -v -p {bot_pct} '
            f'-t "${{{pane_var}}}" -P -F "#{{pane_id}}")'
        ]
        top_rect = (rect[0], rect[1], split_row - rect[0], rect[3])
        bot_rect = (split_row, rect[1], rect[0] + rect[2] - split_row, rect[3])
        cmds += _split_layout(pane_var, top_rect, top, cells_info,
                              row_weights, col_weights, var_counter, pane_map)
        cmds += _split_layout(new_var, bot_rect, bot, cells_info,
                              row_weights, col_weights, var_counter, pane_map)
        return cmds

    # ---- Vertical (left / right) --------------------------------------
    for split_col in range(rect[1] + 1, rect[1] + rect[3]):
        if any(
            cells_info[c]["col"] < split_col
            < cells_info[c]["col"] + cells_info[c]["colspan"]
            for c in cell_ids
        ):
            continue
        left = [c for c in cell_ids if cells_info[c]["col"] < split_col]
        right = [c for c in cell_ids if cells_info[c]["col"] >= split_col]
        if not left or not right:
            continue

        left_w = sum(col_weights[rect[1]:split_col])
        right_w = sum(col_weights[split_col:rect[1] + rect[3]])
        right_pct, keep = _split_percent(left_w, right_w, "col", split_col)
        if not keep:
            continue

        var_counter[0] += 1
        new_var = f"P{var_counter[0]}"
        cmds = [
            f'{new_var}=$(tmux split-window -h -p {right_pct} '
            f'-t "${{{pane_var}}}" -P -F "#{{pane_id}}")'
        ]
        left_rect = (rect[0], rect[1], rect[2], split_col - rect[1])
        right_rect = (rect[0], split_col, rect[2], rect[1] + rect[3] - split_col)
        cmds += _split_layout(pane_var, left_rect, left, cells_info,
                              row_weights, col_weights, var_counter, pane_map)
        cmds += _split_layout(new_var, right_rect, right, cells_info,
                              row_weights, col_weights, var_counter, pane_map)
        return cmds

    for cid in cell_ids:
        pane_map[cid] = pane_var
    return []


def _split_percent(before, after, axis, at):
    """Percentage for the second (bottom/right) pane of a split.

    Returns ``(percent, keep)``. Sizes are absolute integer cells (see
    :func:`resolve_sizes`), so the share is ``after / (before + after)``
    rounded to a whole percent and clamped to 1..99. A split where one side
    has no cells at all has nothing to show: ``keep`` is False and the split
    is skipped (#FIXME such cells then share the other pane).
    """
    if before + after <= 0:
        return 50, False
    if min(before, after) <= 0:
        return 50, False
    percent = round(after * 100.0 / (before + after))
    return max(1, min(99, int(percent))), True


def generate_tmux_script(tables, session_name="table_session", attach=True):
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    lines.append(f"tmux new-session -d -s {shlex.quote(session_name)}")
    lines.append("")

    for win_idx, table_info in enumerate(tables, start=1):
        win_name = f"table_{win_idx}"
        target = f"{session_name}:{win_name}"

        if win_idx == 1:
            lines.append(
                f"tmux rename-window -t {shlex.quote(session_name)}:0 "
                f"{shlex.quote(win_name)}"
            )
        else:
            lines.append(
                f"tmux new-window -t {shlex.quote(session_name)} "
                f"-n {shlex.quote(win_name)}"
            )

        cells_info, pane_map, split_cmds = build_table_layout(table_info)
        if not cells_info:
            lines.append("# (empty table)")
            lines.append("")
            continue

        lines.append(
            f'P0=$(tmux list-panes -t {shlex.quote(target)} '
            f'-F "#{{pane_id}}" | head -n 1)'
        )
        lines.extend(split_cmds)

        for cid in sorted(cells_info.keys()):
            cell = cells_info[cid]
            text = (cell["text"] or "").strip()
            if not text:
                continue
            var = pane_map.get(cid)
            if not var:
                continue
            lines.append(
                f"# ({cell['row']},{cell['col']}) "
                f"{cell['rowspan']}x{cell['colspan']}"
            )
            lines.append(
                f'tmux send-keys -t "${{{var}}}" '
                f"{shlex.quote(text)} C-m"
            )
        lines.append("")

    if attach:
        lines.append(f"tmux attach-session -t {shlex.quote(session_name)}")

    return "\n".join(lines)


# ======================================================================
# 7. Spec-compliant pixel-box output (bonus, for validation)
# ======================================================================
def _axis_sizes(info, cells_info):
    """Absolute cell sizes of both axes, from :func:`layout_sizes`."""
    rows, cols = layout_sizes(info)
    return rows, cols


HTML_RELATIVE_ATTRS = ("width", "height")


def _table_body(rows, widths=None):
    """Build a pandoc 3.x Table payload with ``(text, rowspan, colspan)``
    cells (shared by tests and frontends)."""
    def cell(text, rowspan, colspan):
        return {"t": "Cell", "c": [["", [], []], {"t": "AlignDefault"},
                                   rowspan, colspan,
                                   [{"t": "Para", "c": [{"t": "Str",
                                                         "c": text}]}]]}
    body = {"t": "TableBody", "c": ["", 0,
                                    [{"t": "Row", "c": ["", [cell(*c)
                                                             for c in row]]}
                                     for row in rows], []]}
    colspecs = [["AlignDefault", {"t": "ColWidth", "c": w}]
                for w in widths] if widths else []
    return [["", [], []], ["", []], colspecs,
            {"t": "TableHead", "c": ["", []]}, [body],
            {"t": "TableHead", "c": ["", []]}]


# HTML width/height hints are relative (a fraction of the pane) *unless*
# they use px/em/pt, which are absolute character cells. Values without a
# unit are widest-practice pixels in browsers, but for a terminal layout a
# bare number is read as characters - see README "Size units".
def _html_sizes(html_source, tables):
    """Attach the HTML width/height attributes to the pandoc cells."""
    parser = _HTMLCellAttrsExtractor()
    try:
        parser.feed(html_source)
        parser.close()
    except Exception:            # pragma: no cover - html.parser is lenient
        return tables
    idx = 0
    for table in tables:
        for row in table["rows"]:
            for i, (rowspan, colspan, text, w_hint, h_hint) in enumerate(row):
                if idx >= len(parser.cells):
                    return tables
                cell = parser.cells[idx]
                idx += 1
                if (cell["rowspan"], cell["colspan"]) != (rowspan, colspan):
                    return tables
                # A colspan cell covers its whole row part; a legacy
                # width=5 on it would give every covered column 2.5 cells,
                # which is meaningless. Full width is the useful default.
                # #TODO let the DSL/user override this per cell.
                if colspan > 1:
                    w_hint = ("%", 100.0)
                else:
                    w_hint = w_hint or cell["w_hint"]
                h_hint = h_hint or cell["h_hint"]
                row[i] = (rowspan, colspan, text, w_hint, h_hint)
    return tables


def _colpct(widths, col):
    """Percentage hint from a pandoc column width (1.0 == 100%)."""
    if col >= len(widths) or widths[col] is None:
        return None
    return ("%", round(widths[col] * 100.0, 2))


def to_dsl(text, from_format):
    """Parse any frontend into Table Renderer DSL text.

    ``from_format`` is ``"dsl"`` (returned unchanged), ``"html"`` or
    ``"rst"`` (parsed with pandoc; HTML ``width``/``height`` attributes are
    added as size hints). :func:`main` and the web UI only ever hand DSL
    to :func:`parse_dsl`, so every frontend follows the same path.
    """
    from_format = from_format.lower()
    if from_format == "dsl":
        return text
    if from_format not in ("html", "rst"):
        raise ValueError(f"Unsupported frontend: {from_format!r}")
    ast = pandoc_json(text, from_format)
    tables = extract_tables(ast)
    if not tables:
        raise ValueError("No tables found in the input.")
    if from_format == "html":
        tables = _html_sizes(text, tables)
        # #TODO HTML has no declared character size: a common 80x24 pane is
        # assumed so relative hints have an extent (and #FIXME this should
        # become a --width/--height option shared with the web UI).
        for table in tables:
            table["_W"] = table.get("_W") or 80
            table["_H"] = table.get("_H") or 24
            # Legacy width attributes (e.g. width=5) are absolute cells: a
            # table made only of them has no room for an 80-cell assumption.
            # #TODO make the assumed extent a --width/--height option.
            if not any(c[3] is not None and c[3][0] == "%"
                       for row in table["rows"] for c in row):
                absolute = sum(c[3][1] for row in table["rows"]
                               for c in row if c[3] is not None)
                table["_W"] = min(table["_W"], max(1, int(absolute)))
            # #FIXME a cell without height hints fills the assumed 24 rows;
            # give unhinted axes one unit instead once the resolver supports
            # "provisional" per-axis extents.
            if not any(c[4] is not None for row in table["rows"] for c in row):
                table["_H"] = 1
    else:
        # Pandoc column widths (fractions of the table) are the RST
        # frontend's relative sizes; they become percentages.
        for table in tables:
            widths = table.get("col_widths") or []
            for r, row in enumerate(table["rows"]):
                table["rows"][r] = [
                    (rs, cs, t, w_hint or _colpct(widths, c), h)
                    for c, (rs, cs, t, w_hint, h) in enumerate(row)]
    return "\n# ---\n".join(dsl_serialize(t) for t in tables)


def render_boxes(info):
    rows = info["rows"]
    _, cells_info, max_row, max_col = _build_grid(rows)

    row_sizes, col_sizes = _axis_sizes(info, cells_info)
    X = [0.0]
    for w in col_sizes:
        X.append(X[-1] + w)
    Y = [0.0]
    for h in row_sizes:
        Y.append(Y[-1] + h)

    # Sizes are already whole cells; add the last edge so a span that
    # reaches past the last sized track can still be clipped to the table.
    x_int = [int(math.floor(x + 0.5)) for x in X]
    y_int = [int(math.floor(y + 0.5)) for y in Y]
    x_int.append(max(x_int) + 1)
    y_int.append(max(y_int) + 1)

    boxes = []
    for cid, cell in cells_info.items():
        r, c = cell["row"], cell["col"]
        rs, cs = cell["rowspan"], cell["colspan"]
        x = x_int[c]
        y = y_int[r]
        # #FIXME a span whose tracks were not all sized stops at the last
        # sized track instead of covering the whole table.
        w = x_int[min(c + cs, len(x_int) - 1)] - x
        h = y_int[min(r + rs, len(y_int) - 1)] - y
        word = encode_word(cell["text"])
        boxes.append((x, y, w, h, word))
    boxes.sort()
    return "\n".join(f"{x} {y} {w} {h} {s}" for x, y, w, h, s in boxes)


# ======================================================================
# 8. Format detection + CLI
# ======================================================================
def is_dsl(text):
    lines = text.splitlines()
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines):
        return False
    parts = lines[idx].split()
    if len(parts) != 2:
        return False
    try:
        int(parts[0]); int(parts[1])
    except ValueError:
        return False
    for l in lines[idx + 1:]:
        l = l.strip()
        if not l:
            continue
        toks = l.split()
        return len(toks) >= 2 and toks[0] == "begin" and toks[1] == "table"
    return False


SAMPLE_DSL = """\
80 24
begin table
begin tr
begin td
prop width 30 percentage
text btop
end td
begin td
prop width - pixels
text watch -n 2 nvidia-smi
end td
end tr
end table
"""


def run_web(port):
    """Launch the Streamlit UI (:mod:`deepseek_tools.tmux_manager_web`)."""
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        Path(_web_module().__file__).with_name("tmux_manager_web.py").as_posix(),
        "--server.port", str(port),
    ]
    return subprocess.call(cmd)


def _web_module():
    from deepseek_tools import tmux_manager_web
    return tmux_manager_web


def main():
    parser = argparse.ArgumentParser(
        description="Convert tables (HTML / RST / Table Renderer DSL) to a "
                    "tmux Bash script."
    )
    parser.add_argument(
        "input", nargs="?",
        help="Input file (use '-' for stdin). Format inferred from extension "
             "or content; for stdin you must specify --from.",
    )
    parser.add_argument(
        "--from", dest="from_format", default=None,
        help="Pandoc input format (html, rst, ...) or 'dsl'.",
    )
    parser.add_argument("-o", "--output", default="-",
                        help="Output file (default: stdout).")
    parser.add_argument("--session", default="table_session",
                        help="tmux session name (default: table_session).")
    parser.add_argument("--no-attach", action="store_true",
                        help="Do not append 'tmux attach-session'.")
    parser.add_argument("--emit", choices=("tmux", "boxes"), default="tmux",
                        help="Output format: tmux script (default) or spec "
                             "pixel boxes ('boxes').")
    parser.add_argument("--debug", action="store_true",
                        help="Dump the pandoc JSON AST to stderr.")
    parser.add_argument("--recipe", action="store_true",
                        help="Print a sample DSL table and exit.")
    parser.add_argument("--port", type=int, default=8501,
                        help="Port when opening the web UI.")
    args = parser.parse_args()

    if args.recipe:
        sys.stdout.write(SAMPLE_DSL)
        return
    # `tmux_manager.py web` or `--web` opens the Streamlit UI. The word is
    # checked here rather than by argparse subparsers, which would reject
    # ordinary input paths.
    if args.input == "web":
        run_web(args.port)
        return
    if not getattr(args, "input", None):
        parser.error("the following arguments are required: input")

    if args.input == "-":
        text = sys.stdin.read()
        if args.from_format:
            from_format = args.from_format
        elif is_dsl(text):
            from_format = "dsl"
        else:
            parser.error("Cannot infer format from stdin. Use --from.")
    else:
        path = Path(args.input)
        text = path.read_text(encoding="utf-8")
        if args.from_format:
            from_format = args.from_format
        else:
            suffix = path.suffix.lower()
            if suffix in (".html", ".htm"):
                from_format = "html"
            elif suffix in (".rst", ".rest"):
                from_format = "rst"
            elif suffix in (".dsl", ".table", ".tbl"):
                from_format = "dsl"
            elif is_dsl(text):
                from_format = "dsl"
            else:
                parser.error(
                    f"Cannot infer format from extension '{suffix}'. "
                    "Use --from to specify the Pandoc input format."
                )

    try:
        # Every frontend is parsed into DSL text first, then laid out.
        dsl_text = to_dsl(text, from_format)
    except ValueError as exc:
        sys.exit(str(exc))
    tables = [parse_dsl(chunk) for chunk in dsl_text.split("\n# ---\n")
              if chunk.strip()]
    if not tables:
        sys.exit("No tables found in the input.")
    if not tables:
        sys.exit("No tables found in the input.")

    if args.emit == "boxes":
        chunks = []
        for t_idx, t in enumerate(tables):
            if len(tables) > 1:
                chunks.append(f"# Table {t_idx + 1}")
            chunks.append(render_boxes(t))
        out = "\n".join(chunks)
    else:
        out = generate_tmux_script(
            tables,
            session_name=args.session,
            attach=not args.no_attach,
        )

    if args.output == "-":
        sys.stdout.write(out)
        if not out.endswith("\n"):
            sys.stdout.write("\n")
    else:
        dest = Path(args.output)
        dest.write_text(out, encoding="utf-8")
        dest.chmod(0o755)


if __name__ == "__main__":
    main()
