#!/usr/bin/env python3
"""
table_to_tmux.py – Convert HTML / reStructuredText tables (and the
Table Renderer DSL) into a tmux Bash script.

Supported input formats:
  * HTML       (via pandoc)
  * RST        (via pandoc)
  * DSL        (the line-based "Table Renderer" command language)

Layout of each tmux window mirrors the visual grid of the table:
rowspan/colspan are preserved, and width/height hints are used to
weight the split percentages.

Usage:
    python table_to_tmux.py input.html
    python table_to_tmux.py input.rst -o setup.sh
    python table_to_tmux.py input.dsl -o setup.sh
    cat table.dsl | python table_to_tmux.py - --from dsl
    python table_to_tmux.py input.dsl --emit boxes     # spec pixel boxes
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


def _dsl_unit_to_hint(unit, value):
    if unit == "pixels":
        return ("px", float(value))
    if unit == "percentage":
        return ("%", float(value))
    raise ValueError(f"Unknown DSL unit: {unit!r}")


def parse_dsl(text):
    """Parse Table Renderer DSL. Returns an info dict in internal format,
    with row_weights/col_weights pre-computed per the spec."""
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
            v = int(val)
            if name == "rowspan":
                current_cell["rowspan"] = v
            elif name == "colspan":
                current_cell["colspan"] = v
            elif name == "width":
                current_cell["w_hint"] = _dsl_unit_to_hint(unit, v)
            elif name == "height":
                current_cell["h_hint"] = _dsl_unit_to_hint(unit, v)
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
    row_weights, col_weights = compute_dsl_weights(info, W, H)
    info["row_weights"] = row_weights
    info["col_weights"] = col_weights
    return info


def compute_dsl_weights(info, W, H):
    """Per the DSL spec: explicit constraints fix tracks (divided equally
    across spans); unconstrained tracks share the remainder equally."""
    rows = info["rows"]
    _, cells_info, max_row, max_col = _build_grid(rows)

    col_explicit = [None] * max_col
    for cid, cell in cells_info.items():
        wh = cell["w_hint"]
        if wh is None:
            continue
        unit, val = wh
        length = (W * val / 100.0) if unit == "%" else float(val)
        k = cell["colspan"]
        per = length / k
        for t in range(k):
            col_explicit[cell["col"] + t] = per

    S_w = sum(v for v in col_explicit if v is not None)
    n_unc = sum(1 for v in col_explicit if v is None)
    if n_unc > 0:
        per_unc = (W - S_w) / n_unc
        col_weights = [v if v is not None else per_unc for v in col_explicit]
    else:
        col_weights = col_explicit

    row_explicit = [None] * max_row
    for cid, cell in cells_info.items():
        hh = cell["h_hint"]
        if hh is None:
            continue
        unit, val = hh
        length = (H * val / 100.0) if unit == "%" else float(val)
        l = cell["rowspan"]
        per = length / l
        for t in range(l):
            row_explicit[cell["row"] + t] = per

    S_h = sum(v for v in row_explicit if v is not None)
    n_unc = sum(1 for v in row_explicit if v is None)
    if n_unc > 0:
        per_unc = (H - S_h) / n_unc
        row_weights = [v if v is not None else per_unc for v in row_explicit]
    else:
        row_weights = row_explicit

    return row_weights, col_weights


# ======================================================================
# 6. Layout generation
# ======================================================================
def _pick_weights(info):
    if "row_weights" in info and "col_weights" in info:
        return info["row_weights"], info["col_weights"]
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

        top_w = sum(row_weights[rect[0]:split_row]) or 1.0
        bot_w = sum(row_weights[split_row:rect[0] + rect[2]]) or 1.0
        bot_pct = max(1, min(99, round(bot_w * 100.0 / (top_w + bot_w))))

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

        left_w = sum(col_weights[rect[1]:split_col]) or 1.0
        right_w = sum(col_weights[split_col:rect[1] + rect[3]]) or 1.0
        right_pct = max(1, min(99, round(right_w * 100.0 / (left_w + right_w))))

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
def render_boxes(info):
    rows = info["rows"]
    _, cells_info, max_row, max_col = _build_grid(rows)

    row_w, col_w = _pick_weights(info)

    X = [0.0]
    for w in col_w:
        X.append(X[-1] + w)
    Y = [0.0]
    for h in row_w:
        Y.append(Y[-1] + h)

    x_int = [int(math.floor(x + 0.5)) for x in X]
    y_int = [int(math.floor(y + 0.5)) for y in Y]

    boxes = []
    for cid, cell in cells_info.items():
        r, c = cell["row"], cell["col"]
        rs, cs = cell["rowspan"], cell["colspan"]
        x = x_int[c]
        y = y_int[r]
        w = x_int[c + cs] - x
        h = y_int[r + rs] - y
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


def main():
    parser = argparse.ArgumentParser(
        description="Convert tables (HTML / RST / Table Renderer DSL) to a "
                    "tmux Bash script."
    )
    parser.add_argument(
        "input",
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
    args = parser.parse_args()

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

    if from_format == "dsl":
        tables = [parse_dsl(text)]
    else:
        ast = pandoc_json(text, from_format)
        if args.debug:
            json.dump(ast, sys.stderr, indent=2)
            sys.stderr.write("\n")
        tables = extract_tables(ast)
        if not tables:
            sys.exit("No tables found in the input.")
        if from_format.startswith("html"):
            tables = merge_html_hints(tables, text)

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
