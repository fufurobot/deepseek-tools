#!/usr/bin/env python3
"""
table_to_tmux.py – Convert HTML / reStructuredText tables to a tmux Bash script.

Works with pandoc < 2.10 (old Table AST), pandoc 2.10–2.19 (tagged TableHead/
TableBody/Row/Cell), and pandoc 3.x (pandoc-types >= 1.23, flat records).

The layout of each tmux window mirrors the *visual* grid of the table:
rowspan/colspan are preserved, and panes are created by recursively
bisecting the window along split lines that don't cross any cell.

Usage:
    python table_to_tmux.py input.html
    python table_to_tmux.py input.rst -o setup.sh
    cat table.html | python table_to_tmux.py - --from html
"""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


# ----------------------------------------------------------------------
# 1. Pandoc JSON helpers
# ----------------------------------------------------------------------
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
    """Extract plain text from a list of Pandoc Inline elements."""
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


# ----------------------------------------------------------------------
# 1b. Pandoc table parsing, preserving rowspan / colspan
# ----------------------------------------------------------------------
def unwrap_ctor(node, tag):
    """
    Return the payload `c` of a tagged {t: tag, c: ...} node.
    Otherwise return the node unchanged (pandoc-types >= 1.23 flat records).
    """
    if isinstance(node, dict) and node.get("t") == tag:
        return node.get("c", [])
    return node


def extract_cell_info(cell):
    """Return (rowspan, colspan, text) for a pandoc Cell."""
    cell = unwrap_ctor(cell, "Cell")
    if not isinstance(cell, list) or len(cell) < 5:
        return (1, 1, "")
    rowspan = cell[2] if len(cell) > 2 and isinstance(cell[2], int) else 1
    colspan = cell[3] if len(cell) > 3 and isinstance(cell[3], int) else 1
    text = cell_text_from_blocks(cell[4])
    return (rowspan, colspan, text)


def extract_row_with_spans(row):
    row = unwrap_ctor(row, "Row")
    if not isinstance(row, list) or len(row) < 2:
        return []
    cells = row[1]
    if not isinstance(cells, list):
        return []
    return [extract_cell_info(c) for c in cells]


def extract_head_rows_raw(head):
    """Return the raw row list from a TableHead / TableFoot."""
    head = unwrap_ctor(head, "TableHead")
    if not isinstance(head, list) or len(head) < 2:
        return []
    rows = head[1]
    return rows if isinstance(rows, list) else []


def extract_table_rows_with_spans(table_c):
    if not isinstance(table_c, list):
        return []

    # ---- New format (pandoc >= 2.10): 6 fields -------------------------
    if len(table_c) >= 6:
        rows = []
        for r in extract_head_rows_raw(table_c[3]):
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
        if len(table_c) > 5:
            for r in extract_head_rows_raw(table_c[5]):
                rows.append(extract_row_with_spans(r))
        return rows

    # ---- Old format (pandoc < 2.10): 5 fields --------------------------
    if len(table_c) >= 5:
        rows = []
        headers = table_c[3]
        if headers:
            rows.append([extract_cell_info(c) for c in headers])
        for r in table_c[4]:
            rows.append([extract_cell_info(c) for c in r])
        return rows

    return []


def extract_tables(ast: dict) -> list:
    tables = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("t") == "Table":
                rows = extract_table_rows_with_spans(node.get("c", []))
                if rows:
                    tables.append(rows)
                return
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(ast)
    return tables


# ----------------------------------------------------------------------
# 2. Layout generation
# ----------------------------------------------------------------------
def build_table_layout(table_with_spans):
    """
    table_with_spans: list of rows of (rowspan, colspan, text) tuples.

    Returns:
      cells_info : dict  cell_id -> (top_row, left_col, height, width, text)
      pane_map   : dict  cell_id -> shell variable name (e.g. "P2")
      split_cmds : list of bash commands that create the layout
                   (assumes P0 is the initial pane's ID)
    """
    grid = {}
    cells_info = {}
    cell_id = 0

    for r, row in enumerate(table_with_spans):
        c = 0
        for (rowspan, colspan, text) in row:
            while (r, c) in grid:
                c += 1
            cid = cell_id
            cell_id += 1
            cells_info[cid] = (r, c, rowspan, colspan, text)
            for dr in range(rowspan):
                for dc in range(colspan):
                    grid[(r + dr, c + dc)] = cid
            c += colspan

    if not cells_info:
        return {}, {}, []

    max_row = max(info[0] + info[2] for info in cells_info.values())
    max_col = max(info[1] + info[3] for info in cells_info.values())

    var_counter = [0]
    pane_map = {}
    split_cmds = _split_layout(
        "P0",
        (0, 0, max_row, max_col),
        list(cells_info.keys()),
        cells_info,
        var_counter,
        pane_map,
    )
    return cells_info, pane_map, split_cmds


def _split_layout(pane_var, rect, cell_ids, cells_info, var_counter, pane_map):
    """
    Recursively split `pane_var` (which covers `rect`) into one pane per cell.
    """
    if not cell_ids:
        return []
    if len(cell_ids) == 1:
        pane_map[cell_ids[0]] = pane_var
        return []

    # ---- Horizontal split (top/bottom) ---------------------------------
    for split_row in range(rect[0] + 1, rect[0] + rect[2]):
        crosses = any(
            cells_info[c][0] < split_row < cells_info[c][0] + cells_info[c][2]
            for c in cell_ids
        )
        if crosses:
            continue
        top_cells = [c for c in cell_ids if cells_info[c][0] < split_row]
        bot_cells = [c for c in cell_ids if cells_info[c][0] >= split_row]
        if not top_cells or not bot_cells:
            continue

        var_counter[0] += 1
        new_var = f"P{var_counter[0]}"
        top_h = split_row - rect[0]
        bot_h = rect[0] + rect[2] - split_row
        bot_pct = max(1, min(99, bot_h * 100 // (top_h + bot_h)))

        cmds = [
            f'{new_var}=$(tmux split-window -v -p {bot_pct} '
            f'-t "${{{pane_var}}}" -P -F "#{{pane_id}}")'
        ]
        top_rect = (rect[0], rect[1], top_h, rect[3])
        bot_rect = (split_row, rect[1], bot_h, rect[3])
        cmds += _split_layout(pane_var, top_rect, top_cells,
                              cells_info, var_counter, pane_map)
        cmds += _split_layout(new_var, bot_rect, bot_cells,
                              cells_info, var_counter, pane_map)
        return cmds

    # ---- Vertical split (left/right) -----------------------------------
    for split_col in range(rect[1] + 1, rect[1] + rect[3]):
        crosses = any(
            cells_info[c][1] < split_col < cells_info[c][1] + cells_info[c][3]
            for c in cell_ids
        )
        if crosses:
            continue
        left_cells = [c for c in cell_ids if cells_info[c][1] < split_col]
        right_cells = [c for c in cell_ids if cells_info[c][1] >= split_col]
        if not left_cells or not right_cells:
            continue

        var_counter[0] += 1
        new_var = f"P{var_counter[0]}"
        left_w = split_col - rect[1]
        right_w = rect[1] + rect[3] - split_col
        right_pct = max(1, min(99, right_w * 100 // (left_w + right_w)))

        cmds = [
            f'{new_var}=$(tmux split-window -h -p {right_pct} '
            f'-t "${{{pane_var}}}" -P -F "#{{pane_id}}")'
        ]
        left_rect = (rect[0], rect[1], rect[2], left_w)
        right_rect = (rect[0], split_col, rect[2], right_w)
        cmds += _split_layout(pane_var, left_rect, left_cells,
                              cells_info, var_counter, pane_map)
        cmds += _split_layout(new_var, right_rect, right_cells,
                              cells_info, var_counter, pane_map)
        return cmds

    # Fallback: this shouldn't happen for rectangular tables, but be safe.
    for cid in cell_ids:
        pane_map[cid] = pane_var
    return []


def generate_tmux_script(tables, session_name="table_session", attach=True):
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    lines.append(f"tmux new-session -d -s {shlex.quote(session_name)}")
    lines.append("")

    for win_idx, table in enumerate(tables, start=1):
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

        cells_info, pane_map, split_cmds = build_table_layout(table)
        if not cells_info:
            lines.append("# (empty table)")
            lines.append("")
            continue

        lines.append(
            f'P0=$(tmux list-panes -t {shlex.quote(target)} '
            f'-F "#{{pane_id}}" | head -n 1)'
        )
        lines.extend(split_cmds)

        # Send each cell's command to its pane.
        for cid in sorted(cells_info.keys()):
            top, left, h, w, text = cells_info[cid]
            text = text.strip()
            if not text:
                continue
            var = pane_map.get(cid)
            if not var:
                continue
            lines.append(f"# ({top},{left}) {h}x{w}")
            lines.append(
                f'tmux send-keys -t "${{{var}}}" '
                f'{shlex.quote(text)} C-m'
            )

        lines.append("")

    if attach:
        lines.append(f"tmux attach-session -t {shlex.quote(session_name)}")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# 3. CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Convert HTML / reStructuredText tables to a tmux Bash script."
    )
    parser.add_argument(
        "input",
        help="Input file (use '-' for stdin). The format is inferred from the "
             "extension; for stdin you must specify --from.",
    )
    parser.add_argument(
        "--from", dest="from_format", default=None,
        help="Pandoc input format (e.g. html, rst). Required when reading from stdin.",
    )
    parser.add_argument("-o", "--output", default="-",
                        help="Output file (default: stdout).")
    parser.add_argument("--session", default="table_session",
                        help="tmux session name (default: table_session).")
    parser.add_argument("--no-attach", action="store_true",
                        help="Do not append 'tmux attach-session' to the generated script.")
    parser.add_argument("--debug", action="store_true",
                        help="Dump the pandoc JSON AST to stderr (for debugging).")
    args = parser.parse_args()

    if args.input == "-":
        if not args.from_format:
            parser.error("--from is required when reading from stdin.")
        text = sys.stdin.read()
        from_format = args.from_format
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
            else:
                parser.error(
                    f"Cannot infer format from extension '{suffix}'. "
                    "Use --from to specify the Pandoc input format."
                )

    ast = pandoc_json(text, from_format)

    if args.debug:
        json.dump(ast, sys.stderr, indent=2)
        sys.stderr.write("\n")

    tables = extract_tables(ast)
    if not tables:
        sys.exit("No tables found in the input.")

    script = generate_tmux_script(
        tables,
        session_name=args.session,
        attach=not args.no_attach,
    )

    if args.output == "-":
        sys.stdout.write(script)
    else:
        out = Path(args.output)
        out.write_text(script, encoding="utf-8")
        out.chmod(0o755)


if __name__ == "__main__":
    main()
