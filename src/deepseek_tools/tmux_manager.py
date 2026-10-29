#!/usr/bin/env python3
"""
table_to_tmux.py – Convert HTML / reStructuredText tables to a tmux Bash script.

Works with pandoc < 2.10 (old Table AST), pandoc 2.10–2.19 (tagged TableHead/
TableBody/Row/Cell), and pandoc 3.x (pandoc-types >= 1.23, flat records).

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
            # Code Attr Text → c == [attr, "text"]
            parts.append(c[1] if isinstance(c, list) and len(c) > 1 else "")
        elif t == "Space":
            parts.append(" ")
        elif t in ("SoftBreak", "LineBreak"):
            parts.append("\n")
        elif t in ("Emph", "Strong", "Strikeout", "Superscript",
                   "Subscript", "SmallCaps"):
            parts.append(collect_inlines(c))
        elif t == "Link":
            # c == [attr, inlines, [url, title]]
            parts.append(collect_inlines(c[1]))
        elif t == "Image":
            pass
        elif t == "Quoted":
            # c == [quote_type, inlines]
            parts.append(collect_inlines(c[1]))
        elif t == "Cite":
            # c == [citations, inlines]
            parts.append(collect_inlines(c[1]))
        elif t == "Span":
            # c == [attr, inlines]
            parts.append(collect_inlines(c[1]))
        elif t == "Math":
            # c == [math_type, text]
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
# 1b. Pandoc table parsing (pandoc <2.10, 2.10–2.19, and 3.x)
# ----------------------------------------------------------------------
def unwrap_ctor(node, tag):
    """
    Return the payload `c` of a tagged {t: tag, c: ...} node.
    Otherwise return the node unchanged (handles pandoc-types >= 1.23 flat
    records where TableHead/TableBody/Row/Cell are plain lists).
    """
    if isinstance(node, dict) and node.get("t") == tag:
        return node.get("c", [])
    return node


def extract_cell_text(cell):
    """Cell = [attr, align, rowspan, colspan, [Block]] (same shape old & new)."""
    cell = unwrap_ctor(cell, "Cell")
    if not isinstance(cell, list) or len(cell) < 5:
        return ""
    blocks = cell[4]
    return cell_text_from_blocks(blocks)


def extract_row_cells(row):
    """Row = [attr, [Cell]] or {"t":"Row","c":[attr,[Cell]]}."""
    row = unwrap_ctor(row, "Row")
    if not isinstance(row, list) or len(row) < 2:
        return []
    cells = row[1]
    if not isinstance(cells, list):
        return []
    return [extract_cell_text(c) for c in cells]


def extract_table_head_rows(head):
    """TableHead / TableFoot = [attr, [Row]]."""
    head = unwrap_ctor(head, "TableHead")
    if not isinstance(head, list) or len(head) < 2:
        return []
    rows = head[1]
    if not isinstance(rows, list):
        return []
    return [extract_row_cells(r) for r in rows]


def extract_table_body_rows(body):
    """TableBody = [attr, rowheadcols, [Row], [Row]]."""
    body = unwrap_ctor(body, "TableBody")
    if not isinstance(body, list) or len(body) < 4:
        return []
    intermediate = body[2] if isinstance(body[2], list) else []
    rows         = body[3] if isinstance(body[3], list) else []
    out = []
    for r in intermediate:
        out.append(extract_row_cells(r))
    for r in rows:
        out.append(extract_row_cells(r))
    return out


def extract_table_rows(table_c):
    if not isinstance(table_c, list):
        return []

    # ---- New format (pandoc >= 2.10): 6 fields -------------------------
    #   Table Attr Caption [ColSpec] TableHead [TableBody] TableFoot
    if len(table_c) >= 6:
        rows = []
        rows.extend(extract_table_head_rows(table_c[3]))   # head
        bodies = table_c[4]
        if isinstance(bodies, list):
            for body in bodies:
                rows.extend(extract_table_body_rows(body))
        rows.extend(extract_table_head_rows(table_c[5]))   # foot
        return rows

    # ---- Old format (pandoc < 2.10): 5 fields --------------------------
    #   Table [Inline] [Alignment] [Double] [TableCell] [[TableCell]]
    if len(table_c) >= 5:
        rows = []
        headers   = table_c[3]
        body_rows = table_c[4]
        if headers:
            rows.append([extract_cell_text(c) for c in headers])
        if isinstance(body_rows, list):
            for r in body_rows:
                rows.append([extract_cell_text(c) for c in r])
        return rows

    return []


def extract_tables(ast: dict) -> list:
    tables = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("t") == "Table":
                rows = extract_table_rows(node.get("c", []))
                if rows:
                    tables.append(rows)
                return                      # do not recurse into Table children
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(ast)
    return tables


# ----------------------------------------------------------------------
# 2. tmux script generation
# ----------------------------------------------------------------------
def generate_tmux_script(
    tables: list,
    session_name: str = "table_session",
    attach: bool = True,
) -> str:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    lines.append(f"tmux new-session -d -s {shlex.quote(session_name)}")
    lines.append("")

    for win_idx, table in enumerate(tables, start=1):
        win_name = f"table_{win_idx}"

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

        cmds = [cell.strip() for row in table for cell in row if cell.strip()]
        if not cmds:
            lines.append("# (empty table – no commands)")
            continue

        # First command goes into the initial pane (index 0).
        lines.append(
            f"tmux send-keys -t "
            f"{shlex.quote(session_name)}:{win_name}.0 "
            f"{shlex.quote(cmds[0])} C-m"
        )

        # Remaining commands: split + send.  Use -P -F '#{pane_id}' to get
        # the new pane's id reliably.
        for i, cmd in enumerate(cmds[1:], start=1):
            split_flag = "-h" if i % 2 == 1 else "-v"
            lines.append(
                f'PANE=$(tmux split-window {split_flag} '
                f'-t {shlex.quote(session_name)}:{win_name} '
                f'-P -F "#{{pane_id}}")'
            )
            lines.append(
                f'tmux send-keys -t "$PANE" {shlex.quote(cmd)} C-m'
            )

        lines.append(
            f"tmux select-layout -t "
            f"{shlex.quote(session_name)}:{win_name} tiled"
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
