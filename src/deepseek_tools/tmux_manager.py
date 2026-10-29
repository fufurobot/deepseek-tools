#!/usr/bin/env python3
"""
table_to_tmux.py – Convert HTML / reStructuredText tables to a tmux Bash script.

Works with pandoc < 2.10 (old Table AST) and pandoc >= 2.10 (new Table AST).

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
            # c == [inlines]
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
    return "".join(parts)


def cell_text_from_blocks(blocks: list) -> str:
    texts = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        bt = block.get("t")
        bc = block.get("c")
        if bt in ("Plain", "Para"):
            texts.append(collect_inlines(bc))
    return "\n".join(texts).strip()


def extract_cell_text(cell):
    """
    Works for:
      * pandoc >= 2.10:  Cell        = {"t": "Cell", "c": [attr, align, rowspan, colspan, [Block]]}
      * pandoc <  2.10:  TableCell   = [attr, align, rowspan, colspan, [Block]]
    """
    if isinstance(cell, dict) and cell.get("t") == "Cell":
        cell_c = cell.get("c", [])
        blocks = cell_c[4] if len(cell_c) > 4 else []
        return cell_text_from_blocks(blocks)
    elif isinstance(cell, list):
        # Old-format TableCell is a plain list with 5 elements
        blocks = cell[4] if len(cell) > 4 else []
        return cell_text_from_blocks(blocks)
    return ""


def extract_row_cells(row_node):
    """
    Works for:
      * pandoc >= 2.10:  Row     = {"t": "Row", "c": [attr, [Cell]]}
      * pandoc <  2.10:  a row   = [TableCell]
    """
    if isinstance(row_node, dict) and row_node.get("t") == "Row":
        row_c = row_node.get("c", [None, []])
        cells = row_c[1] if len(row_c) > 1 else []
        return [extract_cell_text(cell) for cell in cells]
    elif isinstance(row_node, list):
        return [extract_cell_text(cell) for cell in row_node]
    return []


def extract_table_rows(table_c: list) -> list[list[str]]:
    """Extract rows from the `c` field of a pandoc Table node."""
    # ---- New format (pandoc >= 2.10) ----------------------------------
    is_new = (
        len(table_c) >= 6
        and isinstance(table_c[3], dict)
        and table_c[3].get("t") == "TableHead"
    )
    if is_new:
        rows = []
        # Head
        head_c = table_c[3].get("c", [None, []])
        head_rows = head_c[1] if len(head_c) > 1 else []
        for r in head_rows:
            rows.append(extract_row_cells(r))
        # Bodies
        for body in table_c[4]:
            if not isinstance(body, dict):
                continue
            body_c = body.get("c", [])
            # TableBody c == [attr, row_head_columns, [Row], [Row]]
            intermediate = body_c[2] if len(body_c) > 2 else []
            body_rows    = body_c[3] if len(body_c) > 3 else []
            for r in intermediate:
                rows.append(extract_row_cells(r))
            for r in body_rows:
                rows.append(extract_row_cells(r))
        # Foot (optional)
        if len(table_c) > 5 and isinstance(table_c[5], dict):
            foot_c = table_c[5].get("c", [None, []])
            foot_rows = foot_c[1] if len(foot_c) > 1 else []
            for r in foot_rows:
                rows.append(extract_row_cells(r))
        return rows

    # ---- Old format (pandoc < 2.10) -----------------------------------
    if len(table_c) >= 5:
        rows = []
        headers   = table_c[3]     # [TableCell]
        body_rows = table_c[4]     # [[TableCell]]
        if headers:
            rows.append([extract_cell_text(c) for c in headers])
        for r in body_rows:
            rows.append([extract_cell_text(c) for c in r])
        return rows

    return []


def extract_tables(ast: dict) -> list[list[list[str]]]:
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
    tables: list[list[list[str]]],
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
        # the new pane's id reliably (the new pane is active anyway, but
        # being explicit avoids any target ambiguity).
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
