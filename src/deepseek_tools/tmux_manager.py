#!/usr/bin/env python3
"""
table_to_tmux.py – Convert HTML / reStructuredText tables to a tmux Bash script.

Usage:
    python table_to_tmux.py input.html            # writes to stdout
    python table_to_tmux.py input.rst -o setup.sh
    cat table.html | python table_to_tmux.py -
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
    """
    Call pandoc to convert *text* (in *from_format*) to its JSON AST.
    """
    cmd = [
        "pandoc",
        "--from", from_format,
        "--to", "json",
        "--output", "-",          # write to stdout
        "-",                      # read from stdin
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
    """
    Recursively extract plain text from a list of Pandoc Inline elements.
    Handles Str, Space, SoftBreak, LineBreak, Emph, Strong, Link, etc.
    """
    parts = []
    for inline in inlines:
        t = inline.get("t")
        c = inline.get("c")
        if t in ("Str", "Code"):
            parts.append(c)
        elif t == "Space":
            parts.append(" ")
        elif t in ("SoftBreak", "LineBreak"):
            parts.append("\n")
        elif t in ("Emph", "Strong", "Strikeout", "Superscript", "Subscript"):
            parts.append(collect_inlines(c))
        elif t == "Link":
            # c = [attr, inlines, target]
            parts.append(collect_inlines(c[1]))
        elif t == "Image":
            # ignore images
            pass
        elif t == "Quoted":
            # c = [quote_type, inlines]
            parts.append(collect_inlines(c[1]))
        elif t == "Cite":
            # c = [citations, inlines]
            parts.append(collect_inlines(c[1]))
        # other inline types can be added here if needed
    return "".join(parts)


def cell_text(cell: dict) -> str:
    """
    Extract the plain‑text content of a table cell.
    A cell is a list of Blocks (usually Plain or Para).
    """
    blocks = cell.get("c", [])
    texts = []
    for block in blocks:
        bt = block.get("t")
        bc = block.get("c")
        if bt in ("Plain", "Para"):
            texts.append(collect_inlines(bc))
        # other block types (e.g. BulletList) are ignored for simplicity
    return "\n".join(texts).strip()


# ----------------------------------------------------------------------
# 2. Table extraction
# ----------------------------------------------------------------------
def extract_tables(ast: dict) -> list[list[list[str]]]:
    """
    Walk the Pandoc AST and return a list of tables.
    Each table is a 2‑D list of cell strings (rows × columns).
    """
    tables = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("t") == "Table":
                # Pandoc's Table constructor (since pandoc 2.10):
                # Table Attr Caption [ColSpec] TableHead [TableBody] TableFoot
                # The actual rows are in the TableBody(s) and TableHead.
                # For simplicity we collect all rows from head and bodies.
                table_data = node["c"]
                # index 3: TableHead, index 4: list of TableBody
                head = table_data[3]
                bodies = table_data[4]

                rows = []
                # ---- header row(s) ----
                head_rows = head.get("c", [])  # list of rows
                for row in head_rows:
                    # row is a list of cells
                    rows.append([cell_text(cell) for cell in row])
                # ---- body rows ----
                for body in bodies:
                    body_rows = body.get("c", [])  # list of rows
                    for row in body_rows:
                        rows.append([cell_text(cell) for cell in row])

                tables.append(rows)
                # do not recurse into the table’s children – we already processed it
                return

            # recurse into other dicts
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(ast)
    return tables


# ----------------------------------------------------------------------
# 3. tmux script generation
# ----------------------------------------------------------------------
def generate_tmux_script(
    tables: list[list[list[str]]],
    session_name: str = "table_session",
    attach: bool = True,
) -> str:
    """
    Build a Bash script that creates a tmux session with one window per table.
    Each cell becomes a pane that runs the cell’s command.
    """
    lines = []
    lines.append("#!/usr/bin/env bash")
    lines.append("set -euo pipefail")
    lines.append("")

    # Create the session in detached mode.
    lines.append(f"tmux new-session -d -s {shlex.quote(session_name)}")
    lines.append("")

    for win_idx, table in enumerate(tables, start=1):
        win_name = f"table_{win_idx}"
        # First window already exists (window 0), so we create a new window for
        # every table except the first.
        if win_idx == 1:
            # Rename the initial window
            lines.append(f"tmux rename-window -t {shlex.quote(session_name)}:0 {shlex.quote(win_name)}")
        else:
            lines.append(
                f"tmux new-window -t {shlex.quote(session_name)} -n {shlex.quote(win_name)}"
            )

        # Flatten the table into a list of commands (row‑major order).
        cmds = []
        for row in table:
            for cell in row:
                cmd = cell.strip()
                if cmd:                         # ignore empty cells
                    cmds.append(cmd)

        if not cmds:
            lines.append(f"# (empty table – no commands)")
            continue

        # The first command goes into the current pane.
        lines.append(f"tmux send-keys -t {shlex.quote(session_name)}:{win_name}.0 {shlex.quote(cmds[0])} C-m")

        # Create additional panes for the remaining commands.
        # We alternate horizontal and vertical splits, then tile them.
        for i, cmd in enumerate(cmds[1:], start=1):
            if i % 2 == 1:
                split_flag = "-h"   # horizontal split
            else:
                split_flag = "-v"   # vertical split
            lines.append(
                f"tmux split-window {split_flag} -t {shlex.quote(session_name)}:{win_name} "
                f"-c '#{{pane_current_path}}'"
            )
            # Send the command to the newly created pane.
            # (The pane index of the new pane is not fixed, so we use the
            #  'last' target which refers to the most recently created pane.)
            lines.append(
                f"tmux send-keys -t {shlex.quote(session_name)}:{win_name}.+ "
                f"{shlex.quote(cmd)} C-m"
            )

        # Arrange all panes in a grid.
        lines.append(f"tmux select-layout -t {shlex.quote(session_name)}:{win_name} tiled")
        lines.append("")

    # Attach to the session unless the user asked for a detached script.
    if attach:
        lines.append(f"tmux attach-session -t {shlex.quote(session_name)}")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# 4. Command‑line interface
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Convert HTML / reStructuredText tables to a tmux Bash script."
    )
    parser.add_argument(
        "input",
        help="Input file (use '-' for stdin). The format is inferred from the extension; "
             "for stdin you must specify --from.",
    )
    parser.add_argument(
        "--from",
        dest="from_format",
        default=None,
        help="Pandoc input format (e.g. html, rst). Required when reading from stdin.",
    )
    parser.add_argument(
        "-o", "--output",
        default="-",
        help="Output file (default: stdout).",
    )
    parser.add_argument(
        "--session",
        default="table_session",
        help="tmux session name (default: table_session).",
    )
    parser.add_argument(
        "--no-attach",
        action="store_true",
        help="Do not append 'tmux attach-session' to the generated script.",
    )
    args = parser.parse_args()

    # Read the input.
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

    # Convert to Pandoc JSON.
    ast = pandoc_json(text, from_format)

    # Extract tables.
    tables = extract_tables(ast)
    if not tables:
        sys.exit("No tables found in the input.")

    # Build the tmux script.
    script = generate_tmux_script(
        tables,
        session_name=args.session,
        attach=not args.no_attach,
    )

    # Write the result.
    if args.output == "-":
        sys.stdout.write(script)
    else:
        Path(args.output).write_text(script, encoding="utf-8")
        # Make the generated script executable.
        Path(args.output).chmod(0o755)


if __name__ == "__main__":
    main()
