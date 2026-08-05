#!/usr/bin/env python3
"""
git-commit-from-clipboard.py
Reads the clipboard and commits with that message.
Usage: python git-commit-from-clipboard.py
"""

import subprocess
import tempfile
import os
import sys

try:
    import pyperclip
except ImportError:
    print("Error: pyperclip is not installed. Run: pip install pyperclip")
    sys.exit(1)


def main():
    # Get clipboard content
    message = pyperclip.paste()
    if not message or message.strip() == "":
        print("Error: Clipboard is empty or contains only whitespace.")
        sys.exit(1)

    # Write message to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(message)
        temp_path = f.name

    try:
        # Run git commit with the temp file as the message source
        result = subprocess.run(
            ["git", "commit", "-F", temp_path],
            capture_output=True,
            text=True
        )
        # Print output
        if result.returncode == 0:
            print("Commit successful:")
            print(result.stdout)
        else:
            print("Commit failed:")
            print(result.stderr, file=sys.stderr)
            sys.exit(result.returncode)
    finally:
        # Remove the temporary file
        os.unlink(temp_path)


if __name__ == "__main__":
    main()
