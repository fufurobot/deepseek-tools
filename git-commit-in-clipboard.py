#!/usr/bin/env python3
"""
git-commit-from-clipboard.py
Reads the clipboard and commits with that message.
If nothing to commit, runs `git commit --amend` instead.
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
        # First attempt: normal commit
        result = subprocess.run(
            ["git", "commit", "-F", temp_path],
            capture_output=True,
            text=True
        )

        # If successful, print and exit
        if result.returncode == 0:
            print("Commit successful:")
            print(result.stdout)
            return
        print("Commit Failed")
        print(result.stdout)
        # Check if the failure is because there's nothing to commit
        stderr = result.stderr.lower()
        if "nothing to commit" in stderr or "nothing added to commit" in stderr:
            print("Nothing to commit – amending last commit instead.")
            # Retry with --amend
            amend_result = subprocess.run(
                ["git", "commit", "--amend", "-F", temp_path],
                capture_output=True,
                text=True
            )
            if amend_result.returncode == 0:
                print("Amend successful:")
                print(amend_result.stdout)
                return
            else:
                print("Amend failed:")
                print(amend_result.stderr, file=sys.stderr)
                sys.exit(amend_result.returncode)
        else:
            # Some other error
            print("Commit failed:")
            print(result.stderr, file=sys.stderr)
            sys.exit(result.returncode)

    finally:
        # Remove the temporary file
        os.unlink(temp_path)


if __name__ == "__main__":
    main()