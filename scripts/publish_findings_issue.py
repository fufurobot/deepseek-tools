#!/usr/bin/env python3
"""Publish the sandbox and cross-platform findings as a GitHub issue."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import requests

API = "https://api.github.com"

TITLE = "pytest in a confined Windows sandbox, and three real git_cloner bugs it hid"

BODY = """\
## Summary

Running `pytest` in this repository from a confined Windows file sandbox
(`workspace-write`) failed in a way that looked like broken project
permissions. It was not. Fixing it exposed three genuine bugs in
`GitRepoCloner` that the local environment had been masking.

Everything below was solved **without relaxing sandbox permissions**.

## 1. pytest cannot create temp directories in a confined sandbox

`tmp_path` failed during fixture setup:

```
PermissionError: [WinError 5] Access is denied:
'...\\AppData\\Local\\Temp\\...\\pytest-of-<user>'
```

**Cause.** pytest creates temp directories with `mode=0o700`. On Windows that
yields an owner-only ACL: a confined token creates such a directory
successfully but can **never enumerate it again**. Because
`make_numbered_dir` calls `find_suffixes(root, prefix)`, which lists the shared
root on every call, one poisoned directory breaks all later runs in that root.
A second factor is that `pytest_configure` runs *before* the sandbox grants
workspace write access, so a basetemp created there is born unlistable.

An ACL inspection of the failing path returned `NOT_THIS_CLASS` — owned by the
current user with full control — confirming confinement rather than a
permissions defect.

**Fix.** `tests/conftest.py` patches three pytest internals, on Windows only:
`make_numbered_dir` (inherited permissions, monotonic counter instead of
scanning the root), `TempPathFactory.getbasetemp` (lazy creation), and
`cleanup_dead_symlinks`/`rm_rf` (best-effort teardown).

## 2. `parse_repo_url`: scp-style SSH URLs

`git@github.com:user/repo.git` kept the `git@github.com:` prefix in the
namespace, so the SSH and HTTPS forms of one repository resolved to two
different local directories.

## 3. `parse_repo_url`: local paths became namespaces

Only visible on Linux, where `git clone` of a local repository succeeds and the
Windows run had been skipping past the assertion:

- `file:///tmp/a/b/origin` cloned to `<base_dir>/tmp/a/b/origin` instead of
  `<base_dir>/origin`, so layout depended on where the source lived.
- `Path(base_dir) / "/abs/path"` **discards `base_dir` entirely** on POSIX, so
  a clone could escape the configured base directory.

## 4. `clone_or_update_repo` retried forever

`while True` with no exit path. A repository that can never be cloned — bad
URL, no network, or a sandbox rejecting `git clone` — spun indefinitely,
hanging every caller. Now bounded by an opt-in `max_retries` / `--max-retries`;
the default stays unbounded for backwards compatibility.

Also fixed: a partially created destination directory was left behind after a
failed clone, blocking every later attempt with "destination path already
exists".

## Result

Suite: **24 passed, 1 skipped** locally, and CI green on all four matrix jobs
(Linux + Windows x Python 3.10 + 3.12).

## Takeaway

Three of these four bugs were only reachable because the two platforms fail
differently: Windows could not clone at all, Linux cloned successfully into the
wrong place. A CI matrix earned its keep here — a single-platform run would
have shipped the namespace bug.
"""


def load_env(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def main() -> int:
    load_env()
    repo = os.environ.get("GITHUB_REPOSITORY", "fufurobot/deepseek-tools")
    headers = {
        "Authorization": f"Bearer {os.environ.get('GITHUB_TOKEN', '')}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "deepseek-tools",
    }

    response = requests.post(
        f"{API}/repos/{repo}/issues",
        headers=headers,
        json={"title": TITLE, "body": BODY},
        timeout=30,
    )
    if response.status_code != 201:
        print(f"Could not create issue: {response.status_code}")
        print(response.text[:400])
        return 1

    issue = response.json()
    print(f"Created issue #{issue['number']}: {issue['html_url']}")

    close = requests.patch(
        f"{API}/repos/{repo}/issues/{issue['number']}",
        headers=headers,
        json={"state": "closed", "state_reason": "completed"},
        timeout=30,
    )
    print(f"Closed: {close.status_code} state={close.json().get('state')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
