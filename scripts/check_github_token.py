#!/usr/bin/env python3
"""Print the GitHub token's identity and effective fine-grained permissions.

Reads GITHUB_TOKEN from the environment or the local .env file (never prints it).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

API = "https://api.github.com"


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
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("GITHUB_TOKEN is not set (checked environment and .env)")
        return 1

    print(f"token length: {len(token)}  prefix: {token[:12]}...")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "deepseek-tools-status",
    }

    r = requests.get(f"{API}/user", headers=headers, timeout=30)
    print(f"=== GET /user -> {r.status_code} ===")
    if r.status_code != 200:
        print(r.text[:500])
        return 1
    user = r.json()
    print(f"login            : {user.get('login')}")
    print(f"id               : {user.get('id')}")
    print(f"type             : {user.get('type')}")
    print(f"X-OAuth-Scopes   : {r.headers.get('X-OAuth-Scopes')!r} (empty => fine-grained PAT)")
    print(f"accepted scopes  : {r.headers.get('X-Accepted-OAuth-Scopes')!r}")

    repo = os.environ.get("GITHUB_REPOSITORY", "fufurobot/deepseek-tools")
    r = requests.get(f"{API}/repos/{repo}", headers=headers, timeout=30)
    print(f"\n=== GET /repos/{repo} -> {r.status_code} ===")
    if r.status_code == 200:
        data = r.json()
        perms = data.get("permissions", {})
        print(f"full_name        : {data.get('full_name')}")
        print(f"private          : {data.get('private')}")
        print(f"default_branch   : {data.get('default_branch')}")
        print(f"permissions      : {json.dumps(perms)}")
        print(f"has_issues       : {data.get('has_issues')}")
    else:
        print(r.text[:500])

    # Probe the capability layers the user selected.
    probes = [
        ("Issues (read)", "GET", f"/repos/{repo}/issues?per_page=1"),
        ("Pull requests (read)", "GET", f"/repos/{repo}/pulls?per_page=1"),
        ("Actions/CI (read)", "GET", f"/repos/{repo}/actions/runs?per_page=1"),
        ("Workflows (read)", "GET", f"/repos/{repo}/actions/workflows"),
        ("Contents (read)", "GET", f"/repos/{repo}/contents/pyproject.toml"),
    ]
    print("\n=== capability probe ===")
    for label, method, path in probes:
        resp = requests.request(method, f"{API}{path}", headers=headers, timeout=30)
        note = ""
        if resp.status_code == 403:
            note = "  <- FORBIDDEN (missing permission)"
        elif resp.status_code == 404:
            note = "  <- NOT FOUND (missing permission or absent)"
        print(f"{label:24s} {resp.status_code}{note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
