#!/usr/bin/env python3
"""Show per-job conclusions for the most recent workflow run."""

from __future__ import annotations

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
    repo = os.environ.get("GITHUB_REPOSITORY", "fufurobot/deepseek-tools")
    headers = {
        "Authorization": f"Bearer {os.environ.get('GITHUB_TOKEN', '')}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "deepseek-tools-ci",
    }

    runs = requests.get(
        f"{API}/repos/{repo}/actions/runs",
        headers=headers, params={"per_page": 3}, timeout=30,
    ).json().get("workflow_runs", [])

    for run in runs:
        print(f"run #{run['run_number']} [{run['conclusion']}] sha={run['head_sha'][:8]}")
        jobs = requests.get(
            f"{API}/repos/{repo}/actions/runs/{run['id']}/jobs",
            headers=headers, timeout=30,
        ).json().get("jobs", [])
        for job in jobs:
            print(f"    [{job['conclusion'] or job['status']:>9}] {job['name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
