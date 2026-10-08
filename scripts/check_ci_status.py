#!/usr/bin/env python3
"""Report GitHub Actions workflow runs for this repository.

Reads GITHUB_TOKEN from the environment or the local .env file. Optionally
waits for in-progress runs to settle.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
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


def headers() -> dict[str, str]:
    token = os.environ.get("GITHUB_TOKEN", "")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "deepseek-tools-ci",
    }


def fetch_runs(repo: str, limit: int = 5) -> list[dict]:
    response = requests.get(
        f"{API}/repos/{repo}/actions/runs",
        headers=headers(),
        params={"per_page": limit},
        timeout=30,
    )
    if response.status_code != 200:
        print(f"Failed to read runs: {response.status_code} {response.text[:300]}")
        return []
    return response.json().get("workflow_runs", [])


def describe(run: dict) -> str:
    status = run.get("status")
    conclusion = run.get("conclusion")
    marker = {"success": "PASS", "failure": "FAIL", "cancelled": "CANCEL"}.get(
        conclusion or "", conclusion or status or "?"
    )
    return (
        f"  [{marker:>6}] {run.get('name')} #{run.get('run_number')} "
        f"branch={run.get('head_branch')} event={run.get('event')} "
        f"sha={(run.get('head_sha') or '')[:8]}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", type=int, default=0,
                        help="Seconds to keep polling while runs are in progress")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    load_env()
    repo = os.environ.get("GITHUB_REPOSITORY", "fufurobot/deepseek-tools")
    if not os.environ.get("GITHUB_TOKEN"):
        print("GITHUB_TOKEN is not set")
        return 1

    deadline = time.time() + args.wait
    while True:
        runs = fetch_runs(repo, args.limit)
        if not runs:
            print("No workflow runs found.")
            return 1

        in_progress = [r for r in runs if r.get("status") != "completed"]
        print(f"runs for {repo} (in progress: {len(in_progress)}):")
        for run in runs:
            print(describe(run))

        if not in_progress or time.time() >= deadline:
            return 0
        print(f"  ... waiting ({int(deadline - time.time())}s left)")
        time.sleep(15)


if __name__ == "__main__":
    sys.exit(main())
