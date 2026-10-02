#!/usr/bin/env python3
"""
pwcodegen.py — Run Playwright codegen with a GitHub-style random isolated
user-data-dir under the current directory.

Usage:
    python pwcodegen.py [URL] [OUTPUT] [--keep] [--name NAME] [--browser BROWSER]
                        [--no-cleanup-on-error] [--extra-arg ARG ...]

Examples:
    python pwcodegen.py
    python pwcodegen.py https://example.com out.py
    python pwcodegen.py https://example.com out.py --keep
    python pwcodegen.py --name my-custom-profile https://example.com
    python pwcodegen.py --browser firefox https://example.com
"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# GitHub-style random name generator (community-researched approximation)
# ---------------------------------------------------------------------------

PREFIXES = [
    "refactored", "reimagined", "stunning", "sturdy", "solid", "cuddly",
    "literate", "animated", "silver", "psychic", "congenial", "crispy",
    "special", "legendary", "expert", "friendly", "shiny", "curly",
    "bug-free", "didactic", "cautious", "laughing", "redesigned",
    "fictional", "improved", "glowing", "bookish", "probable", "upgraded",
    "supreme", "ideal", "miniature", "effective", "efficient", "urban",
    "turbo", "symmetrical", "fluffy", "smooth", "solid", "vigilant",
]

SUFFIXES = [
    "pancake", "guide", "succotash", "lamp", "happiness",
    "computing-machine", "carnival", "umbrella", "system", "waddle",
    "potato", "palm-tree", "parakeet", "guacamole", "eureka", "fortnight",
    "chainsaw", "garbanzo", "winner", "goggles", "memory",
    "robot", "doodle", "funicular", "giggle", "fiesta",
    "disco", "waffle", "bassoon", "couscous", "broccoli",
    "spork", "tribble", "telegram", "engine", "barnacle",
]

INFIXES = ["octo"]  # GitHub's signature infix
INFIX_PROBABILITY = 1 / 11  # ~1 in 11 names get the octo infix


def github_style_name() -> str:
    """Generate a random name in GitHub's repository-name style."""
    prefix = random.choice(PREFIXES)
    suffix = random.choice(SUFFIXES)
    if random.random() < INFIX_PROBABILITY:
        return f"{prefix}-{random.choice(INFIXES)}-{suffix}"
    return f"{prefix}-{suffix}"


def unique_profile_dir(base: Path, name: str | None = None) -> Path:
    """Create a unique profile directory under `base`."""
    base.mkdir(parents=True, exist_ok=True)
    stem = name or github_style_name()
    candidate = base / f"pw_profile_{stem}"
    counter = 1
    while candidate.exists():
        candidate = base / f"pw_profile_{stem}-{counter}"
        counter += 1
    candidate.mkdir()
    return candidate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pwcodegen",
        description=(
            "Run Playwright codegen with a random isolated --user-data-dir "
            "under the current directory. Profile is deleted after recording "
            "unless --keep is passed."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "url",
        nargs="?",
        default="https://example.com",
        help="Starting URL (default: https://example.com)",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="generated_script.py",
        help="Output Python file for generated code (default: generated_script.py)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Fixed profile name instead of a random GitHub-style one",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep the profile directory after recording",
    )
    parser.add_argument(
        "--browser",
        choices=["chromium", "firefox", "webkit"],
        default="chromium",
        help="Browser to record with (default: chromium)",
    )
    parser.add_argument(
        "--profile-root",
        default=".",
        help="Directory under which the profile folder is created (default: cwd)",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra argument passed through to `playwright codegen` (repeatable)",
    )
    return parser.parse_args(argv)


def run_codegen(
    url: str,
    output: str,
    profile_dir: Path,
    browser: str,
    extra_args: list[str],
) -> int:
    cmd = [
        sys.executable, "-m", "playwright", "codegen",
        "-o", output,
        "--user-data-dir", str(profile_dir),
        "--browser", browser,
        *extra_args,
        url,
    ]
    print("▶ Command:")
    print("   " + " ".join(cmd))
    try:
        return subprocess.run(cmd, check=False).returncode
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user.")
        return 130


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    profile_root = Path(args.profile_root).resolve()
    profile_dir = unique_profile_dir(profile_root, args.name)

    print(f"▶ Profile dir: {profile_dir}")
    print(f"▶ Output file: {args.output}")
    print(f"▶ Target URL : {args.url}")
    print(f"▶ Browser    : {args.browser}")

    exit_code = 1
    try:
        exit_code = run_codegen(
            url=args.url,
            output=args.output,
            profile_dir=profile_dir,
            browser=args.browser,
            extra_args=args.extra_arg,
        )
    finally:
        if args.keep:
            print(f"📁 Kept profile: {profile_dir}")
        else:
            shutil.rmtree(profile_dir, ignore_errors=True)
            print(f"🧹 Removed profile: {profile_dir}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())