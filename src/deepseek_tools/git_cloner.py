#!/usr/bin/env python3
"""
Git Repo Cloner - A CLI tool to clone multiple git repositories with namespace isolation
and infinite retry capability.
"""

import os
import re
import shutil
import sys
import subprocess
import time
import argparse
from urllib.parse import urlparse
from pathlib import Path
from typing import List, Optional, Tuple


class GitRepoCloner:
    def __init__(self, base_dir: str = "repos", retry_delay: int = 5,
                 max_retries: Optional[int] = None):
        """
        Initialize the GitRepoCloner.

        Args:
            base_dir: Base directory where repos will be cloned
            retry_delay: Delay in seconds between retries
            max_retries: Maximum attempts per repository. ``None`` (the
                default) preserves the historical retry-forever behaviour;
                pass a positive integer to fail fast instead of hanging.
        """
        self.base_dir = Path(base_dir)
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _should_retry(self, attempt: int) -> bool:
        """Return True when another attempt is allowed for *attempt*."""
        if self.max_retries is None:
            return True
        return attempt < self.max_retries

    def parse_repo_url(self, url: str) -> Tuple[str, str]:
        """
        Parse git URL to extract namespace and repository name.

        Handles standard URLs, scp-style SSH shorthand and local paths, so
        that each remote resolves to one predictable local directory.

        Args:
            url: Git repository URL (HTTPS, SSH, scp-style or local path)

        Returns:
            Tuple of (namespace_path, repo_name). Local paths and single-name
            remotes have an empty namespace.
        """
        url = url.strip()

        # Whether this URL names a local directory rather than a network
        # remote. Local repositories are never namespaced: see below.
        is_local = "://" not in url or url.startswith("file://")

        if "://" not in url:
            # A Windows drive path (C:\src\origin) is local, not scp-style;
            # the drive letter would otherwise be read as a host name.
            drive_path = bool(re.match(r"^[A-Za-z]:[\\/]", url))
            # scp-style shorthand: [user@]host:path/to/repo.git
            # `urlparse` does not understand this form and would fold the whole
            # "user@host:" prefix into the path, so it is handled explicitly.
            match = None if drive_path else re.match(
                r"^(?:[^@/\\]+@)?[^:/\\]+:(?P<path>.+)$", url
            )
            if match:
                path = match.group("path")
                is_local = False
            else:
                # A bare local path such as ./repos/origin or C:\src\origin.
                path = url
        else:
            path = urlparse(url).path

        # Record absoluteness before stripping separators, which would
        # otherwise erase the leading "/" that identifies it.
        raw = path.replace("\\", "/").strip()
        is_absolute = raw.startswith("/") or bool(re.match(r"^[A-Za-z]:", raw))

        # Normalise separators and trailing punctuation.
        path = raw
        if path.endswith(".git"):
            path = path[:-4]
        path = path.strip("/")

        if not path:
            return "", ""

        parts = [p for p in path.split("/") if p and p != "."]

        if not parts:
            return "", ""
        if len(parts) == 1:
            # Just a repo name, no namespace
            return "", parts[0]

        # Local paths are not namespaces: using the directory tree as one would
        # make the target depend on the machine, and an absolute path joined
        # with base_dir would escape base_dir entirely.
        if is_local and (is_absolute or raw.startswith(".")):
            return "", parts[-1]

        # Namespace is everything except the last part (repo name)
        return "/".join(parts[:-1]), parts[-1]

    def get_repo_path(self, url: str) -> Path:
        """
        Get the local filesystem path for a repository.
        
        Args:
            url: Git repository URL
            
        Returns:
            Path object for the local repository
        """
        namespace, repo_name = self.parse_repo_url(url)
        
        # Build path: base_dir/namespace/repo_name
        if namespace:
            repo_path = self.base_dir / namespace / repo_name
        else:
            repo_path = self.base_dir / repo_name
        
        return repo_path

    def run_git_command(self, cmd: List[str], cwd: Optional[Path] = None) -> Tuple[bool, str]:
        """
        Run a git command and return the result.
        
        Args:
            cmd: Git command as list of strings
            cwd: Working directory for the command
            
        Returns:
            Tuple of (success, output/error message)
        """
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr
        except Exception as e:
            return False, str(e)

    def clone_or_update_repo(self, url: str) -> bool:
        """
        Clone or update a repository, retrying on failure.

        Args:
            url: Git repository URL

        Returns:
            True if successful. False only when ``max_retries`` is set and all
            attempts are exhausted; with the default (unbounded) setting this
            never returns False.
        """
        repo_path = self.get_repo_path(url)

        # Check if repository already exists
        if repo_path.exists():
            print(f"Repository exists at {repo_path}, pulling latest changes...")

            # Check if it's a git repository
            git_dir = repo_path / '.git'
            if not git_dir.exists():
                print(f"❌ {repo_path} exists but is not a git repository. Please remove it manually.")
                return False

            attempt = 1
            while True:
                print(f"  Attempt {attempt}: git pull")
                success, output = self.run_git_command(['git', 'pull'], cwd=repo_path)
                if success:
                    print(f"✅ Successfully updated {url} at {repo_path}")
                    return True

                print(f"❌ Pull failed: {output.strip()}")
                if not self._should_retry(attempt):
                    print(f"❌ Giving up on {url} after {attempt} attempt(s).")
                    return False
                print(f"⏳ Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
                attempt += 1
        else:
            # Clone with retry
            print(f"Cloning {url} to {repo_path}...")

            # Create parent directories if they don't exist
            repo_path.parent.mkdir(parents=True, exist_ok=True)

            attempt = 1
            while True:
                print(f"  Attempt {attempt}: git clone")
                success, output = self.run_git_command(['git', 'clone', url, str(repo_path)])
                if success:
                    print(f"✅ Successfully cloned {url} to {repo_path}")
                    return True

                print(f"❌ Clone failed: {output.strip()}")
                # A partially created destination blocks every later attempt.
                if repo_path.exists() and not (repo_path / '.git').exists():
                    shutil.rmtree(repo_path, ignore_errors=True)
                if not self._should_retry(attempt):
                    print(f"❌ Giving up on {url} after {attempt} attempt(s).")
                    return False
                print(f"⏳ Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
                attempt += 1

    def process_repos(self, repo_urls: List[str]) -> None:
        """
        Process a list of repository URLs.
        
        Args:
            repo_urls: List of git repository URLs
        """
        if not repo_urls:
            print("No repositories to process")
            return
        
        print(f"Processing {len(repo_urls)} repositories...")
        print(f"Base directory: {self.base_dir.absolute()}")
        print("-" * 60)
        
        for i, url in enumerate(repo_urls, 1):
            url = url.strip()
            if not url or url.startswith('#'):
                continue
                
            print(f"\n[{i}/{len(repo_urls)}] Processing: {url}")
            try:
                self.clone_or_update_repo(url)
            except KeyboardInterrupt:
                print("\n⚠️  Interrupted by user")
                sys.exit(1)
            except Exception as e:
                print(f"❌ Unexpected error processing {url}: {e}")
                # Continue with next repo despite error
        
        print("\n" + "=" * 60)
        print("✅ All repositories processed!")

    def parse_urls_from_stdin(self) -> List[str]:
        """Read repository URLs from stdin."""
        urls = []
        for line in sys.stdin:
            line = line.strip()
            if line and not line.startswith('#'):
                urls.append(line)
        return urls

    def parse_urls_from_file(self, filepath: str) -> List[str]:
        """Read repository URLs from a file."""
        try:
            with open(filepath, 'r') as f:
                urls = []
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        urls.append(line)
                return urls
        except FileNotFoundError:
            print(f"❌ Error: File '{filepath}' not found")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error reading file '{filepath}': {e}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Clone multiple git repositories with namespace isolation and infinite retry',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Read URLs from stdin (one per line)
  echo "https://github.com/user/repo1.git" | git-clone-multi
  
  # Read URLs from a file
  git-clone-multi -f repos.txt
  
  # Specify custom base directory
  git-clone-multi -f repos.txt -o ~/my-repos
  
  # Read from stdin with custom retry delay
  git-clone-multi -d 10

Input file format (one URL per line, lines starting with # are ignored):
  https://github.com/org/project.git
  git@github.com:user/repo.git
  https://gitlab.com/group/subgroup/project.git
        """
    )
    
    parser.add_argument(
        '-f', '--file',
        help='Input file containing repository URLs (one per line)'
    )
    parser.add_argument(
        '-o', '--output',
        default='repos',
        help='Base output directory (default: repos)'
    )
    parser.add_argument(
        '-d', '--delay',
        type=int,
        default=5,
        help='Retry delay in seconds (default: 5)'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=None,
        help='Maximum attempts per repository (default: retry forever)'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='git-clone-multi 1.0.0'
    )
    
    args = parser.parse_args()
    
    # Read repository URLs
    if args.file:
        urls = GitRepoCloner.parse_urls_from_file(None, args.file)
    else:
        # Read from stdin
        if sys.stdin.isatty():
            print("❌ Error: No input provided. Use -f to specify a file or pipe URLs to stdin.")
            parser.print_help()
            sys.exit(1)
        urls = GitRepoCloner.parse_urls_from_stdin(None)
    
    if not urls:
        print("❌ Error: No valid repository URLs found")
        sys.exit(1)
    
    # Create cloner instance and process repositories
    cloner = GitRepoCloner(
        base_dir=args.output,
        retry_delay=args.delay,
        max_retries=args.max_retries,
    )
    cloner.process_repos(urls)


if __name__ == '__main__':
    main()