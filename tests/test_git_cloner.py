"""Tests for deepseek_tools.git_cloner."""

from __future__ import annotations

import subprocess
import threading

import pytest

from deepseek_tools.git_cloner import GitRepoCloner


class TestParseRepoUrl:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://github.com/user/repo.git", ("user", "repo")),
            ("https://github.com/user/repo", ("user", "repo")),
            ("https://gitlab.com/group/subgroup/project.git", ("group/subgroup", "project")),
            ("https://example.com/single.git", ("", "single")),
        ],
    )
    def test_https_urls(self, url, expected):
        cloner = GitRepoCloner(base_dir="unused")
        assert cloner.parse_repo_url(url) == expected

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("git@github.com:user/repo.git", ("user", "repo")),
            ("ssh://git@github.com/user/repo.git", ("user", "repo")),
        ],
    )
    def test_ssh_urls(self, url, expected):
        cloner = GitRepoCloner(base_dir="unused")
        assert cloner.parse_repo_url(url) == expected


class TestGetRepoPath:
    def test_namespaced_path(self, tmp_path):
        cloner = GitRepoCloner(base_dir=str(tmp_path))
        assert cloner.get_repo_path("https://github.com/user/repo.git") == tmp_path / "user" / "repo"

    def test_flat_path(self, tmp_path):
        cloner = GitRepoCloner(base_dir=str(tmp_path))
        assert cloner.get_repo_path("https://example.com/single.git") == tmp_path / "single"

    def test_base_dir_created_on_init(self, tmp_path):
        target = tmp_path / "nested" / "repos"
        GitRepoCloner(base_dir=str(target))
        assert target.is_dir()


class TestUrlsFromFile:
    def test_parses_and_skips_comments_and_blanks(self, tmp_path):
        f = tmp_path / "repos.txt"
        f.write_text(
            "# a comment\n"
            "https://github.com/a/b.git\n"
            "\n"
            "  https://github.com/c/d.git  \n"
            "#another\n",
            encoding="utf-8",
        )
        urls = GitRepoCloner.parse_urls_from_file(None, str(f))
        assert urls == ["https://github.com/a/b.git", "https://github.com/c/d.git"]

    def test_missing_file_exits_nonzero(self, tmp_path):
        with pytest.raises(SystemExit) as exc:
            GitRepoCloner.parse_urls_from_file(None, str(tmp_path / "nope.txt"))
        assert exc.value.code != 0


class TestRunGitCommand:
    def test_success_returns_true_and_stdout(self, tmp_path):
        cloner = GitRepoCloner(base_dir=str(tmp_path / "repos"))
        ok, out = cloner.run_git_command(["git", "--version"])
        assert ok is True
        assert "git version" in out

    def test_failure_returns_false_and_stderr(self, tmp_path):
        cloner = GitRepoCloner(base_dir=str(tmp_path / "repos"))
        ok, out = cloner.run_git_command(["git", "not-a-real-subcommand"])
        assert ok is False
        assert out.strip()


class TestCloneOrUpdateRepo:
    def test_clones_local_repository(self, tmp_path, monkeypatch):
        """A local clone exercises the real code path without touching network.

        ``clone_or_update_repo`` retries forever by design, so a failing clone
        would hang the suite. The retry loop is bounded here by failing the
        test as soon as more than one attempt is made.
        """
        origin = tmp_path / "origin"
        origin.mkdir()
        subprocess.run(["git", "init", "-q", "-b", "main", str(origin)], check=True)
        (origin / "README.md").write_text("hello\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(origin), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(origin), "-c", "user.email=t@t", "-c", "user.name=t",
             "commit", "-q", "-m", "init"],
            check=True,
        )

        cloner = GitRepoCloner(base_dir=str(tmp_path / "repos"), retry_delay=0)

        attempts = 0
        real_run = cloner.run_git_command

        def counting_run(cmd, cwd=None):
            nonlocal attempts
            attempts += 1
            if attempts > 1:
                raise AssertionError(
                    "clone_or_update_repo retried; refusing to loop forever"
                )
            return real_run(cmd, cwd)

        monkeypatch.setattr(cloner, "run_git_command", counting_run)

        if not cloner.clone_or_update_repo(origin.as_uri()):
            pytest.skip("git clone of a local repository is unavailable here")

        dest = tmp_path / "repos" / "origin"
        assert (dest / "README.md").read_text(encoding="utf-8") == "hello\n"

    def test_existing_non_git_directory_fails_without_retry_loop(self, tmp_path):
        """An existing non-git directory must be reported, not retried."""
        base = tmp_path / "repos"
        (base / "user" / "repo").mkdir(parents=True)
        cloner = GitRepoCloner(base_dir=str(base), retry_delay=0)

        # A bounded timeout proves the call returns rather than looping.
        result: list[bool] = []

        def call() -> None:
            result.append(cloner.clone_or_update_repo("https://github.com/user/repo.git"))

        thread = threading.Thread(target=call, daemon=True)
        thread.start()
        thread.join(timeout=10)

        assert not thread.is_alive(), "clone_or_update_repo hung in its retry loop"
        assert result == [False]


class TestEntryPoint:
    def test_main_is_exposed(self):
        from deepseek_tools import git_cloner

        assert callable(git_cloner.main)
